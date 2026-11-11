import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from hpc_inference_sizing import campaign
from hpc_inference_sizing.campaign import (
    Limits,
    load_matrix,
    run_campaign,
    run_one,
    status,
    stop_server,
)

REPO = Path(__file__).resolve().parent.parent


def write_matrix(
    tmp_path, command="", url="http://127.0.0.1:9", health_timeout=0.5, sync_command=""
):
    (tmp_path / "campaign").mkdir()
    path = tmp_path / "campaign" / "m.toml"
    path.write_text(
        f"""
name = "test"
results_dir = "results/test"
model = "toy"
sync_command = "{sync_command}"
profiles = ["rag", "chat"]
targets_s = [1.0, 3.0]

[server]
url = "{url}"
served_model = "toy"
health_timeout_s = {health_timeout}
command = "{command}"

[units.tp]
flags = "--tp 2"

[units.dp_ep]
flags = "--dp 2"
"""
    )
    return load_matrix(path)


def test_real_matrix_lists_the_protocol_runs():
    m = load_matrix(REPO / "campaign" / "a40-x2.toml")
    assert m.run_ids() == [f"{u}-{p}" for u in ("tp", "ep", "pp", "dp_ep") for p in ("rag", "chat")]
    assert m.results_dir == REPO / "results" / "a40-x2"
    cmd = m.command_for("pp")
    assert cmd[:3] == ["vllm", "serve", "mistralai/Mistral-Small-4-119B-2603-NVFP4"]
    assert "--pipeline-parallel-size" in cmd and "TRITON_MLA" in cmd


def test_run_ids_round_trip_with_dashes_and_underscores(tmp_path):
    m = write_matrix(tmp_path)
    assert m.split("dp_ep-chat") == ("dp_ep", "chat")
    with pytest.raises(ValueError):
        m.split("zz-rag")


def test_campaign_resumes_and_skips_finished_runs(tmp_path):
    m = write_matrix(tmp_path)
    m.results_dir.mkdir(parents=True)
    (m.results_dir / "tp-rag.json").write_text("{}")  # finished before an interruption
    ran = []

    def fake_runner(matrix, run_id, meta, log):
        ran.append(run_id)
        (matrix.results_dir / f"{run_id}.json").write_text("{}")
        return True

    final = run_campaign(m, Limits(), {}, log=lambda _: None, runner=fake_runner)
    assert ran == ["tp-chat", "dp_ep-rag", "dp_ep-chat"]
    assert final.pending == [] and len(final.done) == 4


def test_campaign_stops_at_the_time_limit(tmp_path):
    m = write_matrix(tmp_path)
    limits = Limits(max_hours=1.0, started_at=time.monotonic() - 3601)
    ran, logs = [], []
    run_campaign(m, limits, {}, log=logs.append, runner=lambda *a: ran.append(a[1]))
    assert ran == [] and "time limit" in logs[0]


def test_spending_cap_uses_the_instance_price():
    limits = Limits(budget_usd=1.0, price_per_hour_usd=2.0, started_at=time.monotonic() - 1800)
    assert limits.spent_usd() == pytest.approx(1.0, abs=0.01)
    assert "budget" in limits.reason_to_stop()


def test_a_unit_that_does_not_start_is_recorded_as_failed(tmp_path):
    # `false` exits at once: the runner records the failure instead of sweeping
    m = write_matrix(tmp_path, command="false", health_timeout=30)
    assert run_one(m, "tp-rag", {}, log=lambda _: None) is False
    failure = json.loads((m.results_dir / "tp-rag.failed.json").read_text())
    assert failure["unit"] == "tp" and "exited" in failure["error"]
    assert status(m).failed == ["tp-rag"]


def test_retry_failed_reruns_only_failed_runs(tmp_path):
    m = write_matrix(tmp_path)
    m.results_dir.mkdir(parents=True)
    for run_id in ("tp-rag", "tp-chat", "dp_ep-rag"):
        (m.results_dir / f"{run_id}.json").write_text("{}")
    (m.results_dir / "dp_ep-chat.failed.json").write_text("{}")
    ran = []
    run_campaign(
        m,
        Limits(),
        {},
        retry_failed=True,
        log=lambda _: None,
        runner=lambda matrix, run_id, meta, log: ran.append(run_id),
    )
    assert ran == ["dp_ep-chat"]


def test_the_campaign_clock_adds_up_across_processes(tmp_path):
    # a SLURM array runs each run in its own process: the caps must hold for the whole array
    clock = tmp_path / "campaign-clock.json"
    first = Limits(max_hours=1.0, clock_file=clock, started_at=time.monotonic() - 3000)
    assert first.reason_to_stop() is None
    first.save()
    second = Limits(max_hours=1.0, clock_file=clock, started_at=time.monotonic() - 700)
    assert "time limit" in second.reason_to_stop()


def test_run_campaign_saves_the_clock_after_each_run(tmp_path):
    m = write_matrix(tmp_path)
    clock = tmp_path / "clock.json"
    run_campaign(m, Limits(clock_file=clock), {}, log=lambda _: None, runner=lambda *a: True)
    assert json.loads(clock.read_text())["elapsed_s"] >= 0


def test_a_budget_without_a_price_is_refused():
    with pytest.raises(ValueError, match="price"):
        Limits(budget_usd=15.0)


def test_a_crash_during_the_sweep_is_recorded_and_does_not_raise(tmp_path, monkeypatch):
    m = write_matrix(tmp_path)
    monkeypatch.setattr(campaign, "wait_healthy", lambda *a: 0.0)

    def broken_piaf():
        raise RuntimeError("server went away")

    monkeypatch.setattr("hpc_inference_sizing.bench.data.load_piaf", broken_piaf)
    assert run_one(m, "tp-rag", {}, log=lambda _: None) is False
    failure = json.loads((m.results_dir / "tp-rag.failed.json").read_text())
    assert failure["stage"] == "sweep" and "server went away" in failure["error"]


def test_results_leave_the_node_even_after_a_failed_run(tmp_path):
    m = write_matrix(
        tmp_path, command="false", health_timeout=30, sync_command="touch {results_dir}/synced"
    )
    run_one(m, "tp-rag", {}, log=lambda _: None)
    assert (m.results_dir / "synced").exists()


def test_stop_server_also_stops_the_workers(tmp_path):
    pid_file = tmp_path / "worker.pid"
    proc = subprocess.Popen(
        ["sh", "-c", f"sleep 300 & echo $! > {pid_file}; wait"], start_new_session=True
    )
    for _ in range(50):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.1)
    worker = int(pid_file.read_text())
    stop_server(proc)
    assert proc.returncode is not None
    for _ in range(50):
        try:
            os.kill(worker, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail("the worker outlived stop_server")


def test_the_clock_is_saved_when_a_run_is_interrupted(tmp_path):
    m = write_matrix(tmp_path)
    clock = tmp_path / "clock.json"

    def interrupted(*_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_campaign(m, Limits(clock_file=clock), {}, log=lambda _: None, runner=interrupted)
    assert "elapsed_s" in json.loads(clock.read_text())


def test_an_unreadable_clock_is_a_clear_error(tmp_path):
    clock = tmp_path / "clock.json"
    clock.write_text("{}")
    with pytest.raises(ValueError, match="unreadable campaign clock"):
        Limits(clock_file=clock)


def test_progress_line_after_each_run(tmp_path):
    m = write_matrix(tmp_path)
    logs = []

    def runner(matrix, run_id, meta, log):
        matrix.results_dir.mkdir(parents=True, exist_ok=True)
        (matrix.results_dir / f"{run_id}.json").write_text("{}")

    run_campaign(m, Limits(price_per_hour_usd=1.5), {}, log=logs.append, runner=runner)
    progress = [line for line in logs if line.startswith("progress:")]
    assert progress[0].startswith("progress: 1/4 runs (1 done, 0 failed)")
    assert progress[-1].startswith("progress: 4/4 runs") and "spent so far" in progress[-1]


def test_step_line_says_which_targets_a_step_meets(tmp_path):
    from hpc_inference_sizing.bench.stats import StepSummary

    m = write_matrix(tmp_path)
    s = StepSummary(0.5, 60, 33, 0, 0.8, 1.99, 0.028, 0.037, 157.6)
    assert campaign.step_line(s, m) == (
        "step 0.5 req/s: 33 requests, 0 errors, TTFT p95 1.99 s, TPOT p95 37 ms; meets 3 s"
    )
    slow = StepSummary(1.0, 60, 61, 0, 2.2, 3.5, 0.11, 0.158, 265.0)
    assert campaign.step_line(slow, m).endswith("meets no target")
