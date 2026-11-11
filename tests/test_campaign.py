import json
import time
from pathlib import Path

import pytest

from hpc_inference_sizing.campaign import Limits, load_matrix, run_campaign, run_one, status

REPO = Path(__file__).resolve().parent.parent


def write_matrix(tmp_path, command="", url="http://127.0.0.1:9", health_timeout=0.5):
    (tmp_path / "campaign").mkdir()
    path = tmp_path / "campaign" / "m.toml"
    path.write_text(
        f"""
name = "test"
results_dir = "results/test"
model = "toy"
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
