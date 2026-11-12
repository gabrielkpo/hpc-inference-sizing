import json

from hpc_inference_sizing import analysis


def step(rate, n, ttft95, tpot95, errors=0):
    return {
        "summary": {
            "rate": rate,
            "duration_s": 180,
            "n_requests": n,
            "n_errors": errors,
            "ttft_p50": ttft95 / 2,
            "ttft_p95": ttft95,
            "tpot_p50": tpot95 / 2,
            "tpot_p95": tpot95,
        }
    }


def write_run(folder, run_id, steps, mode=None, rates=None):
    unit, profile = run_id.rsplit("-", 1)
    meta = {
        "run_id": run_id,
        "unit": unit,
        "profile": profile,
        "server_flags": "",
        "cold_start_s": 150.0,
        "git_commit": "abc",
    }
    if mode:
        meta.update(mode=mode, rates=rates)
    (folder / f"{run_id}.json").write_text(json.dumps({"metadata": meta, "steps": steps}))


def test_capacity_is_the_highest_passing_step_with_its_bracket(tmp_path):
    steps = [
        step(0.25, 44, 1.3, 0.02),
        step(0.5, 84, 1.99, 0.044),
        step(1.0, 148, 4.2, 0.146),
        step(0.75, 124, 2.6, 0.074),
        step(0.53125, 102, 1.77, 0.056),
    ]
    write_run(tmp_path, "tp-rag", steps)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "tp-rag.server.log").write_text("x GPU KV cache size: 237,280 tokens, y\n")
    run = analysis.summarize_dir(tmp_path, [1.0, 2.0])["runs"][0]
    assert run["capacity"]["1"]["capacity"] is None  # unreachable
    c = run["capacity"]["2"]
    assert c["capacity"] == 0.53125 and c["next_fail"] == 0.75 and c["broke_on"] == "TTFT+TPOT"
    assert c["achieved"] == round(102 / 180, 4)
    assert run["kv_cache_tokens"] == 237280


def test_dp_kv_caches_are_summed_and_repeats_scored(tmp_path):
    write_run(
        tmp_path,
        "repeat_tp-chat",
        [step(2.5, 450, 0.3, 0.064), step(2.5, 440, 0.3, 0.068)],
        mode="repeat",
        rates=[2.5],
    )
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "repeat_tp-chat.server.log").write_text(
        "GPU KV cache size: 91,728 tokens\nGPU KV cache size: 91,728 tokens\n"
    )
    run = analysis.summarize_dir(tmp_path, [1.0])["runs"][0]
    assert run["repeat_pass"] == {"1": {"2.5": [True, False]}}
    assert run["kv_cache_tokens"] == 183456


def test_sizing_helpers():
    assert analysis.arrival_rate(2000, 10.0) == 2000 / 70
    assert analysis.units_needed(28.6, 0.5) == 58
    assert analysis.units_needed(28.6, None) is None
