import json

import pytest

from hpc_inference_sizing import campaign, chain
from hpc_inference_sizing.campaign import Limits, Matrix

TARGETS = [1.0, 1.5, 2.0, 3.0]


def base(tmp_path):
    return Matrix(
        name="a40-x2",
        results_dir=tmp_path / "results",
        server_url="",
        served_model="",
        server_command="",
        units={"tp": "--tp 2", "ep": "--tp 2 --ep", "pp": "--pp 2", "dp_ep": "--dp 2 --ep"},
        profiles=["rag", "chat"],
        targets_s=TARGETS,
    )


def write(results, run_id, caps):
    results.mkdir(parents=True, exist_ok=True)
    doc = {"sweep": {"capacity_rps": {str(t): c for t, c in zip(TARGETS, caps, strict=True)}}}
    (results / f"{run_id}.json").write_text(json.dumps(doc))


# capacities (1, 1.5, 2, 3 s) the fake runner writes for each run
CAPS = {
    "tp-rag": [None, 0.25, 0.5, 0.5],
    "ep-rag": [None, 0.25, 0.375, 0.5],
    "pp-rag": [None, None, 0.5, 0.75],  # ties tp at 2 s, better at 3 s: pp is the best RAG unit
    "dp_ep-rag": [None, 0.25, 0.25, 0.5],
    "tp-chat": [1.0, 2.0, 2.0, 4.0],
    "ep-chat": [1.0, 2.0, 3.0, 4.0],
    "pp-chat": [0.5, 1.0, 2.0, 2.0],
    "dp_ep-chat": [None, 1.0, 2.0, 3.0],
    "pp_seqs64-rag": [None, None, 0.625, 0.75],  # helps
    "pp_batch2048-rag": [None, None, 0.6, 0.75],  # helps
    "pp_noprefix-rag": [None, None, 0.25, 0.5],  # hurts
}


def fake_runner(ran):
    def runner(matrix, run_id, meta, log):
        ran.append(run_id)
        if run_id in matrix.fixed_rates:
            matrix.results_dir.mkdir(parents=True, exist_ok=True)
            steps = [{"rate": r} for _ in range(matrix.repeat) for r in matrix.fixed_rates[run_id]]
            (matrix.results_dir / f"{run_id}.json").write_text(json.dumps({"steps": steps}))
        else:
            write(matrix.results_dir, run_id, CAPS.get(run_id, [None, None, 0.5, 0.5]))
        return True

    return runner


def test_best_unit_uses_2_s_then_breaks_ties_with_looser_targets(tmp_path):
    m = base(tmp_path)
    for run_id, caps in CAPS.items():
        write(m.results_dir, run_id, caps)
    assert chain.best_unit(m, list(m.units), "rag") == ("pp", 0.5, 2.0)
    assert chain.best_unit(m, list(m.units), "chat") == ("ep", 3.0, 2.0)


def test_best_unit_falls_back_to_a_looser_target_when_2_s_is_unreachable(tmp_path):
    m = base(tmp_path)
    for unit in m.units:
        write(m.results_dir, f"{unit}-rag", [None, None, None, 0.25 if unit == "pp" else 0.125])
    assert chain.best_unit(m, list(m.units), "rag") == ("pp", 0.25, 3.0)


def test_the_whole_chain_runs_each_phase_on_the_rules(tmp_path):
    m = base(tmp_path)
    ran = []
    phases = chain.run_chain(m, Limits(), {}, log=lambda _: None, runner=fake_runner(ran))
    assert ran[:8] == m.run_ids()
    levers = [f"pp_{n}-rag" for opts in chain.LEVERS.values() for n, _ in opts]
    assert ran[8:15] == levers
    assert ran[15] == "pp_combo-rag"
    assert ran[16:] == ["repeat_pp-rag", "repeat_ep-chat"]
    assert set(phases) == {"base", "levers", "combo", "repeats"}

    lever_plan = json.loads((m.results_dir / "plan-levers.json").read_text())
    assert lever_plan["best_unit"] == "pp" and lever_plan["start_rate"] == pytest.approx(0.125)
    combo = json.loads((m.results_dir / "plan-combo.json").read_text())
    assert set(combo["chosen"]) == {"max_num_seqs", "max_num_batched_tokens"}  # noprefix hurt
    assert combo["chosen"]["max_num_seqs"]["flags"] == "--max-num-seqs 64"
    repeats = json.loads((m.results_dir / "plan-repeats.json").read_text())
    assert repeats["runs"]["repeat_pp-rag"]["rates"] == [0.5, 0.75]
    assert repeats["runs"]["repeat_ep-chat"]["rates"] == [1.0, 2.0, 3.0, 4.0]
    steps = json.loads((m.results_dir / "repeat_ep-chat.json").read_text())["steps"]
    assert len(steps) == 3 * 4


def test_the_lever_runs_carry_the_best_units_flags(tmp_path):
    m = base(tmp_path)
    for run_id, caps in CAPS.items():
        write(m.results_dir, run_id, caps)
    matrix, _ = chain.lever_phase(m)
    assert matrix.units["pp_seqs32"] == "--pp 2 --max-num-seqs 32"
    assert matrix.units["pp_noprefix"] == "--pp 2 --no-enable-prefix-caching"
    assert matrix.run_ids() == [f"{u}-rag" for u in matrix.units]


def test_no_helpful_lever_means_no_combination_run(tmp_path):
    m = base(tmp_path)
    for run_id, caps in CAPS.items():
        write(m.results_dir, run_id, caps)
    for name in ("pp_seqs64", "pp_batch2048"):
        write(m.results_dir, f"{name}-rag", [None, None, 0.5, 0.75])  # equal is not better
    _, plan = chain.lever_phase(m)
    matrix, combo = chain.combo_phase(m, plan)
    assert matrix is None and combo["chosen"] == {} and "defaults" in combo["note"]


def test_the_cap_drops_the_later_phases_first(tmp_path):
    m = base(tmp_path)
    ran = []

    class Capped(Limits):
        def reason_to_stop(self):
            return "budget reached" if len(ran) >= 8 else None

    phases = chain.run_chain(m, Capped(), {}, log=lambda _: None, runner=fake_runner(ran))
    assert ran == m.run_ids() and set(phases) == {"base"}  # levers and repeats dropped


def test_a_rerun_resumes_without_redoing_finished_runs(tmp_path):
    m = base(tmp_path)
    first, second = [], []
    chain.run_chain(m, Limits(), {}, log=lambda _: None, runner=fake_runner(first))
    chain.run_chain(m, Limits(), {}, log=lambda _: None, runner=fake_runner(second))
    assert second == []


def test_run_one_measures_fixed_rates_for_a_repeat_run(tmp_path, monkeypatch):
    from hpc_inference_sizing.bench import data, runner, workload
    from hpc_inference_sizing.bench.stats import StepSummary

    class FakeBench:
        def __init__(self, *a, **kw):
            self.steps = []

        async def warm_up(self, rate, duration_s):
            return 0

        async def step(self, rate):
            self.steps.append({"summary": {"rate": rate}})
            return StepSummary(rate, 1, 10, 0, 0.5, 0.6, 0.01, 0.02, 100)

    monkeypatch.setattr(runner, "Bench", FakeBench)
    monkeypatch.setattr(runner, "metadata", lambda bench, meta: meta)
    monkeypatch.setattr(data, "load_piaf", lambda: [])
    monkeypatch.setattr(data, "MistralTokenizer", lambda: None)
    monkeypatch.setattr(workload, "Workload", lambda *a, **kw: None)
    monkeypatch.setattr(campaign, "wait_healthy", lambda *a: 1.0)
    m = base(tmp_path)
    rep = Matrix(
        **{
            **m.__dict__,
            "units": {"repeat_tp": "--tp 2"},
            "run_list": ("repeat_tp-rag",),
            "fixed_rates": {"repeat_tp-rag": [0.5, 0.75]},
            "repeat": 3,
        }
    )
    logs = []
    assert campaign.run_one(rep, "repeat_tp-rag", {}, log=logs.append)
    doc = json.loads((rep.results_dir / "repeat_tp-rag.json").read_text())
    assert [s["summary"]["rate"] for s in doc["steps"]] == [0.5, 0.75] * 3
    assert "sweep" not in doc and doc["metadata"]["mode"] == "repeat"
    assert sum("step 0.5 req/s" in line for line in logs) == 3
