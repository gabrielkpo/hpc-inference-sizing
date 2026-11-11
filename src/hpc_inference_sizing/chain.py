"""The whole 2-GPU matrix in one session (protocol §5.2, §6.6, §10 and amendment A5).

The levers and the repeats depend on the base results, so they cannot be written before the
campaign. The chain derives them from the run files with fixed rules and runs every phase on
the same node, in this order:

1. **Base**: every unit x profile of the matrix (§5.1).
2. **Levers** (§5.2): on the best unit (highest capacity, RAG profile, T = 2 s), one lever at a
   time from vLLM's defaults; each lever ladder starts two steps below that unit's baseline
   capacity (a quarter of it, the ladder doubling at each step).
3. **Best combination** (§5.2): the value of each lever that raised the T = 2 s capacity over the
   baseline, combined, measured once with a full ladder (it reads every target). Skipped when no
   lever helped: the best combination is then vLLM's defaults, already measured.
4. **Repeats** (§6.6): on the best unit of each profile, its capacity point for each target
   measured 3 more times at that rate.

The order is also the cut order of §10: when the time or budget cap is reached, the repeats
are lost first, then lever values, never the base ladders. Each phase's choices are written to
`plan-<phase>.json` in the results folder. Rerunning the chain resumes: finished runs are kept.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from hpc_inference_sizing.campaign import Limits, Matrix, Status, run_campaign, run_one

LEVER_PROFILE = "rag"
LEVER_TARGET = 2.0
# Protocol §5.2, one lever at a time; vLLM 0.32 enables prefix caching by default.
LEVERS: dict[str, list[tuple[str, str]]] = {
    "max_num_seqs": [
        ("seqs32", "--max-num-seqs 32"),
        ("seqs64", "--max-num-seqs 64"),
        ("seqs128", "--max-num-seqs 128"),
    ],
    "max_num_batched_tokens": [
        ("batch2048", "--max-num-batched-tokens 2048"),
        ("batch8192", "--max-num-batched-tokens 8192"),
        ("batch16384", "--max-num-batched-tokens 16384"),
    ],
    "prefix_caching": [("noprefix", "--no-enable-prefix-caching")],
}
REPEATS = 3


def capacity(results_dir: Path, run_id: str, target: float) -> float | None:
    """Capacity of a finished run at one target; None if absent, failed or unreachable."""
    try:
        caps = json.loads((results_dir / f"{run_id}.json").read_text())["sweep"]["capacity_rps"]
        value = caps.get(str(float(target)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return float(value) if value else None


def best_unit(
    matrix: Matrix, units: list[str], profile: str, target: float = LEVER_TARGET
) -> tuple[str, float, float] | None:
    """(unit, capacity, target used): highest capacity at `target`, ties broken by the
    looser targets; if no unit reaches `target`, the next looser target is used."""
    looser = sorted((t for t in matrix.targets_s if t >= target), key=float)
    for t in looser or sorted(matrix.targets_s, reverse=True):
        scored = []
        for unit in units:
            cap = capacity(matrix.results_dir, f"{unit}-{profile}", t)
            if cap:
                ties = [
                    capacity(matrix.results_dir, f"{unit}-{profile}", u) or 0.0
                    for u in sorted(matrix.targets_s, reverse=True)
                ]
                scored.append((cap, ties, -units.index(unit), unit))
        if scored:
            cap, _, _, unit = max(scored)
            return unit, cap, t
    return None


def lever_phase(base: Matrix) -> tuple[Matrix, dict] | None:
    best = best_unit(base, list(base.units), LEVER_PROFILE)
    if best is None:
        return None
    unit, cap, target = best
    units = {
        f"{unit}_{name}": f"{base.units[unit]} {flags}".strip()
        for options in LEVERS.values()
        for name, flags in options
    }
    plan = {
        "best_unit": unit,
        "baseline_capacity": cap,
        "target_s": target,
        "start_rate": cap / 4,
        "runs": [f"{u}-{LEVER_PROFILE}" for u in units],
    }
    matrix = replace(
        base,
        units=units,
        profiles=[LEVER_PROFILE],
        start_rate=cap / 4,
        run_list=None,
        fixed_rates={},
    )
    return matrix, plan


def combo_phase(base: Matrix, lever_plan: dict) -> tuple[Matrix | None, dict]:
    unit, baseline, target = (
        lever_plan["best_unit"],
        lever_plan["baseline_capacity"],
        lever_plan["target_s"],
    )
    chosen: dict[str, tuple[str, float]] = {}
    for lever, options in LEVERS.items():
        scored = [
            (capacity(base.results_dir, f"{unit}_{name}-{LEVER_PROFILE}", target) or 0.0, flags)
            for name, flags in options
        ]
        cap, flags = max(scored)
        if cap > baseline:
            chosen[lever] = (flags, cap)
    plan = {
        "best_unit": unit,
        "baseline_capacity": baseline,
        "target_s": target,
        "chosen": {k: {"flags": f, "capacity": c} for k, (f, c) in chosen.items()},
    }
    if not chosen:
        plan["note"] = "no lever raised the capacity: the best combination is vLLM's defaults"
        return None, plan
    flags = " ".join([base.units[unit]] + [f for f, _ in chosen.values()])
    matrix = replace(
        base,
        units={f"{unit}_combo": flags},
        profiles=[LEVER_PROFILE],
        start_rate=baseline / 4,
        run_list=None,
        fixed_rates={},
    )
    plan["runs"] = [f"{unit}_combo-{LEVER_PROFILE}"]
    return matrix, plan


def repeat_phase(base: Matrix) -> tuple[Matrix, dict] | None:
    units: dict[str, str] = {}
    runs: list[str] = []
    rates: dict[str, list[float]] = {}
    plan: dict = {"repeat": REPEATS, "runs": {}}
    for profile in base.profiles:
        best = best_unit(base, list(base.units), profile)
        if best is None:
            continue
        unit = best[0]
        points = sorted(
            {c for t in base.targets_s if (c := capacity(base.results_dir, f"{unit}-{profile}", t))}
        )
        run_id = f"repeat_{unit}-{profile}"
        units[f"repeat_{unit}"] = base.units[unit]
        runs.append(run_id)
        rates[run_id] = points
        plan["runs"][run_id] = {"unit": unit, "rates": points}
    if not runs:
        return None
    matrix = replace(base, units=units, run_list=tuple(runs), fixed_rates=rates, repeat=REPEATS)
    return matrix, plan


def run_chain(
    base: Matrix,
    limits: Limits,
    meta: dict[str, str],
    log: Callable[[str], None] = print,
    runner: Callable = run_one,
) -> dict[str, Status]:
    """Run the four phases on this node; returns the final status of each phase run."""
    done: dict[str, Status] = {}

    def save(name: str, plan: dict) -> None:
        base.results_dir.mkdir(parents=True, exist_ok=True)
        (base.results_dir / f"plan-{name}.json").write_text(json.dumps(plan, indent=1))
        log(f"chain: {name} plan: {json.dumps(plan)}")

    def phase(name: str, matrix: Matrix) -> bool:
        log(f"chain: phase {name}, {len(matrix.run_ids())} runs")
        done[name] = run_campaign(matrix, limits, meta, log=log, runner=runner)
        if reason := limits.reason_to_stop():
            log(f"chain: stopping after {name}: {reason}")
            return False
        return True

    if not phase("base", base):
        return done
    levers = lever_phase(base)
    if levers is None:
        log("chain: no unit has a capacity on the RAG profile; levers and combination skipped")
    else:
        lever_matrix, lever_plan = levers
        save("levers", lever_plan)
        if not phase("levers", lever_matrix):
            return done
        combo_matrix, combo_plan = combo_phase(base, lever_plan)
        save("combo", combo_plan)
        if combo_matrix is not None and not phase("combo", combo_matrix):
            return done
    repeats = repeat_phase(base)
    if repeats is None:
        log("chain: no capacity to repeat")
        return done
    repeat_matrix, repeat_plan = repeats
    save("repeats", repeat_plan)
    phase("repeats", repeat_matrix)
    return done
