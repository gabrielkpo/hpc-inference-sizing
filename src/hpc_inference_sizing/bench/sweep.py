"""Capacity search (protocol §6): a load ladder, then a bisection per latency target.

1. Ladder: rates start, 2 x start, 4 x start, ... until the first step that breaks the loosest
   target. One ladder serves every target: each target reads its result from the same steps.
2. Bisection: for each target, `bisect_steps` steps between its last passing and first failing
   rate. The capacity is the highest passing rate.

The step itself is injected, so the search logic is tested without a server.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from hpc_inference_sizing.bench.stats import TPOT_BOUND_S, StepSummary

StepFn = Callable[[float], Awaitable[StepSummary]]


@dataclass
class SweepResult:
    targets_s: list[float]
    tpot_bound_s: float = TPOT_BOUND_S
    steps: list[StepSummary] = field(default_factory=list)
    capacity: dict[float, float | None] = field(default_factory=dict)  # target -> req/s
    reached_max_rate: bool = False  # the loosest target never broke within max_rate

    def to_dict(self) -> dict:
        return {
            "targets_s": self.targets_s,
            "tpot_bound_s": self.tpot_bound_s,
            "capacity_rps": {str(t): c for t, c in self.capacity.items()},
            "reached_max_rate": self.reached_max_rate,
            "steps": [s.to_dict() for s in self.steps],
        }


async def sweep(
    step: StepFn,
    targets_s: list[float],
    start_rate: float = 0.25,
    max_rate: float = 64.0,
    bisect_steps: int = 2,
    tpot_bound_s: float = TPOT_BOUND_S,
    on_step: Callable[[StepSummary], None] | None = None,
) -> SweepResult:
    """`on_step` is called with each step as soon as it is measured (live progress)."""
    result = SweepResult(targets_s=sorted(targets_s), tpot_bound_s=tpot_bound_s)
    by_rate: dict[float, StepSummary] = {}

    async def run(rate: float) -> StepSummary:
        if rate not in by_rate:  # never measure the same rate twice
            by_rate[rate] = await step(rate)
            result.steps.append(by_rate[rate])
            if on_step is not None:
                on_step(by_rate[rate])
        return by_rate[rate]

    loosest = max(targets_s)
    rate = start_rate
    while True:
        if not (await run(rate)).meets(loosest, tpot_bound_s):
            break
        if rate * 2 > max_rate:
            result.reached_max_rate = True
            break
        rate *= 2

    for target in result.targets_s:
        passing = [r for r, s in by_rate.items() if s.meets(target, tpot_bound_s)]
        if not passing:
            result.capacity[target] = None  # fails even at the lowest rate
            continue
        lo = max(passing)
        failing = [r for r, s in by_rate.items() if r > lo and not s.meets(target, tpot_bound_s)]
        if failing:
            hi = min(failing)
            for _ in range(bisect_steps):
                mid = (lo + hi) / 2
                if (await run(mid)).meets(target, tpot_bound_s):
                    lo = mid
                else:
                    hi = mid
        result.capacity[target] = lo
    return result
