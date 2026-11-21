"""The sizing tool: turn the measured capacities into the two clients' answers (protocol §3.2,
§8), with a range rather than one number.

For a target T and a profile, a serving unit (2 GPUs by default; `--units two --gpus-per-unit 4`
sizes the 4-GPU pair behind a load balancer) has two capacities:
- optimistic: the best capacity any unit reached in a single run (results summary);
- prudent: the highest rate that met T in every repeat of that profile (§6.6), when repeats
  exist; otherwise the optimistic one is all we have, and the tool says so.

Users send N / (R + Z) requests per second (Little's law, §3.2), with R the response time
measured at that capacity (TTFT p50 + output tokens x TPOT p50) and Z the think time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hpc_inference_sizing.analysis import rate_key

HOURS_PER_MONTH = 730
GPUS_PER_UNIT = 2
BASE_UNITS = ("tp", "ep", "pp", "dp_ep")  # the protocol's 2-GPU units (§5.1), defaults only


@dataclass(frozen=True)
class Point:
    """One capacity of a serving unit: requests per second and the response time there."""

    capacity: float
    response_s: float
    source: str


@dataclass(frozen=True)
class Capacity:
    optimistic: Point | None  # None: no unit met the target (unreachable)
    prudent: Point | None  # None: no repeat passed, or no repeat for this profile
    prudent_note: str
    optimistic_repeats: str = ""  # how the optimistic point fared when repeated, if it was


def _response_s(step: dict, output_tokens: int) -> float:
    return step["ttft_p50"] + output_tokens * step["tpot_p50"]


def _step(run: dict, rate: float) -> dict:
    return next(s for s in run["steps"] if s["rate"] == rate)


def capacity(
    summary: dict,
    profile: str,
    target_s: float,
    output_tokens: int = 300,
    units: tuple[str, ...] = BASE_UNITS,
) -> Capacity:
    key = f"{target_s:g}"
    measured = [f"{t:g}" for t in summary.get("targets_s", [])]
    if measured and key not in measured:
        raise ValueError(
            f"target {key} s was not measured; this summary has {', '.join(measured)} s"
        )
    base = [
        r
        for r in summary["runs"]
        if r["profile"] == profile and "capacity" in r and r["unit"] in units
    ]
    if not base:
        raise ValueError(f"no {profile} run of unit {', '.join(units)} in this summary")
    best = None
    for run in base:
        c = run["capacity"].get(key, {}).get("capacity")
        if c and (best is None or c > best[0]):
            best = (c, run)
    optimistic = None
    if best:
        rate, run = best
        optimistic = Point(rate, _response_s(_step(run, rate), output_tokens), run["unit"])

    prudent, note, opt_repeats = None, f"no repeat of the {profile} profile", ""
    for run in summary["runs"]:
        if run.get("mode") != "repeat" or run["profile"] != profile:
            continue
        passes = run["repeat_pass"].get(key, {})
        if optimistic and run["unit"] == f"repeat_{optimistic.source}":
            done = passes.get(rate_key(optimistic.capacity))
            if done:
                opt_repeats = f"passed {sum(done)} of {len(done)} repeats"
        always = [float(r) for r, results in passes.items() if results and all(results)]
        unit = run["unit"].removeprefix("repeat_")
        if always:
            rate = max(always)
            if prudent is not None and prudent.capacity >= rate:
                continue  # several repeated units: keep the highest rate held every time
            steps = [s for s in run["steps"] if s["rate"] == rate]
            if not steps:
                continue  # a key that matches no step: nothing to measure the response time on
            r_s = sum(_response_s(s, output_tokens) for s in steps) / len(steps)
            prudent, note = Point(rate, r_s, f"{unit}, passed every repeat"), ""
        elif prudent is None:
            lowest = min((float(r) for r in passes), default=None)
            tried = f" (lowest tried: {lowest:g} req/s)" if lowest is not None else ""
            note = f"no repeated rate of {unit} met it{tried}"
    return Capacity(optimistic, prudent, note, opt_repeats)


def arrival_rate(users: int, response_s: float, think_s: float) -> float:
    return users / (response_s + think_s)


def gpus_for(users: int, point: Point, think_s: float, gpus_per_unit: int = GPUS_PER_UNIT) -> int:
    units = math.ceil(arrival_rate(users, point.response_s, think_s) / point.capacity)
    return units * gpus_per_unit


def users_for(gpus: int, point: Point, think_s: float, gpus_per_unit: int = GPUS_PER_UNIT) -> int:
    rate = (gpus // gpus_per_unit) * point.capacity
    return int(rate * (point.response_s + think_s))


def monthly_cost(gpus: int, price_per_gpu_hour: float) -> float:
    return gpus * price_per_gpu_hour * HOURS_PER_MONTH


def report(
    summary: dict,
    profile: str,
    target_s: float,
    think_s: float,
    price_per_gpu_hour: float,
    users: int | None = None,
    gpus: int | None = None,
    units: tuple[str, ...] = BASE_UNITS,
    gpus_per_unit: int = GPUS_PER_UNIT,
) -> str:
    if gpus_per_unit < 1 or think_s < 0 or price_per_gpu_hour < 0:
        raise ValueError("GPUs per unit must be positive; think time and price cannot be negative")
    if users is not None and users < 1:
        raise ValueError("the user count must be positive")
    if gpus is not None and gpus < gpus_per_unit:
        raise ValueError(f"a fleet needs at least one unit of {gpus_per_unit} GPUs")
    cap = capacity(summary, profile, target_s, units=units)
    lines = [
        f"Profile {profile}, first token within {target_s:g} s for 95 % of requests, "
        f"think time {think_s:g} s"
    ]
    if cap.optimistic is None:
        lines.append(
            f"Unreachable with these units ({', '.join(units)}): none met this target at any "
            "measured load, so adding more of them does not reach it. Other layouts may "
            "(4 GPUs behind a load balancer: --summary docs/results/a40-x4.json --units two "
            "--gpus-per-unit 4)"
        )
        return "\n".join(lines)
    points = [("optimistic", cap.optimistic)] + ([("prudent", cap.prudent)] if cap.prudent else [])
    for name, p in points:
        extra = ""
        if name == "optimistic" and cap.optimistic_repeats:
            extra = f", {cap.optimistic_repeats}"
        lines.append(
            f"Unit capacity ({name})  {p.capacity:.3g} req/s per {gpus_per_unit} GPUs "
            f"({p.source}{extra}), "
            f"answer in {p.response_s:.1f} s"
        )
    if cap.prudent is None:
        lines.append(f"Prudent bound          unknown: {cap.prudent_note}")
    if users is not None:
        for name, p in points:
            g = gpus_for(users, p, think_s, gpus_per_unit)
            rate = arrival_rate(users, p.response_s, think_s)
            lines.append(
                f"GPUs needed ({name})   {g} for {users:,} active users "
                f"({rate:.1f} req/s), {monthly_cost(g, price_per_gpu_hour):,.0f} USD "
                f"a month at {price_per_gpu_hour:g} USD per GPU-hour"
            )
    if gpus is not None:
        for name, p in points:
            served = users_for(gpus, p, think_s, gpus_per_unit)
            lines.append(f"Users served ({name})  about {served:,} active users on {gpus} GPUs")
    return "\n".join(lines)
