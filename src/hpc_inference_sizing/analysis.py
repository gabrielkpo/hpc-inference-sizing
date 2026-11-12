"""Read a campaign's run files and summarize them as the frozen protocol defines (§4, amendments
A8 and A9): capacities recomputed from every step, the rate actually sent, what broke first,
the KV cache vLLM reported, and how the repeated steps came out.

The raw run files (15-40 MB each) stay out of Git; the summary this module writes is small and
is what the published results are built from.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

TPOT_BOUND_S = 1 / 15  # protocol §4: at least 15 tokens per second per user
KV_LINE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def meets(step: dict, target_s: float, tpot_bound_s: float = TPOT_BOUND_S) -> bool:
    if step["n_requests"] == 0 or step["n_errors"] > 0:
        return False
    return step["ttft_p95"] <= target_s and step["tpot_p95"] <= tpot_bound_s


def what_broke(step: dict, target_s: float, tpot_bound_s: float = TPOT_BOUND_S) -> str:
    reasons = []
    if step["n_errors"]:
        reasons.append("errors")
    if step["ttft_p95"] > target_s:
        reasons.append("TTFT")
    if step["tpot_p95"] > tpot_bound_s:
        reasons.append("TPOT")
    return "+".join(reasons) or "none"


def achieved(step: dict) -> float:
    return step["n_requests"] / step["duration_s"] if step["duration_s"] else 0.0


def capacity(steps: list[dict], target_s: float, tpot_bound_s: float = TPOT_BOUND_S) -> dict:
    """Highest nominal rate whose step meets the SLO (§4), with the rate actually sent there and
    the next step above it that failed (the bracket) and why."""
    passing = [s for s in steps if meets(s, target_s, tpot_bound_s)]
    if not passing:
        return {"capacity": None, "achieved": None, "next_fail": None, "broke_on": None}
    best = max(passing, key=lambda s: s["rate"])
    above = [s for s in steps if s["rate"] > best["rate"] and not meets(s, target_s, tpot_bound_s)]
    nxt = min(above, key=lambda s: s["rate"]) if above else None
    return {
        "capacity": best["rate"],
        "achieved": round(achieved(best), 4),
        "next_fail": nxt["rate"] if nxt else None,
        "broke_on": what_broke(nxt, target_s, tpot_bound_s) if nxt else None,
    }


def kv_cache_tokens(server_log: Path) -> int | None:
    """Total KV cache vLLM reported: one line per engine (two for DP2+EP), summed."""
    if not server_log.exists():
        return None
    sizes = [int(m.group(1).replace(",", "")) for m in KV_LINE.finditer(server_log.read_text())]
    return sum(sizes) or None


def compact_step(step: dict) -> dict:
    keys = ("n_requests", "n_errors", "ttft_p50", "ttft_p95", "tpot_p50", "tpot_p95")
    out = {"rate": step["rate"]}  # exact: capacities refer to it
    out.update({k: (round(step[k], 4) if isinstance(step[k], float) else step[k]) for k in keys})
    out["achieved"] = round(achieved(step), 4)
    return out


def summarize_run(path: Path, targets_s: list[float]) -> dict:
    doc = json.loads(path.read_text())
    meta = doc["metadata"]
    if "quality" in doc:  # protocol §9: answer quality, no load steps
        q = doc["quality"]
        return {
            "run_id": meta["run_id"],
            "unit": meta["unit"],
            "profile": "quality",
            "mode": "quality",
            "server_flags": meta["server_flags"],
            "git_commit": meta.get("git_commit"),
            "quality": {k: q.get(k) for k in ("n", "exact_match", "f1", "truncated")},
        }
    steps = [s["summary"] for s in doc["steps"]]
    run = {
        "run_id": meta["run_id"],
        "unit": meta["unit"],
        "profile": meta["profile"],
        "server_flags": meta["server_flags"],
        "cold_start_s": meta.get("cold_start_s"),
        "git_commit": meta.get("git_commit"),
        "kv_cache_tokens": kv_cache_tokens(path.parent / "logs" / f"{meta['run_id']}.server.log"),
        "steps": [compact_step(s) for s in steps],
    }
    if meta.get("mode") == "repeat":
        run["mode"] = "repeat"
        run["repeat_pass"] = {
            f"{t:g}": {
                f"{r:g}": [meets(s, t) for s in steps if s["rate"] == r] for r in meta["rates"]
            }
            for t in targets_s
        }
    else:
        run["capacity"] = {f"{t:g}": capacity(steps, t) for t in targets_s}
    return run


def summarize_dir(results_dir: Path, targets_s: list[float]) -> dict:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith(("plan-", "campaign-clock")) or path.name.endswith(".failed.json"):
            continue
        runs.append(summarize_run(path, targets_s))
    plans = {p.stem: json.loads(p.read_text()) for p in sorted(results_dir.glob("plan-*.json"))}
    return {"targets_s": targets_s, "tpot_bound_s": TPOT_BOUND_S, "runs": runs, "plans": plans}


def units_needed(peak_rps: float, capacity_rps: float | None) -> int | None:
    """§8: units = ceil(peak / capacity); None when the target is unreachable (§4)."""
    if not capacity_rps:
        return None
    return math.ceil(peak_rps / capacity_rps)


def arrival_rate(users: int, response_s: float, think_s: float = 60.0) -> float:
    """§3.2, Little's law: N active users send N / (R + Z) requests per second."""
    return users / (response_s + think_s)


def capacity_table(summary: dict, profile: str, units: list[str]) -> str:
    """Markdown table of one profile for the given units: capacity (req/s), the rate actually
    sent at that step, and the next step above it that failed and why."""
    targets = summary["targets_s"]
    head = "| Unit | " + " | ".join(f"T = {t:g} s" for t in targets) + " | KV cache (tokens) |"
    rows = [head, "|---" * (len(targets) + 2) + "|"]
    for run in summary["runs"]:
        if run["profile"] != profile or "capacity" not in run or run["unit"] not in units:
            continue
        cells = []
        for t in targets:
            c = run["capacity"][f"{t:g}"]
            if c["capacity"] is None:
                cells.append("unreachable")
            else:
                broke = (
                    f", next {c['next_fail']:g} broke on {c['broke_on']}" if c["next_fail"] else ""
                )
                cells.append(f"{c['capacity']:g} (sent {c['achieved']:.3g}{broke})")
        kv = f"{run['kv_cache_tokens']:,}" if run["kv_cache_tokens"] else "-"
        rows.append(f"| {run['unit']} | " + " | ".join(cells) + f" | {kv} |")
    return "\n".join(rows)
