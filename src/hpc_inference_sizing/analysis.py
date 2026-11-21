"""Read a campaign's run files and summarize them as the frozen protocol defines (§4, amendments
A8 and A9): capacities recomputed from every step, the rate actually sent, what broke first,
the KV cache vLLM reported, and how the repeated steps came out.

Capacity stays the highest passing step (§4). Because pass or fail at one step is noisy, each
capacity also reports its margin to the bounds and the lowest step that failed, so a capacity
that sits above a failing step (non-monotone) is visible instead of hidden.

The raw run files (15-40 MB each) stay out of Git; the summary this module writes is small, is
what the published results are built from, and lists the sha256 of every file it read.

Run files come from a rented third-party node, so strings copied into the published summary are
checked against a restricted pattern first (identifiers, command-line tokens, hex commits).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

TPOT_BOUND_S = 1 / 15  # protocol §4: at least 15 tokens per second per user
KV_LINE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
IDENT = re.compile(r"[A-Za-z0-9_.+-]{1,64}")
TOKEN = re.compile(r"[A-Za-z0-9_.,:=/+-]{1,200}")  # one command-line token
COMMIT = re.compile(r"[0-9a-f]{7,40}")
PROVENANCE = ("created_at", "protocol_version", "server_version", "gpus", "model", "seed")


def rate_key(rate: float) -> str:
    """Key of a rate in the summary: exact for the bisection rates (2.578125, not 2.57812)."""
    return f"{rate:.10g}"


def _checked(value, pattern: re.Pattern, what: str):
    """Node-written text enters the published summary only if it matches `pattern`."""
    if value is None:
        return None
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"unexpected {what} in a run file; not copied into the summary")
    return value


def _command(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 2000:
        raise ValueError("unexpected server flags in a run file; not copied into the summary")
    for token in value.split():
        _checked(token, TOKEN, "server flags")
    return value


def _clean(value, where: str):
    """A plan file, kept only if made of numbers, booleans, identifiers and flag strings."""
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, str):
        return _command(value) if value.startswith("--") else _checked(value, IDENT, where)
    if isinstance(value, list):
        return [_clean(v, where) for v in value]
    if isinstance(value, dict):
        return {_checked(k, IDENT, where): _clean(v, where) for k, v in value.items()}
    raise ValueError(f"unexpected value in {where}; not copied into the summary")


def _source(path: Path) -> dict:
    data = path.read_bytes()
    return {"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


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
    """Highest nominal rate whose step meets the SLO (§4), with the rate actually sent there, the
    next step above it that failed (the bracket) and why, the margins of the passing step to the
    two bounds (seconds), and the lowest failing step: below the capacity, the cell is
    non-monotone and a lower load already failed the same target."""
    failing = [s for s in steps if not meets(s, target_s, tpot_bound_s)]
    lowest = min(failing, key=lambda s: s["rate"]) if failing else None
    passing = [s for s in steps if meets(s, target_s, tpot_bound_s)]
    if not passing:
        return {
            "capacity": None,
            "achieved": None,
            "next_fail": None,
            "broke_on": None,
            "lowest_fail": lowest["rate"] if lowest else None,
            "monotone": True,
        }
    best = max(passing, key=lambda s: s["rate"])
    above = [s for s in failing if s["rate"] > best["rate"]]
    nxt = min(above, key=lambda s: s["rate"]) if above else None
    return {
        "capacity": best["rate"],
        "achieved": round(achieved(best), 4),
        "n_requests": best["n_requests"],
        "ttft_margin_s": round(target_s - best["ttft_p95"], 4),
        "tpot_margin_s": round(tpot_bound_s - best["tpot_p95"], 4),
        "next_fail": nxt["rate"] if nxt else None,
        "broke_on": what_broke(nxt, target_s, tpot_bound_s) if nxt else None,
        "lowest_fail": lowest["rate"] if lowest else None,
        "monotone": lowest is None or lowest["rate"] > best["rate"],
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


def _provenance(meta: dict) -> dict:
    """What the run file recorded about where and how it ran (absent keys stay absent)."""
    out = {}
    for key in PROVENANCE:
        if key not in meta:
            continue
        value = meta[key]
        if key == "gpus":
            value = [_command(g) for g in value] if isinstance(value, list) else None
        elif key == "seed":
            value = value if isinstance(value, int) else None
        else:
            value = _command(value)
        if value is not None:
            out[key] = value
    return out


def summarize_run(path: Path, targets_s: list[float]) -> dict:
    doc = json.loads(path.read_text())
    meta = doc["metadata"]
    head = {
        "run_id": _checked(meta["run_id"], IDENT, "run id"),
        "unit": _checked(meta["unit"], IDENT, "unit"),
    }
    if "quality" in doc:  # protocol §9: answer quality, no load steps
        q = doc["quality"]
        rows = [
            {
                "id": _checked(a["id"], IDENT, "question id"),
                "em": a["em"],
                "f1": round(a["f1"], 4),
                "truncated": bool(a["truncated"]),
            }
            for a in q.get("answers", [])
        ]
        return {
            **head,
            "profile": "quality",
            "mode": "quality",
            "server_flags": _command(meta["server_flags"]),
            "git_commit": _checked(meta.get("git_commit"), COMMIT, "git commit"),
            **_provenance(meta),
            "quality": {k: q.get(k) for k in ("n", "exact_match", "f1", "truncated")},
            "questions": rows,  # scores per question, for paired comparisons (no answer text)
        }
    steps = [s["summary"] for s in doc["steps"]]
    run = {
        **head,
        "profile": _checked(meta["profile"], IDENT, "profile"),
        "server_flags": _command(meta["server_flags"]),
        "cold_start_s": meta.get("cold_start_s"),
        "git_commit": _checked(meta.get("git_commit"), COMMIT, "git commit"),
        **_provenance(meta),
        "kv_cache_tokens": kv_cache_tokens(path.parent / "logs" / f"{meta['run_id']}.server.log"),
        "steps": [compact_step(s) for s in steps],
    }
    if meta.get("mode") == "repeat":
        run["mode"] = "repeat"
        run["repeat_pass"] = {
            f"{t:g}": {
                rate_key(r): [meets(s, t) for s in steps if s["rate"] == r] for r in meta["rates"]
            }
            for t in targets_s
        }
    else:
        run["capacity"] = {f"{t:g}": capacity(steps, t) for t in targets_s}
    return run


def summarize_dir(results_dir: Path, targets_s: list[float]) -> dict:
    runs, sources = [], []
    for path in sorted(results_dir.glob("*.json")):
        sources.append(_source(path))
        if path.name.startswith(("plan-", "campaign-clock")) or path.name.endswith(".failed.json"):
            continue
        runs.append(summarize_run(path, targets_s))
    logs = results_dir / "logs"
    if logs.is_dir():
        sources += [_source(p) for p in sorted(logs.glob("*.server.log"))]
    plans = {
        p.stem: _clean(json.loads(p.read_text()), p.name)
        for p in sorted(results_dir.glob("plan-*.json"))
    }
    return {
        "targets_s": targets_s,
        "tpot_bound_s": TPOT_BOUND_S,
        "runs": runs,
        "plans": plans,
        "sources": sources,  # the raw files this summary was built from
    }


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
                low = "" if c.get("monotone", True) else f", {c['lowest_fail']:g} failed"
                cells.append(f"{c['capacity']:g} (sent {c['achieved']:.3g}{broke}{low})")
        kv = f"{run['kv_cache_tokens']:,}" if run["kv_cache_tokens"] else "-"
        rows.append(f"| {run['unit']} | " + " | ".join(cells) + f" | {kv} |")
    return "\n".join(rows)
