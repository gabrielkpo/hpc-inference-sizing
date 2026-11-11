"""Ties the bench together: one step (load + /metrics scraping), a full sweep, and the run file.

A run file is one JSON document: metadata (git commit, versions, host, server, options), the
step summaries, the per-request records and the /metrics samples. It is written once the run
ends; the campaign (M3) saves it off the machine and resumes from the runs already present.
"""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from hpc_inference_sizing import __version__
from hpc_inference_sizing.bench.client import run_step
from hpc_inference_sizing.bench.scrape import scrape_until
from hpc_inference_sizing.bench.stats import StepSummary, summarize
from hpc_inference_sizing.bench.sweep import SweepResult, sweep
from hpc_inference_sizing.bench.workload import Workload, poisson_arrivals


@dataclass
class Bench:
    base_url: str
    model: str
    workload: Workload
    step_duration_s: float
    seed: int = 0
    min_requests: int = 20  # a step lasts long enough to expect at least this many requests
    scrape_interval_s: float = 5.0
    transport: httpx.AsyncBaseTransport | None = None
    steps: list[dict] = field(default_factory=list)  # detailed record of every step
    _next_index: int = 0

    async def step(self, rate: float) -> StepSummary:
        """One load step at `rate`, with /metrics scraped in parallel.

        The step lasts `step_duration_s`, longer at low rates so that about `min_requests`
        requests arrive: a step with a handful of requests measures nothing.
        """
        duration = max(self.step_duration_s, self.min_requests / rate)
        seed = self.seed + len(self.steps)
        # Build every prompt first, so that request and /metrics times share one origin (t0).
        first = self._next_index
        n = len(poisson_arrivals(rate, duration, seed))
        prompts = [self.workload.prompt(first + i)[0] for i in range(n)]
        stop = asyncio.Event()
        t0 = time.perf_counter()
        scraper = asyncio.create_task(
            scrape_until(self.base_url, stop, t0, self.scrape_interval_s, transport=self.transport)
        )
        records = await run_step(
            self.base_url,
            self.model,
            lambda i: prompts[i - first],
            self.workload.profile.output_tokens,
            rate,
            duration,
            seed=seed,
            first_index=first,
            transport=self.transport,
            t0=t0,
        )
        stop.set()
        samples = await scraper
        self._next_index += len(records)
        summary = summarize(records, rate, duration)
        self.steps.append(
            {
                "summary": summary.to_dict(),
                "requests": [r.to_dict() for r in records],
                "metrics": samples,
            }
        )
        return summary

    async def sweep(self, targets_s: list[float], **kw) -> SweepResult:
        return await sweep(self.step, targets_s, **kw)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def metadata(bench: Bench, extra: dict[str, str]) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "server_url": bench.base_url,
        "model": bench.model,
        "profile": bench.workload.profile.name,
        "input_tokens": bench.workload.profile.input_tokens,
        "output_tokens": bench.workload.profile.output_tokens,
        "seed": bench.seed,
        "step_duration_s": bench.step_duration_s,
        "min_requests_per_step": bench.min_requests,
        **extra,  # e.g. gpu, split, vllm flags, image digest: given on the command line
    }


def write_run(path: Path, meta: dict, bench: Bench, sweep_result: SweepResult | None) -> None:
    doc = {"metadata": meta, "steps": bench.steps}
    if sweep_result is not None:
        doc["sweep"] = sweep_result.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.rename(path)  # a run file is either complete or absent
