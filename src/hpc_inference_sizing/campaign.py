"""Campaign runner: every (unit, profile) run of a matrix, one after the other, resumable.

A matrix (TOML, see `campaign/`) lists the serving units (the server command flags of each
split), the profiles and the sweep settings. For each run the runner:

1. skips it if its result file already exists (resume after an interruption);
2. starts the server for that unit and waits for `/health`, timing the cold start;
3. runs the sweep and writes `<results_dir>/<run_id>.json`;
4. stops the server, runs the optional sync command (results leave the machine at once);
5. stops the campaign when the time or spending limit is reached.

A unit whose server does not start is recorded in `<run_id>.failed.json`: that is a result
("not supported on this hardware"), not something to debug on paid time.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass(frozen=True)
class Matrix:
    name: str
    results_dir: Path
    server_url: str
    served_model: str
    server_command: str  # empty: the server is already running (e.g. a laptop dry run)
    units: dict[str, str]  # unit -> server flags
    profiles: list[str]
    targets_s: list[float]
    step_duration_s: float = 180.0
    min_requests: int = 20
    start_rate: float = 0.25
    max_rate: float = 64.0
    bisect_steps: int = 2
    tpot_bound_ms: float = 1000 / 15
    health_timeout_s: float = 1800.0
    sync_command: str = ""
    model: str = ""

    def run_ids(self) -> list[str]:
        return [f"{unit}-{profile}" for unit in self.units for profile in self.profiles]

    def split(self, run_id: str) -> tuple[str, str]:
        unit, _, profile = run_id.rpartition("-")
        if unit not in self.units or profile not in self.profiles:
            raise ValueError(f"unknown run {run_id!r}; runs: {', '.join(self.run_ids())}")
        return unit, profile

    def command_for(self, unit: str) -> list[str]:
        if not self.server_command:
            return []
        return shlex.split(self.server_command.format(model=self.model, flags=self.units[unit]))


def load_matrix(path: Path) -> Matrix:
    raw = tomllib.loads(path.read_text())
    server = raw.pop("server", {})
    units = {name: spec.get("flags", "") for name, spec in raw.pop("units").items()}
    results_dir = Path(raw.pop("results_dir"))
    if not results_dir.is_absolute():
        results_dir = path.parent.parent / results_dir  # relative to the repository root
    return Matrix(
        results_dir=results_dir,
        server_command=server.get("command", ""),
        server_url=server.get("url", "http://127.0.0.1:8000"),
        served_model=server.get("served_model", raw.get("model", "")),
        health_timeout_s=server.get("health_timeout_s", 1800.0),
        units=units,
        **raw,
    )


@dataclass
class Status:
    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)


def status(matrix: Matrix) -> Status:
    st = Status()
    for run_id in matrix.run_ids():
        if (matrix.results_dir / f"{run_id}.json").exists():
            st.done.append(run_id)
        elif (matrix.results_dir / f"{run_id}.failed.json").exists():
            st.failed.append(run_id)
        else:
            st.pending.append(run_id)
    return st


def wait_healthy(url: str, timeout_s: float, proc: subprocess.Popen | None) -> float:
    """Seconds until `GET url/health` answers 200; raises if the server dies or times out."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before being ready")
        try:
            if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError(f"server not ready after {timeout_s:.0f} s")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@dataclass
class Limits:
    """Stop rules of protocol §10: wall-clock and spending caps for the whole campaign."""

    max_hours: float | None = None
    budget_usd: float | None = None
    price_per_hour_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def spent_usd(self) -> float:
        return (time.monotonic() - self.started_at) / 3600 * self.price_per_hour_usd

    def reason_to_stop(self) -> str | None:
        hours = (time.monotonic() - self.started_at) / 3600
        if self.max_hours is not None and hours >= self.max_hours:
            return f"time limit reached ({hours:.2f} h >= {self.max_hours} h)"
        if self.budget_usd is not None and self.spent_usd() >= self.budget_usd:
            return f"budget reached ({self.spent_usd():.2f} >= {self.budget_usd} USD)"
        return None


def run_one(matrix: Matrix, run_id: str, meta: dict[str, str], log: Callable[[str], None]) -> bool:
    """One run; True if a result was written, False if the unit failed to start."""
    from hpc_inference_sizing.bench.data import MistralTokenizer, load_piaf
    from hpc_inference_sizing.bench.runner import Bench, metadata, write_run
    from hpc_inference_sizing.bench.workload import PROFILES, Workload

    unit, profile = matrix.split(run_id)
    matrix.results_dir.mkdir(parents=True, exist_ok=True)
    command = matrix.command_for(unit)
    proc = None
    log_path = matrix.results_dir / "logs" / f"{run_id}.server.log"
    try:
        if command:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log(f"[{run_id}] starting: {shlex.join(command)}")
            proc = subprocess.Popen(command, stdout=log_path.open("w"), stderr=subprocess.STDOUT)
        try:
            cold_start_s = wait_healthy(matrix.server_url, matrix.health_timeout_s, proc)
        except RuntimeError as err:
            failure = {
                "run_id": run_id,
                "unit": unit,
                "flags": matrix.units[unit],
                "error": str(err),
                "server_log": str(log_path) if command else None,
            }
            (matrix.results_dir / f"{run_id}.failed.json").write_text(json.dumps(failure, indent=1))
            log(f"[{run_id}] FAILED to start: {err}")
            return False
        log(f"[{run_id}] server ready after {cold_start_s:.0f} s")

        workload = Workload(load_piaf(), MistralTokenizer(), PROFILES[profile], seed=0)
        bench = Bench(
            matrix.server_url,
            matrix.served_model,
            workload,
            matrix.step_duration_s,
            min_requests=matrix.min_requests,
        )
        result = asyncio.run(
            bench.sweep(
                matrix.targets_s,
                start_rate=matrix.start_rate,
                max_rate=matrix.max_rate,
                bisect_steps=matrix.bisect_steps,
                tpot_bound_s=matrix.tpot_bound_ms / 1000,
            )
        )
        run_meta = {
            "campaign": matrix.name,
            "run_id": run_id,
            "unit": unit,
            "server_flags": matrix.units[unit],
            "server_command": shlex.join(command) if command else "external",
            "cold_start_s": round(cold_start_s, 1),
            **meta,
        }
        write_run(matrix.results_dir / f"{run_id}.json", metadata(bench, run_meta), bench, result)
        caps = ", ".join(f"{t:g} s: {c}" for t, c in result.capacity.items())
        log(f"[{run_id}] done; capacity (req/s) {caps}")
        return True
    finally:
        stop_server(proc)
        if matrix.sync_command:
            subprocess.run(
                shlex.split(matrix.sync_command.format(results_dir=matrix.results_dir)),
                check=False,
            )


def run_campaign(
    matrix: Matrix,
    limits: Limits,
    meta: dict[str, str],
    only: list[str] | None = None,
    retry_failed: bool = False,
    log: Callable[[str], None] = print,
    runner: Callable[[Matrix, str, dict[str, str], Callable[[str], None]], bool] = run_one,
) -> Status:
    """Run the pending runs (or `only` those), honouring the limits; returns the final status."""
    st = status(matrix)
    todo = st.pending + (st.failed if retry_failed else [])
    if only:
        for run_id in only:
            matrix.split(run_id)  # validates the name
        todo = [r for r in only if r in todo]
    for run_id in todo:
        if reason := limits.reason_to_stop():
            log(f"stopping before {run_id}: {reason}")
            break
        if retry_failed:
            (matrix.results_dir / f"{run_id}.failed.json").unlink(missing_ok=True)
        runner(matrix, run_id, meta, log)
        if limits.price_per_hour_usd:
            log(f"spent so far: {limits.spent_usd():.2f} USD")
    return status(matrix)
