"""Follow a campaign running on the rented node, from the laptop (protocol §10).

Every `every_s` seconds `follow`:

1. reads the instance from the Vast.ai API (gone: stop);
2. copies the node's results folder into the local one (results leave the machine);
3. reports each run that finished since the last look, with its capacities;
4. when the campaign process has ended (or no run is pending) and the results are copied,
   destroys the instance, so a finished campaign does not keep billing until the deadline;
5. raises an alert when the campaign never starts (`start_timeout_s` after the follow began)
   or stalls (its log unchanged for `stall_timeout_s` while it runs), so someone can look; if
   nothing changes for `alert_grace_s` more, destroys the instance, so a node that does
   nothing is not left billing until the guard's deadline (`safety_destroy=False`: alert only).

Each notable event is also appended to `events_path` as one JSON line, for notifications.
The guard (`vast guard`) still runs beside it: it owns the time and budget caps.
"""

from __future__ import annotations

import io
import json
import re
import shlex
import subprocess
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from hpc_inference_sizing.campaign import Matrix, status
from hpc_inference_sizing.vast import VastClient, VastError, _listed, destroy_with_retries

# The only files taken from the node: run files, failure records, plans, the campaign clock and
# server logs, with plain names. Anything else in the archive is skipped, so the third-party
# host cannot plant or overwrite other files in the results folder.
ALLOWED_MEMBER = re.compile(
    r"(\./)?([A-Za-z0-9_+-]+(\.failed)?\.json|plan-[a-z]+\.json|campaign-clock\.json"
    r"|logs/[A-Za-z0-9_+-]+\.server\.log)"
)

# The brackets keep pgrep from matching the shell that runs it (its command line holds the
# pattern, not a match of it). With SLURM, a queued or running array task also counts, so the
# few seconds between two tasks do not look like the end of the campaign.
CAMPAIGN_PATTERN = "[h]pc-sizing campaign run"
RUNNING_CHECK = (
    f"pgrep -f {shlex.quote(CAMPAIGN_PATTERN)} >/dev/null || "
    '{ command -v squeue >/dev/null && [ -n "$(squeue -h 2>/dev/null)" ]; }'
)


class Remote(Protocol):
    def pull(self, local_dir: Path) -> None: ...
    def campaign_running(self) -> bool: ...
    def last_log_line(self) -> str: ...
    def log_age_s(self) -> float | None: ...


@dataclass
class SshRemote:
    """The node, reached with the owner's ssh key (no password, no agent forwarding)."""

    host: str
    port: int
    remote_dir: str  # results folder, relative to root's home on the node
    log_file: str = "campaign.log"
    timeout_s: float = 120

    def _ssh(self, command: str, **kw) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "ssh",
                "-p",
                str(self.port),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "StrictHostKeyChecking=accept-new",
                f"root@{self.host}",
                command,
            ],
            timeout=self.timeout_s,
            **kw,
        )

    def pull(self, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        remote = shlex.quote(self.remote_dir)
        out = self._ssh(f"test -d {remote} && tar czf - -C {remote} .", capture_output=True)
        if out.returncode != 0:
            raise OSError(f"copy from the node failed: {out.stderr.decode()[-200:].strip()}")
        # The host is a third party: only allow-listed regular files are extracted, and the
        # "data" filter still refuses absolute paths, ".." and special files.
        try:
            with tarfile.open(fileobj=io.BytesIO(out.stdout), mode="r:gz") as archive:
                wanted = [
                    m
                    for m in archive.getmembers()
                    if m.isfile() and ALLOWED_MEMBER.fullmatch(m.name)
                ]
                archive.extractall(local_dir, members=wanted, filter="data")
        except (tarfile.TarError, OSError) as err:
            raise OSError(f"unpacking the results failed: {err}") from err

    def campaign_running(self) -> bool:
        out = self._ssh(f"{RUNNING_CHECK} && echo running || echo stopped", capture_output=True)
        answer = out.stdout.decode().strip()
        if out.returncode != 0 or answer not in ("running", "stopped"):
            raise OSError(f"cannot tell whether the campaign runs: {out.stderr.decode()[-200:]}")
        return answer == "running"

    def last_log_line(self) -> str:
        out = self._ssh(f"tail -n 1 {shlex.quote(self.log_file)}", capture_output=True)
        return plain(out.stdout.decode(errors="replace"))

    def log_age_s(self) -> float | None:
        """Seconds since the campaign log last changed on the node; None without a log (e.g. a
        SLURM array writes its own output files), in which case no stall is judged."""
        script = (
            "import os, sys, time; p = sys.argv[1]; "
            "print(time.time() - os.path.getmtime(p) if os.path.exists(p) else 'none')"
        )
        out = self._ssh(
            f"python3 -c {shlex.quote(script)} {shlex.quote(self.log_file)}", capture_output=True
        )
        if out.returncode != 0:
            raise OSError(f"cannot read the campaign log age: {out.stderr.decode()[-200:]}")
        answer = out.stdout.decode().strip()
        return None if answer == "none" else float(answer)


@dataclass
class FollowResult:
    reason: str
    destroyed: bool
    status: dict[str, list[str]] = field(default_factory=dict)


def run_files(results_dir: Path) -> tuple[list[str], list[str]]:
    """(finished, failed) run ids found in the results folder, every phase included."""
    if not results_dir.is_dir():
        return [], []
    failed = sorted(
        plain(p.name.removesuffix(".failed.json")) for p in results_dir.glob("*.failed.json")
    )
    done = sorted(
        plain(p.stem)
        for p in results_dir.glob("*.json")
        if not p.name.endswith(".failed.json")
        and not p.name.startswith(("plan-", "campaign-clock"))
    )
    return done, failed


def plain(text: str, limit: int = 160) -> str:
    """Text from the node, made safe to log: printable characters only, one line, short."""
    return "".join(c if c.isprintable() else " " for c in text).strip()[-limit:]


def capacities(results_dir: Path, run_id: str) -> str | None:
    """'1 s: unreachable, 3 s: 0.5' from a run file, numbers only; None if not readable yet."""
    try:
        doc = json.loads((results_dir / f"{run_id}.json").read_text())
        if "sweep" not in doc:  # a repeat run: fixed rates, no capacity search
            return f"{len(doc['steps']):d} repeat steps"
        caps = doc["sweep"]["capacity_rps"]
        parts = []
        for target, cap in caps.items():
            value = "unreachable" if cap is None else f"{float(cap):g} req/s"
            parts.append(f"{float(target):g} s: {value}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return ", ".join(parts)


def follow(
    client: VastClient,
    instance_id: int,
    matrix: Matrix,
    remote_for: Callable[[dict], Remote],
    events_path: Path | None = None,
    every_s: float = 300,
    destroy_when_done: bool = True,
    start_timeout_s: float = 1800,
    stall_timeout_s: float = 2700,
    alert_grace_s: float = 1800,
    safety_destroy: bool = True,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> FollowResult:
    """Follow until the campaign is finished and copied (then destroy) or the instance is gone.

    `stall_timeout_s` must exceed the longest silent stretch of a healthy run: a server start
    (up to the matrix's health timeout, 40 min) logs nothing until the server is ready.
    """
    started_at = clock()
    alerted: set[str] = set()  # "never_started", "stalled": alerted, not yet resolved
    seen: set[str] = set()
    seen_running = False
    stopped_looks = 0  # the end is declared on two looks in a row, not on one
    failures = 0

    def event(kind: str, message: str) -> None:
        message = plain(message, 300)  # parts come from the node (names, ssh errors): plain text
        log(f"follow: {message}")
        if events_path is not None:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a") as f:
                stamp = datetime.now(UTC).isoformat(timespec="seconds")
                f.write(json.dumps({"at": stamp, "kind": kind, "message": message}) + "\n")

    def as_dict(st) -> dict[str, list[str]]:
        return {"done": st.done, "failed": st.failed, "pending": st.pending}

    def end(reason: str, st) -> FollowResult:
        if not destroy_when_done:
            return FollowResult(reason, False, as_dict(st))
        destroyed = destroy_with_retries(client, instance_id, log=log, sleep=sleep)
        if destroyed:
            event("destroyed", f"instance {instance_id} destroyed ({reason})")
        else:
            event("destroy_failed", f"DESTROY FAILED for {instance_id}: by hand now")
        return FollowResult(reason, destroyed, as_dict(st))

    seen.update(sum(run_files(matrix.results_dir), []))  # copied before this follow started
    event("start", f"following instance {instance_id}, {len(seen)} runs already here")
    while True:
        try:
            info = client.show(instance_id)
        except VastError as err:
            log(f"follow: cannot read the instance ({err}); retrying")
            info = None
        if info is not None and not info and not _listed(client, instance_id):
            event("gone", f"instance {instance_id} no longer exists")
            return FollowResult("instance gone", False, as_dict(status(matrix)))
        if info:
            try:
                remote = remote_for(info)
                running = remote.campaign_running()
                remote.pull(matrix.results_dir)  # after the check: a run that just ended is in
                last = remote.last_log_line()
                age = remote.log_age_s() if running else None
                failures = 0
            except Exception as err:  # noqa: BLE001 - detached: a surprise must not end it
                failures += 1
                log(f"follow: node unreachable ({err}); attempt {failures}")
                if failures == 3:
                    event("unreachable", f"node unreachable 3 times in a row: {err}")
            else:
                seen_running = seen_running or running
                st = status(matrix)
                done, failed = run_files(matrix.results_dir)
                for run_id in done + failed:
                    if run_id in seen:
                        continue
                    if run_id in done:
                        caps = capacities(matrix.results_dir, run_id)
                        if caps is None:
                            continue  # copied mid-write: reported on a later look
                        event("run_done", f"{run_id} done ({caps})")
                    else:
                        event("run_failed", f"{run_id} failed (see {run_id}.failed.json)")
                    seen.add(run_id)
                log(
                    f"follow: {len(done) + len(failed)} runs copied ({len(failed)} failed; base "
                    f"{len(st.done) + len(st.failed)}/{len(matrix.run_ids())}), "
                    f"campaign {'running' if running else 'stopped'}"
                    f"; node: {last[-160:]}"
                )
                stopped_looks = 0 if running else stopped_looks + 1
                waited = clock() - started_at
                never = not seen_running and bool(st.pending) and waited >= start_timeout_s
                stalled = running and age is not None and age >= stall_timeout_s
                for kind, now, minutes, deadline in (
                    ("never_started", never, waited / 60, start_timeout_s),
                    ("stalled", stalled, (age or 0) / 60, stall_timeout_s),
                ):
                    if now and kind not in alerted:
                        alerted.add(kind)
                        what = "not started" if kind == "never_started" else "log unchanged"
                        action = (
                            f"destroy in {alert_grace_s / 60:.0f} min if nothing changes"
                            if safety_destroy and destroy_when_done
                            else "alert only, the node is kept"
                        )
                        event(kind, f"ALERT campaign {what} for {minutes:.0f} min; {action}")
                    elif not now and kind in alerted:
                        alerted.discard(kind)
                        state = "moving again" if running else "no longer running"
                        event(
                            "recovered", f"{kind.replace('_', ' ')} alert cleared: campaign {state}"
                        )
                    elif (
                        now
                        and safety_destroy
                        and destroy_when_done  # keep following when nothing may be destroyed
                        and (
                            (kind == "never_started" and waited >= deadline + alert_grace_s)
                            or (kind == "stalled" and (age or 0) >= deadline + alert_grace_s)
                        )
                    ):
                        return end(f"campaign {kind.replace('_', ' ')}", st)
                if stopped_looks >= 2 and (seen_running or not st.pending):
                    if st.pending:
                        event("stopped", f"campaign stopped with {len(st.pending)} runs pending")
                    else:
                        event(
                            "finished",
                            f"campaign finished: {len(st.done)} done, "
                            f"{len(st.failed)} failed, results copied",
                        )
                    return end("campaign ended", st)
        sleep(every_s)


def remote_for_instance(remote_dir: str) -> Callable[[dict], Remote]:
    def make(info: dict) -> Remote:
        host, port = info.get("ssh_host"), info.get("ssh_port")
        if not host or not port:
            raise ValueError("no ssh address yet (instance loading)")
        return SshRemote(str(host), int(port), remote_dir)

    return make
