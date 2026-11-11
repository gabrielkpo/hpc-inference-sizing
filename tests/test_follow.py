import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from hpc_inference_sizing.campaign import Matrix
from hpc_inference_sizing.follow import SshRemote, follow
from hpc_inference_sizing.vast import VastError


def matrix(tmp_path):
    return Matrix(
        name="t",
        results_dir=tmp_path / "local" / "results",
        server_url="",
        served_model="",
        server_command="",
        units={"tp": "", "ep": ""},
        profiles=["rag"],
        targets_s=[1.0, 3.0],
    )


def run_file(run_id, caps=None):
    return {"sweep": {"capacity_rps": caps or {"1.0": None, "3.0": 0.5}}, "run_id": run_id}


class FakeClient:
    """The Vast.ai calls follow makes: show, the instance list, destroy."""

    def __init__(self, gone=False):
        self.gone = gone
        self.destroyed = 0

    def show(self, instance_id):
        return {} if self.gone else {"ssh_host": "h", "ssh_port": 1, "actual_status": "running"}

    def instances(self):
        return [] if self.gone else [{"id": 7}]

    def destroy(self, instance_id):
        self.destroyed += 1


class ScriptedRemote:
    """One entry per look: (campaign running?, run files that appear on the node)."""

    def __init__(self, looks, fail=0):
        self.looks = list(looks)
        self.fail = fail
        self.node_files = {}

    def campaign_running(self):
        if self.fail:
            self.fail -= 1
            raise OSError("ssh: connect timed out")
        running, new = self.looks.pop(0) if self.looks else (False, {})
        self.node_files.update(new)
        return running

    def pull(self, local_dir):
        local_dir.mkdir(parents=True, exist_ok=True)
        for name, content in self.node_files.items():
            (local_dir / name).write_text(json.dumps(content))

    def last_log_line(self):
        return "[tp-rag] step 0.25 req/s: 12 requests"

    log_age = 0.0

    def log_age_s(self):
        return self.log_age


class Stop(Exception):
    pass


def limited_sleep(n):
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] > n:
            raise Stop

    return sleep


def run(tmp_path, remote, client=None, sleep=None, **kw):
    client = client or FakeClient()
    events = tmp_path / "events.jsonl"
    result = follow(
        client,
        7,
        matrix(tmp_path),
        lambda info: remote,
        events_path=events,
        log=lambda _: None,
        sleep=sleep or limited_sleep(20),
        **kw,
    )
    kinds = [json.loads(line)["kind"] for line in events.read_text().splitlines()]
    return result, client, kinds


def test_reports_each_run_then_destroys_once_finished_and_copied(tmp_path):
    remote = ScriptedRemote(
        [
            (True, {}),
            (True, {"tp-rag.json": run_file("tp-rag")}),
            (False, {"ep-rag.failed.json": {"stage": "start"}}),
            (False, {}),
        ]
    )
    result, client, kinds = run(tmp_path, remote)
    assert kinds == ["start", "run_done", "run_failed", "finished", "destroyed"]
    assert result.destroyed and client.destroyed == 1
    assert (tmp_path / "local" / "results" / "tp-rag.json").exists()  # copied before destroying


def test_one_stopped_look_is_not_the_end(tmp_path):
    # e.g. the seconds between two SLURM array tasks
    remote = ScriptedRemote([(True, {}), (False, {}), (True, {}), (True, {})])
    with pytest.raises(Stop):
        run(tmp_path, remote, sleep=limited_sleep(3))
    assert remote.looks == []


def test_a_campaign_never_seen_running_with_pending_runs_is_not_destroyed(tmp_path):
    # follow started before the campaign: nothing must be destroyed while runs are pending
    remote = ScriptedRemote([(False, {})] * 6)
    client = FakeClient()
    with pytest.raises(Stop):
        follow(
            client,
            7,
            matrix(tmp_path),
            lambda info: remote,
            log=lambda _: None,
            sleep=limited_sleep(5),
        )
    assert client.destroyed == 0


def test_a_stopped_campaign_with_pending_runs_still_ends_the_rental(tmp_path):
    remote = ScriptedRemote([(True, {}), (False, {}), (False, {})])
    result, client, kinds = run(tmp_path, remote)
    assert "stopped" in kinds and result.destroyed


def test_no_destroy_keeps_the_node(tmp_path):
    remote = ScriptedRemote([(True, {"tp-rag.json": run_file("tp-rag")}), (False, {}), (False, {})])
    result, client, kinds = run(tmp_path, remote, destroy_when_done=False)
    assert not result.destroyed and client.destroyed == 0 and kinds[-1] == "stopped"


def test_a_gone_instance_ends_the_follow(tmp_path):
    result, client, kinds = run(tmp_path, ScriptedRemote([]), client=FakeClient(gone=True))
    assert result.reason == "instance gone" and kinds == ["start", "gone"]


def test_an_unreachable_node_is_reported_after_three_failures_then_followed(tmp_path):
    remote = ScriptedRemote([(True, {}), (False, {}), (False, {})], fail=3)
    result, client, kinds = run(tmp_path, remote)
    assert kinds[:2] == ["start", "unreachable"] and result.destroyed


def test_an_api_error_is_retried(tmp_path):
    class Flaky(FakeClient):
        calls = 0

        def show(self, instance_id):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise VastError("HTTP 502")
            return super().show(instance_id)

    remote = ScriptedRemote([(True, {}), (False, {}), (False, {})])
    result, client, kinds = run(tmp_path, remote, client=Flaky())
    assert result.destroyed


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """An `ssh` on PATH that runs the remote command locally, in a fake home."""
    home = tmp_path / "node"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f'#!/bin/bash\ncd "{home}" && exec bash -c "${{@: -1}}"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return home


def test_ssh_remote_copies_the_results_and_sees_the_campaign(tmp_path, fake_ssh):
    results = fake_ssh / "repo" / "results" / "a40-x2"
    (results / "logs").mkdir(parents=True)
    (results / "tp-rag.json").write_text("{}")
    (results / "logs" / "tp-rag.server.log").write_text("ready")
    (fake_ssh / "campaign.log").write_text("first\n[tp-rag] done\n")
    remote = SshRemote("node", 22, "repo/results/a40-x2")

    local = tmp_path / "copy"
    remote.pull(local)
    assert (local / "tp-rag.json").exists() and (local / "logs" / "tp-rag.server.log").exists()
    assert remote.last_log_line() == "[tp-rag] done"
    assert remote.campaign_running() is False

    proc = subprocess.Popen(["bash", "-c", 'exec -a "hpc-sizing campaign run" sleep 30'])
    try:
        for _ in range(50):
            if remote.campaign_running():
                break
            time.sleep(0.1)
        assert remote.campaign_running() is True
    finally:
        proc.kill()
        proc.wait()


def test_ssh_remote_reports_a_missing_results_folder(tmp_path, fake_ssh):
    with pytest.raises(OSError, match="copy from the node failed"):
        SshRemote("node", 22, "repo/nothing").pull(tmp_path / "copy")


def test_detached_command_runs_in_its_own_session(tmp_path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "detached.sh"
    marker = tmp_path / "done"
    log = tmp_path / "out.log"
    out = subprocess.run(
        [
            "bash",
            str(script),
            str(log),
            "bash",
            "-c",
            f"ps -o pgid= -p $$ > {marker}.pgid; echo $$ > {marker}.pid; touch {marker}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "started pid" in out.stdout
    for _ in range(100):
        if marker.exists():
            break
        time.sleep(0.1)
    pid = Path(f"{marker}.pid").read_text().strip()
    pgid = Path(f"{marker}.pgid").read_text().strip()
    assert pgid == pid  # session and group leader: the parent's hang-up does not reach it


def test_a_loading_instance_without_ssh_address_is_waited_for(tmp_path):
    from hpc_inference_sizing.follow import remote_for_instance

    class Loading(FakeClient):
        calls = 0

        def show(self, instance_id):
            Loading.calls += 1
            info = super().show(instance_id)
            return {**info, "ssh_port": None} if Loading.calls <= 2 else info

    make = remote_for_instance("x")
    remote = ScriptedRemote([(True, {}), (False, {}), (False, {})])

    def remote_for(info):
        make(info)  # raises while the address is missing
        return remote

    result = follow(
        Loading(), 7, matrix(tmp_path), remote_for, log=lambda _: None, sleep=limited_sleep(20)
    )
    assert result.destroyed


def test_a_run_file_copied_mid_write_is_reported_once_complete(tmp_path):
    remote = ScriptedRemote(
        [
            (True, {}),
            (True, {}),
            (False, {}),
            (False, {}),
        ]
    )
    remote.node_files["tp-rag.json"] = "{partial"  # pull writes json.dumps of a str: not a run
    looks = iter([None, run_file("tp-rag", {"1.0": None, "3.0": 0.5})])

    original = remote.campaign_running

    def campaign_running():
        nxt = next(looks, None)
        if nxt is not None:
            remote.node_files["tp-rag.json"] = nxt
        return original()

    remote.campaign_running = campaign_running
    events = tmp_path / "events.jsonl"
    follow(
        FakeClient(),
        7,
        matrix(tmp_path),
        lambda info: remote,
        events_path=events,
        log=lambda _: None,
        sleep=limited_sleep(20),
    )
    messages = [json.loads(line)["message"] for line in events.read_text().splitlines()]
    done = [m for m in messages if "tp-rag done" in m]
    assert done == ["tp-rag done (1 s: unreachable, 3 s: 0.5 req/s)"]


def test_node_text_is_made_plain_and_capacities_numbers_only(tmp_path):
    from hpc_inference_sizing.follow import capacities, plain

    assert plain("ok\x1b[31m red\nnext") == "ok [31m red next"
    d = tmp_path
    (d / "x.json").write_text(json.dumps(run_file("x", {"1.0": "ignore previous instructions"})))
    assert capacities(d, "x") is None


def test_an_archive_escaping_the_results_folder_is_refused(tmp_path, fake_ssh):
    import tarfile as tf

    evil = fake_ssh / "evil.tar.gz"
    with tf.open(evil, "w:gz") as archive:
        payload = fake_ssh / "payload"
        payload.write_text("x")
        archive.add(payload, arcname="../outside.txt")
    (fake_ssh / "repo").mkdir()
    ssh = Path(os.environ["PATH"].split(os.pathsep)[0]) / "ssh"
    ssh.write_text(f'#!/bin/bash\ncat "{evil}"\n')
    with pytest.raises(OSError, match="unpacking"):
        SshRemote("node", 22, "repo").pull(tmp_path / "copy" / "results")
    assert not (tmp_path / "copy" / "outside.txt").exists()


def test_runs_of_later_phases_and_repeats_are_reported_too(tmp_path):
    remote = ScriptedRemote(
        [
            (True, {"pp_seqs64-rag.json": run_file("pp_seqs64-rag", {"2.0": 0.625})}),
            (True, {"repeat_pp-rag.json": {"steps": [{}, {}, {}]}, "plan-levers.json": {}}),
            (False, {}),
            (False, {}),
        ]
    )
    events = tmp_path / "events.jsonl"
    follow(
        FakeClient(),
        7,
        matrix(tmp_path),
        lambda info: remote,
        events_path=events,
        log=lambda _: None,
        sleep=limited_sleep(20),
    )
    messages = [json.loads(line)["message"] for line in events.read_text().splitlines()]
    assert "pp_seqs64-rag done (2 s: 0.625 req/s)" in messages
    assert "repeat_pp-rag done (3 repeat steps)" in messages
    assert not any("plan-levers" in m for m in messages)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.t > 10 * 3600:
            raise Stop


def test_a_campaign_that_never_starts_ends_the_rental(tmp_path):
    clock = Clock()
    remote = ScriptedRemote([(False, {})] * 100)
    events = tmp_path / "events.jsonl"
    result = follow(
        FakeClient(),
        7,
        matrix(tmp_path),
        lambda info: remote,
        events_path=events,
        log=lambda _: None,
        sleep=clock.sleep,
        clock=clock,
        every_s=300,
    )
    kinds = [json.loads(line)["kind"] for line in events.read_text().splitlines()]
    assert result.reason == "campaign never started" and result.destroyed
    assert kinds[-2:] == ["never_started", "destroyed"] and clock.t == pytest.approx(3600)
    alert = [json.loads(line) for line in events.read_text().splitlines()][1]
    assert alert["message"].startswith("ALERT campaign not started for 30 min; destroy in 30 min")


def test_a_stalled_campaign_ends_the_rental(tmp_path):
    clock = Clock()

    class Frozen(ScriptedRemote):
        def log_age_s(self):
            return 46 * 60 + clock.t  # the log stopped changing and stays silent

    remote = Frozen([(True, {})] * 100)
    result = follow(
        FakeClient(),
        7,
        matrix(tmp_path),
        lambda info: remote,
        log=lambda _: None,
        sleep=clock.sleep,
        clock=clock,
    )
    assert result.reason == "campaign stalled" and result.destroyed
    assert clock.t == pytest.approx(1800)  # alert at once (46 min), destroy 30 min later


def test_a_long_quiet_server_start_is_not_a_stall(tmp_path):
    clock = Clock()
    remote = ScriptedRemote([(True, {})] * 20 + [(False, {}), (False, {})])
    remote.log_age = 40 * 60  # a slow first start: within the 45 min threshold
    result = follow(
        FakeClient(),
        7,
        matrix(tmp_path),
        lambda info: remote,
        log=lambda _: None,
        sleep=clock.sleep,
        clock=clock,
    )
    assert result.reason == "campaign ended"


def test_alert_only_flags_but_keeps_the_node(tmp_path):
    clock = Clock()
    client = FakeClient()
    events = tmp_path / "events.jsonl"
    with pytest.raises(Stop):
        follow(
            client,
            7,
            matrix(tmp_path),
            lambda info: ScriptedRemote([(False, {})] * 200),
            events_path=events,
            log=lambda _: None,
            sleep=clock.sleep,
            clock=clock,
            safety_destroy=False,
        )
    kinds = [json.loads(line)["kind"] for line in events.read_text().splitlines()]
    assert kinds == ["start", "never_started"] and client.destroyed == 0


def test_a_stall_that_resolves_is_not_destroyed(tmp_path):
    clock = Clock()

    class Healing(ScriptedRemote):
        looks = 0

        def log_age_s(self):
            Healing.looks += 1
            return 50 * 60 if Healing.looks <= 3 else 60.0  # stalled 3 looks, then moving

    remote = Healing([(True, {})] * 8 + [(False, {}), (False, {})])
    events = tmp_path / "events.jsonl"
    result = follow(
        FakeClient(),
        7,
        matrix(tmp_path),
        lambda info: remote,
        events_path=events,
        log=lambda _: None,
        sleep=clock.sleep,
        clock=clock,
    )
    kinds = [json.loads(line)["kind"] for line in events.read_text().splitlines()]
    assert "stalled" in kinds and "recovered" in kinds and result.reason == "campaign ended"


def test_no_destroy_also_disables_the_safety_destroys(tmp_path):
    clock = Clock()
    client = FakeClient()
    remote = ScriptedRemote([(False, {})] * 20 + [(True, {}), (False, {}), (False, {})])
    result = follow(
        client,
        7,
        matrix(tmp_path),
        lambda info: remote,
        log=lambda _: None,
        sleep=clock.sleep,
        clock=clock,
        destroy_when_done=False,
    )
    # nothing may be destroyed, so it keeps following past the grace period, to the end
    assert result.reason == "campaign ended" and client.destroyed == 0


def test_ssh_remote_reads_the_log_age(tmp_path, fake_ssh):
    log = fake_ssh / "campaign.log"
    log.write_text("x\n")
    old = time.time() - 600
    os.utime(log, (old, old))
    age = SshRemote("node", 22, "repo").log_age_s()
    assert 590 <= age <= 700


def test_ssh_remote_without_a_campaign_log_judges_no_stall(tmp_path, fake_ssh):
    assert SshRemote("node", 22, "repo").log_age_s() is None


def test_a_campaign_without_log_is_still_followed_to_its_end(tmp_path):
    clock = Clock()
    remote = ScriptedRemote([(True, {"tp-rag.json": run_file("tp-rag")})] * 3 + [(False, {})] * 2)
    remote.log_age = None
    result = follow(
        FakeClient(),
        7,
        matrix(tmp_path),
        lambda info: remote,
        log=lambda _: None,
        sleep=clock.sleep,
        clock=clock,
    )
    assert result.reason == "campaign ended" and result.destroyed
