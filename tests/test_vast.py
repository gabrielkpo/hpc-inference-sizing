import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from hpc_inference_sizing import vast
from hpc_inference_sizing.vast import OfferQuery, RentSpec, VastClient, VastError

KEY = "test-key-not-real"


def offer(id_, price, **kw):
    return {
        "id": id_,
        "gpu_name": "A40",
        "num_gpus": 2,
        "dph_total": price,
        "gpu_ram": 46068,
        "reliability2": 0.99,
        "inet_down": 900,
        "disk_space": 400,
        "cuda_max_good": 13.0,
        "rentable": True,
        **kw,
    }


class FakeVast:
    """In-memory Vast.ai API: records every request, serves offers and one instance."""

    def __init__(
        self,
        offers=(),
        instance=None,
        destroy_fails=0,
        listed=False,
        create_fails=False,
        destroy_status=200,
    ):
        self.offers = list(offers)
        self.destroy_status = destroy_status
        self.listed = listed
        self.create_fails = create_fails
        self.instance = instance
        self.destroy_fails = destroy_fails
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.startswith("/api/") and not path.endswith("/"):
            # the live API moves paths without their trailing slash, with an HTML page
            return httpx.Response(301, text="<html>301 Moved Permanently</html>")
        if path == "/api/v0/bundles/":
            q = json.loads(request.url.params["q"])
            offers = self.offers
            if "ask_contract_id" in q:
                offers = [o for o in offers if o["id"] == q["ask_contract_id"]["eq"]]
            return httpx.Response(200, json={"offers": offers})
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": "unauthorized"})
        if request.method == "PUT" and path.startswith("/api/v0/asks/"):
            if self.create_fails:
                return httpx.Response(504, text="gateway timeout")
            return httpx.Response(200, json={"success": True, "new_contract": 777})
        if path == "/api/v0/instances/777/":
            if request.method == "DELETE":
                if self.destroy_fails:
                    self.destroy_fails -= 1
                    return httpx.Response(500, json={"error": "busy"})
                self.instance = None
                if self.destroy_status == 204:
                    return httpx.Response(204)
                if self.destroy_status != 200:
                    return httpx.Response(self.destroy_status, json={"error": "not found"})
                return httpx.Response(200, json={"success": True})
            return httpx.Response(200, json={"instances": self.instance})
        if path == "/api/v1/instances/":
            listed = [{"id": 777}] if self.listed else []
            return httpx.Response(200, json={"instances": listed})
        if path == "/api/v0/users/current/":
            return httpx.Response(200, json={"balance": 0, "credit": 19.5})
        return httpx.Response(404)

    def client(self, key=KEY):
        return VastClient(api_key=key, transport=httpx.MockTransport(self.handler))

    def calls(self, method):
        return [r for r in self.requests if r.method == method]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_search_needs_no_key_and_returns_the_cheapest_first():
    fake = FakeVast([offer(2, 1.5), offer(1, 1.3), offer(3, 2.4)])
    found = fake.client(key="").search(OfferQuery(max_dph=2.0))
    assert [o.id for o in found] == [1, 2]
    req = fake.requests[0]
    assert "authorization" not in req.headers
    q = json.loads(req.url.params["q"])
    assert q["gpu_name"] == {"eq": "A40"} and q["verified"] == {"eq": True}
    assert q["reliability2"] == {"gte": 0.98} and q["allocated_storage"] == 150


def test_the_key_is_never_shown():
    client = VastClient(api_key=KEY)
    assert KEY not in repr(client)
    spec = RentSpec(env={"HF_TOKEN": "hf_secret"})
    assert "hf_secret" not in json.dumps(spec.plan()) and "hf_secret" not in repr(spec)
    assert spec.body()["env"]["HF_TOKEN"] == "hf_secret"  # only the API request carries it


def test_a_paid_call_without_a_key_is_refused_before_any_request():
    fake = FakeVast([offer(1, 1.3)])
    with pytest.raises(VastError, match="VAST_API_KEY"):
        fake.client(key="").create(1, RentSpec())
    assert fake.requests == []


def test_rent_without_confirmation_only_prints_the_plan():
    fake = FakeVast([offer(1, 1.3)])
    logs = []
    assert (
        vast.rent(fake.client(), 1, RentSpec(), max_dph=1.5, confirm=False, log=logs.append) is None
    )
    assert fake.calls("PUT") == [] and "dry run" in logs[-1]


def test_rent_checks_the_live_price_then_creates_with_the_backstop():
    fake = FakeVast([offer(1, 1.3)])
    rental = vast.rent(fake.client(), 1, RentSpec(max_hours=2), 1.5, confirm=True, log=print)
    assert rental == vast.Rental(777, 1.3)
    body = json.loads(fake.calls("PUT")[0].content)
    assert body["image"] == "vllm/vllm-openai:v0.32.0" and body["disk"] == 150
    assert "sleep 7800" in body["onstart"] and "CONTAINER_API_KEY" in body["onstart"]


@pytest.mark.parametrize("offers, match", [([offer(1, 1.9)], "cap"), ([], "no longer")])
def test_rent_refuses_a_price_above_the_cap_or_a_gone_offer(offers, match):
    fake = FakeVast(offers)
    with pytest.raises(VastError, match=match):
        vast.rent(fake.client(), 1, RentSpec(), 1.5, confirm=True, log=print)
    assert fake.calls("PUT") == []


def run_guard(fake, wall=lambda: 1_000_000.0, **kw):
    clock = FakeClock()
    args = {"max_hours": 1.0, "budget_usd": 100.0, "poll_s": 60, "log": lambda _: None}
    args.update(kw)
    return vast.guard(
        fake.client(), 777, clock=clock, wall=lambda: wall() + clock(), sleep=clock.sleep, **args
    )


def test_guard_destroys_at_the_time_limit():
    fake = FakeVast(instance={"actual_status": "running", "dph_total": 1.3})
    result = run_guard(fake, max_hours=0.5)
    assert result.destroyed and "time limit" in result.reason
    assert result.elapsed_h == pytest.approx(0.5)
    assert len(fake.calls("DELETE")) == 1


def test_guard_destroys_at_the_budget():
    fake = FakeVast(instance={"actual_status": "running", "dph_total": 2.0})
    result = run_guard(fake, budget_usd=1.0)
    assert result.destroyed and "budget" in result.reason
    assert result.spent_usd == pytest.approx(1.0)


def test_guard_destroys_an_instance_that_stays_dead():
    fake = FakeVast(instance={"actual_status": "exited", "dph_total": 1.3})
    result = run_guard(fake, dead_grace_s=300)
    assert result.destroyed and "exited" in result.reason and result.elapsed_h < 0.2


def test_guard_stops_when_the_instance_is_already_gone():
    fake = FakeVast(instance={}, destroy_status=404)
    result = run_guard(fake)
    assert not result.destroyed and "no longer exists" in result.reason
    assert len(fake.calls("DELETE")) == 1  # a destroy is always tried before giving up


def test_an_instance_the_api_does_not_report_yet_is_destroyed_anyway():
    # just after creation the API may answer null and not list it: the guard still destroys
    fake = FakeVast(instance={})
    result = run_guard(fake)
    assert result.destroyed and "to be sure" in result.reason


def test_a_destroy_answered_without_a_body_is_a_success():
    fake = FakeVast(instance={"actual_status": "running"}, destroy_status=204)
    logs = []
    assert vast.destroy_with_retries(fake.client(), 777, log=logs.append, sleep=lambda _: None)
    assert len(fake.calls("DELETE")) == 1


def test_guard_destroys_on_ctrl_c():
    fake = FakeVast(instance={"actual_status": "running", "dph_total": 1.3})
    result = vast.guard(
        fake.client(),
        777,
        max_hours=1.0,
        budget_usd=100.0,
        log=lambda _: None,
        sleep=lambda _: os.kill(os.getpid(), signal.SIGINT),
    )
    assert result.destroyed and "SIGINT" in result.reason


def test_destroy_retries_then_tells_to_destroy_by_hand():
    fake = FakeVast(instance={"actual_status": "running"}, destroy_fails=2)
    assert vast.destroy_with_retries(fake.client(), 777, log=print, sleep=lambda _: None)
    fake = FakeVast(instance={"actual_status": "running"}, destroy_fails=9)
    logs = []
    assert not vast.destroy_with_retries(fake.client(), 777, log=logs.append, sleep=lambda _: None)
    assert "by hand" in logs[-1]


def test_balance_and_ssh_command():
    fake = FakeVast()
    assert fake.client().balance() == 19.5
    assert vast.ssh_command({"ssh_host": "ssh4.vast.ai", "ssh_port": 2222}) == (
        "ssh -p 2222 root@ssh4.vast.ai"
    )


@pytest.mark.parametrize("curl_works", [True, False])
def test_the_backstop_destroys_the_instance_from_inside(tmp_path, curl_works):
    """Runs the instance start script for real against a local stand-in for the API."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_DELETE(self):  # noqa: N802
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"success": true}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "CONTAINER_ID": "42",
        "CONTAINER_API_KEY": "instance-key",
        "VAST_API_URL": f"http://127.0.0.1:{server.server_port}",
    }
    if not curl_works:  # a fake curl that fails: the Python fallback must do it
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "curl").write_text("#!/bin/sh\nexit 1\n")
        (bin_dir / "curl").chmod(0o755)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
    script = RentSpec(max_hours=0, backstop_margin_s=0).onstart()
    try:
        subprocess.run(["bash", "-c", script], env=env, check=True, timeout=10)
        for _ in range(100):
            if seen:
                break
            time.sleep(0.1)
    finally:
        server.shutdown()
    assert seen == [("/api/v0/instances/42/", "Bearer instance-key")]


def test_a_restarted_guard_counts_the_instance_age():
    # the instance started 50 min ago; a new guard with --max-hours 1 must stop it in ~10 min
    fake = FakeVast(
        instance={"actual_status": "running", "dph_total": 1.3, "start_date": 1_000_000 - 3000}
    )
    result = run_guard(fake, max_hours=1.0)
    assert result.destroyed and "time limit" in result.reason
    assert result.elapsed_h == pytest.approx(1.0, abs=0.02)
    assert result.spent_usd == pytest.approx(1.3, abs=0.05)


def test_an_empty_read_of_a_listed_instance_keeps_guarding():
    fake = FakeVast(instance={}, listed=True)
    result = run_guard(fake, max_hours=0.1)
    assert result.destroyed and "time limit" in result.reason


def test_the_budget_counts_from_the_offer_price_while_the_api_is_unreadable():
    clock = FakeClock()
    result = vast.guard(
        FakeVast().client(key="wrong-key"),  # every read (and the destroy) fails with 401
        777,
        max_hours=10,
        budget_usd=1.0,
        price_hint=2.0,
        log=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert "budget" in result.reason and not result.destroyed
    assert result.elapsed_h == pytest.approx(0.5, abs=0.02)


def test_a_failed_creation_says_to_check_for_a_live_instance():
    fake = FakeVast([offer(1, 1.3)], create_fails=True)
    logs = []
    with pytest.raises(VastError, match="504"):
        vast.rent(fake.client(), 1, RentSpec(), 1.5, confirm=True, log=logs.append)
    assert "vast list" in logs[-1]


def test_a_fresh_rental_ignores_start_date():
    # until start_date is checked on the test session, the rent path does not use it
    fake = FakeVast(instance={"actual_status": "running", "dph_total": 1.3, "start_date": 1.0})
    result = run_guard(fake, max_hours=0.5, count_instance_age=False)
    assert result.elapsed_h == pytest.approx(0.5)


def test_an_unexpected_error_in_the_guard_still_destroys():
    fake = FakeVast(instance={"actual_status": "running", "dph_total": "not-a-number"})
    result = run_guard(fake)
    assert result.destroyed and "guard error" in result.reason
    assert len(fake.calls("DELETE")) == 1


def test_the_instance_list_is_read_from_the_live_path():
    assert FakeVast(listed=True).client().instances() == [{"id": 777}]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(301, text="<html>301 Moved Permanently</html>"),
        httpx.Response(200, text="<html>maintenance</html>"),
        httpx.Response(200, json=[]),
    ],
)
def test_a_redirect_or_a_non_json_answer_is_an_error_not_an_empty_result(response):
    client = VastClient(api_key=KEY, transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(VastError):
        client.instances()
    # the guard cannot confirm the instance is gone, so it keeps guarding
    assert vast._listed(client, 777)
