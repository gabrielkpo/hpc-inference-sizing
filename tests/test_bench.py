import asyncio
import json

import pytest

from fake_server import FakeServer
from hpc_inference_sizing.bench.client import run_step, send
from hpc_inference_sizing.bench.runner import Bench, metadata, write_run
from hpc_inference_sizing.bench.scrape import parse_prometheus
from hpc_inference_sizing.bench.stats import RequestRecord, StepSummary, percentile, summarize
from hpc_inference_sizing.bench.sweep import sweep
from hpc_inference_sizing.bench.workload import PROFILES, Profile, Workload, fit, poisson_arrivals


class CharTokenizer:
    """One token per character: every expected length can be checked by hand."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


ROWS = [
    {
        "title": f"article {a}",
        "context": f"Contexte {a}.{k} " + "texte " * 30,
        "question": f"Q{a}{k} ?",
    }
    for a in range(6)
    for k in range(3)
]


# --- statistics ---------------------------------------------------------------------------


def test_percentile_matches_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([5], 95) == 5


def test_request_ttft_and_tpot():
    r = RequestRecord(sent_at=1.0, first_token_at=1.5, last_token_at=3.5, completion_tokens=5)
    assert r.ttft == 0.5
    assert r.tpot == 0.5  # 2 s over 4 intervals


def test_slo_needs_ttft_tpot_and_no_error():
    good = [RequestRecord(0, 0.5, 2.0, 7, 31) for _ in range(20)]  # TPOT 50 ms
    s = summarize(good, rate=1, duration_s=10)
    assert s.meets(1.0) and not s.meets(0.4)
    slow = [RequestRecord(0, 0.5, 4.0, 7, 31) for _ in range(20)]  # TPOT ~117 ms
    assert not summarize(slow, 1, 10).meets(3.0)
    with_error = good + [RequestRecord(0, error="HTTP 503")]
    assert not summarize(with_error, 1, 10).meets(3.0)


# --- workload -----------------------------------------------------------------------------


def test_rag_prompts_have_the_exact_length_and_share_the_system_prompt():
    profile = Profile("rag", 800, 10, system_tokens=200, n_passages=3, passage_tokens=150)
    w = Workload(ROWS, CharTokenizer(), profile, seed=1)
    a, na = w.prompt(0)
    b, nb = w.prompt(1)
    assert na == nb == 800
    assert len(w.system) == 200
    assert a.startswith(w.system) and b.startswith(w.system)
    assert a != b


def test_prompts_are_deterministic_for_a_seed():
    w1 = Workload(ROWS, CharTokenizer(), PROFILES["chat"], seed=3)
    w2 = Workload(ROWS, CharTokenizer(), PROFILES["chat"], seed=3)
    w3 = Workload(ROWS, CharTokenizer(), PROFILES["chat"], seed=4)
    assert w1.prompt(5) == w2.prompt(5)
    assert w1.prompt(5) != w3.prompt(5)
    assert w1.prompt(5)[1] == 500


def test_fit_never_exceeds_the_budget():
    assert fit(CharTokenizer(), "abcdef", 4) == "abcd"
    assert fit(CharTokenizer(), "ab", 4) == "ab"


def test_poisson_arrivals_have_the_expected_rate():
    times = poisson_arrivals(rate=5.0, duration_s=200, seed=1)
    assert all(0 < t < 200 for t in times) and times == sorted(times)
    assert 900 < len(times) < 1100  # 1,000 expected


# --- client -------------------------------------------------------------------------------


def test_send_times_first_token_and_tokens():
    import time

    import httpx

    server = FakeServer(first_delay=0.05, token_delay=0.01)

    async def go(t0):
        async with httpx.AsyncClient(transport=server.transport) as client:
            return await send(client, "http://fake", "m", "hello", 11, t0=t0)

    rec = asyncio.run(go(time.perf_counter()))
    assert rec.error is None
    assert rec.completion_tokens == 11 and rec.prompt_tokens == 7
    assert 0 <= rec.sent_at < 0.05  # measured from t0, not from an absolute clock
    assert 0.04 < rec.ttft < 0.15
    assert 0.008 < rec.tpot < 0.03
    body = server.requests[0]
    assert body["stream"] and body["max_tokens"] == body["min_tokens"] == 11 and body["ignore_eos"]


def test_send_records_a_server_error_event_mid_stream():
    import httpx

    server = FakeServer(first_delay=0.0, token_delay=0.0, error_after=3)

    async def go():
        async with httpx.AsyncClient(transport=server.transport) as client:
            return await send(client, "http://fake", "m", "x", 10, t0=0.0)

    rec = asyncio.run(go())
    assert rec.error.startswith("server error") and "out of KV cache" in rec.error


def test_send_records_http_errors():
    server = FakeServer(status=503)

    async def go():
        import httpx

        async with httpx.AsyncClient(transport=server.transport) as client:
            return await send(client, "http://fake", "m", "x", 5, t0=0.0)

    rec = asyncio.run(go())
    assert rec.error.startswith("HTTP 503")


def test_send_counts_chunks_when_the_server_sends_no_usage():
    server = FakeServer(usage=False, first_delay=0.0, token_delay=0.0)

    async def go():
        import httpx

        async with httpx.AsyncClient(transport=server.transport) as client:
            return await send(client, "http://fake", "m", "x", 4, t0=0.0)

    assert asyncio.run(go()).completion_tokens == 4


def test_run_step_sends_the_poisson_arrivals():
    server = FakeServer(first_delay=0.0, token_delay=0.0)
    records = asyncio.run(
        run_step(
            "http://fake",
            "m",
            lambda i: f"p{i}",
            3,
            rate=40,
            duration_s=0.5,
            seed=2,
            transport=server.transport,
        )
    )
    assert len(records) == len(poisson_arrivals(40, 0.5, 2)) == len(server.requests)
    assert all(r.error is None for r in records)
    assert {b["prompt"] for b in server.requests} == {f"p{i}" for i in range(len(records))}


# --- sweep --------------------------------------------------------------------------------


def fake_step_factory():
    """A step whose TTFT p95 equals its rate: target T is met if and only if rate <= T."""
    calls = []

    async def step(rate):
        calls.append(rate)
        return StepSummary(rate, 1.0, 10, 0, rate, rate, 0.01, 0.01, 100.0)

    return step, calls


def test_sweep_ladder_then_bisection_per_target():
    step, calls = fake_step_factory()
    result = asyncio.run(sweep(step, [0.75, 3.0], start_rate=0.25, max_rate=64, bisect_steps=2))
    # ladder 0.25, 0.5, 1, 2, 4 (4 breaks the 3 s target)
    assert calls[:5] == [0.25, 0.5, 1, 2, 4]
    assert result.capacity[0.75] == 0.75  # 0.5..1 -> 0.75 passes -> 0.875 fails
    assert result.capacity[3.0] == 3.0  # 2..4 -> 3 passes -> 3.5 fails
    assert len(calls) == len(set(calls))  # no rate measured twice


def test_sweep_reports_no_capacity_and_max_rate():
    step, _ = fake_step_factory()
    result = asyncio.run(sweep(step, [0.1, 3.0], start_rate=0.25, max_rate=2))
    assert result.capacity[0.1] is None  # fails even at the lowest rate
    assert result.reached_max_rate and result.capacity[3.0] == 2


# --- scraping and run file ----------------------------------------------------------------


def test_parse_prometheus_keeps_only_the_prefixes():
    text = '# HELP\nvllm:num_requests_waiting{model="m"} 2\nvllm:x 1.5\nprocess_cpu 9\nbad line\n'
    assert parse_prometheus(text) == {'vllm:num_requests_waiting{model="m"}': 2.0, "vllm:x": 1.5}


def test_low_rate_steps_last_long_enough_for_min_requests():
    server = FakeServer(first_delay=0.0, token_delay=0.0)
    bench = Bench(
        "http://fake",
        "m",
        Workload(ROWS, CharTokenizer(), Profile("chat", 20, 2), 0),
        step_duration_s=0.05,
        min_requests=10,
        scrape_interval_s=0.05,
        transport=server.transport,
    )
    summary = asyncio.run(bench.step(40))  # 0.05 s x 40/s = 2 expected; lengthened to 0.25 s
    assert summary.duration_s == pytest.approx(0.25)
    assert summary.n_requests >= 3


def test_bench_step_writes_a_complete_run_file(tmp_path):
    server = FakeServer(first_delay=0.01, token_delay=0.0)
    profile = Profile("chat", 50, 5)
    bench = Bench(
        "http://fake",
        "m",
        Workload(ROWS, CharTokenizer(), profile, 0),
        step_duration_s=0.3,
        scrape_interval_s=0.1,
        transport=server.transport,
    )
    summary = asyncio.run(bench.step(20))
    assert summary.n_requests == len(server.requests) > 0
    path = tmp_path / "run.json"
    write_run(path, metadata(bench, {"gpu": "fake"}), bench, None)
    doc = json.loads(path.read_text())
    assert doc["metadata"]["gpu"] == "fake" and doc["metadata"]["profile"] == "chat"
    assert "host" not in doc["metadata"]  # no hostname unless given with --meta
    step = doc["steps"][0]
    assert len(step["requests"]) == summary.n_requests
    assert step["metrics"] and "vllm:num_requests_running" in step["metrics"][0]["metrics"]
    # requests and /metrics samples share the same time origin
    assert 0 <= step["metrics"][0]["t"] < 0.1 and 0 <= step["requests"][0]["sent_at"] < 0.3
    assert not list(tmp_path.glob("*.part"))


def test_sweep_reports_each_step_as_it_is_measured():
    seen = []
    step, _ = fake_step_factory()
    asyncio.run(sweep(step, [0.75, 3.0], start_rate=0.25, max_rate=64, on_step=seen.append))
    assert [s.rate for s in seen][:4] == [0.25, 0.5, 1.0, 2.0]


def test_a_step_measured_for_a_looser_target_counts_for_the_stricter_ones():
    # TTFT never binds; TPOT breaks above 2.58 req/s: every target has the same capacity
    async def step(rate):
        tpot = 0.05 if rate <= 2.58 else 0.08
        return StepSummary(rate, 180, int(rate * 180), 0, 0.2, 0.3, tpot, tpot, 100.0)

    result = asyncio.run(sweep(step, [1.0, 1.5, 2.0, 3.0], start_rate=0.125, max_rate=64))
    assert set(result.capacity.values()) == {2.578125}
    assert list(result.to_dict()["capacity_rps"]) == ["1.0", "1.5", "2.0", "3.0"]


def test_a_step_records_the_rate_actually_sent():
    s = StepSummary(1.0, 180, 148, 0, 1.0, 2.0, 0.05, 0.1, 100.0)
    assert s.achieved_rate == pytest.approx(0.822, abs=0.001)
    assert s.to_dict()["achieved_rate"] == pytest.approx(0.822, abs=0.001)


def test_the_warm_up_loads_the_server_but_records_nothing():
    server = FakeServer(first_delay=0.0, token_delay=0.0)
    bench = Bench(
        "http://fake",
        "m",
        Workload(ROWS, CharTokenizer(), Profile("chat", 20, 2), 0),
        step_duration_s=0.2,
        min_requests=1,
        scrape_interval_s=0.05,
        transport=server.transport,
    )
    sent = asyncio.run(bench.warm_up(50, 0.2))
    assert sent > 0 and bench.steps == []
    asyncio.run(bench.step(50))
    assert len(bench.steps) == 1  # the first measured step keeps seed index 0
