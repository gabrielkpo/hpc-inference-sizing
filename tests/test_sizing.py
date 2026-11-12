import json
from pathlib import Path

import pytest

from hpc_inference_sizing import sizing

SUMMARY = json.loads(
    (Path(__file__).resolve().parent.parent / "docs/results/a40-x2.json").read_text()
)


def step(rate, ttft50=1.0, tpot50=0.03):
    return {"rate": rate, "ttft_p50": ttft50, "tpot_p50": tpot50}


def summary():
    return {
        "runs": [
            {
                "unit": "tp",
                "profile": "rag",
                "steps": [step(0.5)],
                "capacity": {"2": {"capacity": 0.5}, "1": {"capacity": None}},
            },
            {
                "unit": "ep",
                "profile": "rag",
                "steps": [step(0.75)],
                "capacity": {"2": {"capacity": 0.75}, "1": {"capacity": None}},
            },
            {
                "unit": "ep_batch8192",
                "profile": "rag",
                "steps": [step(1.0)],
                "capacity": {"2": {"capacity": 1.0}},
            },  # a lever run: not a base unit
            {
                "unit": "repeat_ep",
                "profile": "rag",
                "mode": "repeat",
                "steps": [step(0.375), step(0.375), step(0.75)],
                "repeat_pass": {"2": {"0.375": [True, True], "0.75": [False, False, False]}},
            },
        ]
    }


def test_optimistic_and_prudent_capacities():
    cap = sizing.capacity(summary(), "rag", 2.0)
    assert cap.optimistic.capacity == 0.75 and cap.optimistic.source == "ep"  # levers excluded
    assert cap.optimistic_repeats == "passed 0 of 3 repeats"
    assert cap.prudent.capacity == 0.375
    assert cap.optimistic.response_s == pytest.approx(1.0 + 300 * 0.03)


def test_unreachable_target():
    text = sizing.report(summary(), "rag", 1.0, 60, 0.61, users=2000)
    assert "Unreachable" in text


def test_gpus_and_users_follow_the_protocol_formulas():
    p = sizing.Point(capacity=0.5, response_s=10.0, source="tp")
    assert sizing.gpus_for(2000, p, 60) == 2 * 58  # ceil(2000 / 70 / 0.5) units of 2 GPUs
    assert sizing.users_for(8, p, 60) == int(4 * 0.5 * 70)
    assert sizing.monthly_cost(100, 0.61) == pytest.approx(100 * 0.61 * 730)


def test_the_published_campaign_answers_the_two_clients():
    admin = sizing.report(SUMMARY, "rag", 2.0, 60, 0.61, users=2000)
    assert "GPUs needed (optimistic)   108" in admin and "GPUs needed (prudent)   160" in admin
    assert "Unreachable" in sizing.report(SUMMARY, "rag", 1.0, 60, 0.61, users=2000)
    defence = sizing.report(SUMMARY, "chat", 2.0, 60, 0.61, gpus=8)
    assert "about 798 active users" in defence and "about 762 active users" in defence
