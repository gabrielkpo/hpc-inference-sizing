import json
from pathlib import Path

import pytest

from hpc_inference_sizing import sizing
from hpc_inference_sizing.cli import main

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = json.loads((ROOT / "docs/results/a40-x2.json").read_text())
PAIR = json.loads((ROOT / "docs/results/a40-x4.json").read_text())


def step(rate, ttft50=1.0, tpot50=0.03):
    return {"rate": rate, "ttft_p50": ttft50, "tpot_p50": tpot50}


def summary():
    return {
        "targets_s": [1.0, 2.0],
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
        ],
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


def test_the_prudent_bound_is_the_best_of_several_repeated_units():
    data = summary()
    data["runs"].append(
        {
            "unit": "repeat_tp",
            "profile": "rag",
            "mode": "repeat",
            "steps": [step(0.25)],
            "repeat_pass": {"2": {"0.25": [True, True]}},
        }
    )
    assert sizing.capacity(data, "rag", 2.0).prudent.capacity == 0.375


def test_a_target_that_was_not_measured_is_an_error_not_unreachable():
    with pytest.raises(ValueError, match="2.5 s was not measured"):
        sizing.report(SUMMARY, "rag", 2.5, 60, 0.61, users=2000)


def test_units_absent_from_the_summary_are_an_error():
    with pytest.raises(ValueError, match="no rag run"):
        sizing.report(PAIR, "rag", 2.0, 60, 0.61, users=2000)  # default units: 2-GPU layouts


def test_the_pair_behind_a_balancer_is_sized_by_the_tool():
    one = dict(units=("two",), gpus_per_unit=4)
    assert "GPUs needed (optimistic)   248" in sizing.report(
        PAIR, "rag", 1.0, 60, 0.61, 2000, **one
    )
    assert "GPUs needed (optimistic)   80" in sizing.report(PAIR, "rag", 2.0, 60, 0.61, 2000, **one)
    assert "about 206 active users" in sizing.report(PAIR, "rag", 2.0, 60, 0.61, gpus=8, **one)
    single = sizing.report(PAIR, "rag", 2.0, 60, 0.61, 2000, units=("one",))
    assert "GPUs needed (optimistic)   120" in single


@pytest.mark.parametrize(
    "kwargs", [{"users": 0}, {"users": 10, "think_s": -20}, {"gpus": 1}, {"gpus": 8, "price": -1}]
)
def test_impossible_inputs_are_refused(kwargs):
    think, price = kwargs.pop("think_s", 60), kwargs.pop("price", 0.61)
    with pytest.raises(ValueError):
        sizing.report(SUMMARY, "chat", 2.0, think, price, **kwargs)


def test_a_bisection_rate_that_passed_every_repeat_does_not_crash():
    data = summary()
    data["runs"][3]["steps"] = [step(2.578125)] * 3
    data["runs"][3]["repeat_pass"] = {"2": {"2.578125": [True, True, True]}}
    assert sizing.capacity(data, "rag", 2.0).prudent.capacity == 2.578125


def test_the_cli_reports_a_missing_summary_in_one_line(capsys):
    with pytest.raises(SystemExit):
        main(["size", "--summary", "nope.json", "--users", "10"])
    assert "cannot read the summary nope.json" in capsys.readouterr().err


def test_the_cli_prints_the_readme_example(capsys):
    main(["size", "--summary", str(ROOT / "docs/results/a40-x2.json"), "--users", "2000"])
    assert "GPUs needed (prudent)   160" in capsys.readouterr().out
