"""The `hpc-sizing` command wiring: arguments reach the right code, errors are one line."""

import json
from pathlib import Path

import pytest

from hpc_inference_sizing import cli, vast

ROOT = Path(__file__).resolve().parent.parent

MATRIX = """
name = "t"
results_dir = "{results}"
profiles = ["chat"]
targets_s = [2.0]

[server]
url = "http://127.0.0.1:9"
command = ""

[units.tp]
flags = "--tensor-parallel-size 2"
"""


def run(argv, capsys):
    cli.main(argv)
    return capsys.readouterr().out


def test_memory_prints_the_plan_and_rejects_an_impossible_split(capsys):
    assert "KV cache" in run(["memory", "--split", "pp"], capsys)
    with pytest.raises(SystemExit):
        cli.main(["memory", "--gpus", "1", "--split", "tp"])


def test_campaign_list_and_status(tmp_path, capsys):
    matrix = tmp_path / "m.toml"
    matrix.write_text(MATRIX.format(results=tmp_path / "results"))
    assert run(["campaign", "list", "--matrix", str(matrix)], capsys).split() == ["tp-chat"]
    out = run(["campaign", "status", "--matrix", str(matrix)], capsys)
    assert "pending (1): tp-chat" in out


def test_results_summarize_writes_the_summary_and_prints_tables(tmp_path, capsys):
    step = {
        "rate": 0.5,
        "duration_s": 180,
        "n_requests": 90,
        "n_errors": 0,
        "ttft_p50": 0.5,
        "ttft_p95": 1.0,
        "tpot_p50": 0.02,
        "tpot_p95": 0.04,
    }
    meta = {
        "run_id": "tp-rag",
        "unit": "tp",
        "profile": "rag",
        "server_flags": "--tensor-parallel-size 2",
        "git_commit": "abc1234",
    }
    (tmp_path / "tp-rag.json").write_text(
        json.dumps({"metadata": meta, "steps": [{"summary": step}]})
    )
    out_file = tmp_path / "out" / "s.json"
    out = run(["results", "summarize", "--dir", str(tmp_path), "--out", str(out_file)], capsys)
    assert "| tp |" in out and "summary of 1 runs" in out
    assert json.loads(out_file.read_text())["runs"][0]["capacity"]["2"]["capacity"] == 0.5


def test_quality_diff_on_the_published_summary(capsys):
    summary = str(ROOT / "docs/results/a40-x4.json")
    out = run(["results", "quality-diff", "--summary", summary], capsys)
    assert "F1 qnvfp4-rag - qfp8-rag: +0.42 points" in out


def test_quality_diff_without_scores_is_an_error(capsys):
    summary = str(ROOT / "docs/results/a40-x2.json")
    with pytest.raises(SystemExit):
        cli.main(["results", "quality-diff", "--summary", summary])
    assert "no per-question scores" in capsys.readouterr().err


def test_bench_rejects_malformed_metadata(capsys):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "bench",
                "step",
                "--url",
                "http://x",
                "--model",
                "m",
                "--rate",
                "1",
                "--meta",
                "novalue",
            ]
        )
    assert "KEY=VALUE" in capsys.readouterr().err


class FakeVast:
    def __init__(self, *a, **kw):
        pass

    def balance(self):
        return 5.49

    def instances(self):
        return [
            {
                "id": 1,
                "label": "hpc-sizing",
                "actual_status": "running",
                "num_gpus": 2,
                "gpu_name": "A40",
            }
        ]

    def show(self, instance):
        return {"actual_status": "running", "dph_total": 1.35}


def test_vast_read_commands(monkeypatch, capsys):
    monkeypatch.setattr(vast, "VastClient", FakeVast)
    assert "credit: 5.49 USD" in run(["vast", "balance"], capsys)
    assert "1  hpc-sizing  running  2 x A40" in run(["vast", "list"], capsys)
    assert "1: status running, 1.35 USD/h" in run(["vast", "show", "1"], capsys)
