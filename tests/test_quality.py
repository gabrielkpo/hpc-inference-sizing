import asyncio
import json

import httpx
import pytest

from hpc_inference_sizing import campaign, quality


def test_normalisation_drops_case_punctuation_and_french_articles():
    assert quality.normalize("  L'Assemblée nationale, en 1958 !") == "assemblée nationale en 1958"
    assert quality.normalize("« Le Monde »") == "monde"


def test_exact_match_and_f1_take_the_best_reference():
    refs = ["la tour Eiffel", "Eiffel"]
    assert quality.exact_match("Eiffel.", refs) == 1.0
    assert quality.exact_match("la tour", refs) == 0.0
    assert quality.f1("la tour", refs) == pytest.approx(
        2 * 1 * 0.5 / 1.5
    )  # "tour" of "tour eiffel"
    assert quality.f1("rien", refs) == 0.0


def test_evaluate_scores_every_question_at_temperature_zero():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        answer = "Paris" if "capitale" in body["messages"][0]["content"] else "je ne sais pas"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    questions = [
        {
            "id": "1",
            "context": "Paris est la capitale.",
            "question": "La capitale ?",
            "answers": ["Paris"],
        },
        {"id": "2", "context": "Lyon.", "question": "Ville ?", "answers": ["Lyon"]},
    ]
    result = asyncio.run(
        quality.evaluate("http://fake", "m", questions, transport=httpx.MockTransport(handler))
    )
    assert result["n"] == 2 and result["exact_match"] == 50.0 and result["f1"] == 50.0
    assert all(b["temperature"] == 0 and b["max_tokens"] == 32 for b in seen)


def test_a_quality_run_writes_its_scores_in_the_run_file(tmp_path, monkeypatch):
    m = campaign.Matrix(
        name="x4",
        results_dir=tmp_path / "results",
        server_url="http://127.0.0.1:9",
        served_model="small4",
        server_command="",
        units={"qfp8": ""},
        profiles=["rag"],
        targets_s=[2.0],
        run_list=("qfp8-rag",),
        quality=("qfp8-rag",),
        quality_n=2,
    )
    m.results_dir.mkdir(parents=True)
    monkeypatch.setattr(campaign, "wait_healthy", lambda *a: 3.0)
    monkeypatch.setattr(campaign, "server_provenance", lambda *a: {"server_version": "0.32.0"})
    monkeypatch.setattr(quality, "load_questions", lambda n: [{"id": "1"}] * n)

    async def fake_evaluate(url, model, questions):
        return {"n": len(questions), "exact_match": 61.5, "f1": 74.2, "truncated": 1, "answers": []}

    monkeypatch.setattr(quality, "evaluate", fake_evaluate)
    logs = []
    assert campaign.run_one(m, "qfp8-rag", {"gpu": "4xA40"}, log=logs.append)
    doc = json.loads((m.results_dir / "qfp8-rag.json").read_text())
    assert doc["quality"]["f1"] == 74.2 and doc["metadata"]["mode"] == "quality"
    assert doc["metadata"]["server_version"] == "0.32.0" and doc["metadata"]["gpu"] == "4xA40"
    assert "PIAF exact match 61.5, F1 74.2, 1 answers cut" in logs[-1]

    from hpc_inference_sizing import analysis
    from hpc_inference_sizing.follow import capacities

    run = analysis.summarize_dir(m.results_dir, [2.0])["runs"][0]
    assert run["quality"] == {"n": 2, "exact_match": 61.5, "f1": 74.2, "truncated": 1}
    assert capacities(m.results_dir, "qfp8-rag") == "PIAF exact match 61.5, F1 74.2"


def test_a_null_answer_scores_zero_and_a_cut_one_is_counted():
    def handler(request):
        if "Lyon" in json.loads(request.content)["messages"][0]["content"]:
            choice = {"message": {"content": None}, "finish_reason": "stop"}
        else:
            choice = {"message": {"content": "Paris est"}, "finish_reason": "length"}
        return httpx.Response(200, json={"choices": [choice]})

    questions = [
        {"id": "1", "context": "Paris.", "question": "Ville ?", "answers": ["Paris"]},
        {"id": "2", "context": "Lyon.", "question": "Ville ?", "answers": ["Lyon"]},
    ]
    result = asyncio.run(
        quality.evaluate("http://fake", "m", questions, transport=httpx.MockTransport(handler))
    )
    assert result["truncated"] == 1 and result["exact_match"] == 0.0
    assert [a["prediction"] for a in result["answers"]] == ["Paris est", ""]


def test_a_failed_quality_run_leaves_a_full_failure_record(tmp_path, monkeypatch):
    m = campaign.Matrix(
        name="x4",
        results_dir=tmp_path / "results",
        server_url="http://127.0.0.1:9",
        served_model="small4",
        server_command="",
        units={"qfp8": ""},
        profiles=["rag"],
        targets_s=[2.0],
        run_list=("qfp8-rag",),
        quality=("qfp8-rag",),
    )
    m.results_dir.mkdir(parents=True)
    monkeypatch.setattr(campaign, "wait_healthy", lambda *a: 3.0)
    monkeypatch.setattr(campaign, "server_provenance", lambda *a: {})
    monkeypatch.setattr(quality, "load_questions", lambda n: [])

    async def broken(url, model, questions):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(quality, "evaluate", broken)
    assert not campaign.run_one(m, "qfp8-rag", {}, log=lambda line: None)
    failure = json.loads((m.results_dir / "qfp8-rag.failed.json").read_text())
    assert failure["stage"] == "quality" and failure["error"].startswith("ConnectError")
    assert "flags" in failure
