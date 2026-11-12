"""Answer quality on PIAF (protocol §9, amendment A10): does the compressed NVFP4 model answer as
well as the FP8 one?

200 questions drawn with a fixed seed; each prompt is the context passage and the question, asked
for a short answer, at temperature 0 through the chat endpoint (the model's own template). Scores
are exact match and token-level F1 against the reference answers, with the same normalisation for
both formats: lower case, punctuation removed, French articles removed, spaces collapsed.
"""

from __future__ import annotations

import asyncio
import random
import re
import string
import unicodedata
from collections import Counter

import httpx

PROMPT = (
    "Contexte :\n{context}\n\nQuestion : {question}\n\n"
    "Réponds en quelques mots seulement, avec les mots du contexte."
)
ARTICLES = re.compile(r"\b(le|la|les|l|un|une|des|du|de|d)\b")


def load_questions(n: int = 200, seed: int = 0) -> list[dict]:
    """`n` PIAF questions with their reference answers, drawn with a fixed seed."""
    import pyarrow.parquet as pq

    from hpc_inference_sizing.bench.data import PIAF_URL, fetch

    table = pq.read_table(fetch(PIAF_URL, "piaf-train.parquet"))
    cols = table.select(["id", "context", "question", "answers"]).to_pydict()
    rows = [
        {"id": i, "context": c, "question": q, "answers": list(a["text"])}
        for i, c, q, a in zip(
            cols["id"], cols["context"], cols["question"], cols["answers"], strict=True
        )
        if a["text"]
    ]
    return random.Random(seed).sample(rows, n)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = text.replace("’", "'")
    text = "".join(" " if ch in string.punctuation or ch in "«»" else ch for ch in text)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    p = normalize(prediction)
    return float(any(p == normalize(r) for r in references))


def f1(prediction: str, references: list[str]) -> float:
    def one(ref: str) -> float:
        pred_tokens, ref_tokens = normalize(prediction).split(), normalize(ref).split()
        common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
        if not pred_tokens or not ref_tokens or not common:
            return float(pred_tokens == ref_tokens)
        precision, recall = common / len(pred_tokens), common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(one(r) for r in references)


async def _ask(client: httpx.AsyncClient, url: str, model: str, row: dict, limit) -> dict:
    async with limit:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT.format(**row)}],
                "temperature": 0,
                "max_tokens": 32,
            },
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        answer = (choice["message"].get("content") or "").strip()  # null: scored as empty
    return {
        "id": row["id"],
        "prediction": answer,
        "truncated": choice.get("finish_reason") == "length",
        "references": row["answers"],
        "em": exact_match(answer, row["answers"]),
        "f1": f1(answer, row["answers"]),
    }


async def evaluate(
    url: str,
    model: str,
    questions: list[dict],
    concurrency: int = 8,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    limit = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120, transport=transport) as client:
        answers = await asyncio.gather(*(_ask(client, url, model, q, limit) for q in questions))
    n = len(answers)
    return {
        "n": n,
        "exact_match": round(100 * sum(a["em"] for a in answers) / n, 2),
        "f1": round(100 * sum(a["f1"] for a in answers) / n, 2),
        "truncated": sum(a["truncated"] for a in answers),  # cut at max_tokens
        "answers": answers,
    }
