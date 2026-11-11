"""Load generator: prompts of exact token lengths built from PIAF, and Poisson arrival times.

Profiles (protocol §3.1):

    rag   system prompt 1,000 (shared by every request) + 5 passages x 1,000 + question 100,
          output 300  ->  6,100 input tokens
    chat  one message of 500 tokens, output 300

A passage concatenates consecutive PIAF contexts of one article until it passes the target, then
is cut to it. Each request draws different articles, so only the system prompt is shared.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

SYSTEM_INSTRUCTIONS = (
    "Tu es l'assistant documentaire d'une administration. Réponds uniquement à partir des "
    "extraits fournis, en français, de façon brève et factuelle. Si les extraits ne contiennent "
    "pas la réponse, dis-le clairement. Cite l'extrait utilisé. "
)
QUESTION_INSTRUCTIONS = (
    " Réponds en une ou deux phrases, en citant le numéro de l'extrait qui contient la réponse. "
    "N'invente aucun chiffre ni aucun nom qui ne figure pas dans les extraits. "
)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class Profile:
    name: str
    input_tokens: int
    output_tokens: int
    system_tokens: int = 0
    n_passages: int = 0
    passage_tokens: int = 0


PROFILES = {
    "rag": Profile(
        "rag",
        input_tokens=6_100,
        output_tokens=300,
        system_tokens=1_000,
        n_passages=5,
        passage_tokens=1_000,
    ),
    "chat": Profile("chat", input_tokens=500, output_tokens=300),
}


def fit(tok: Tokenizer, text: str, n: int) -> str:
    """Cut `text` to at most `n` tokens (decoding can merge tokens, hence the check)."""
    out = tok.decode(tok.encode(text)[:n])
    while out and len(tok.encode(out)) > n:
        out = out[:-1]
    return out


class Workload:
    """Deterministic prompts for one profile; request `i` is the same for a given seed."""

    def __init__(self, rows: list[dict[str, str]], tok: Tokenizer, profile: Profile, seed: int):
        self.tok, self.profile, self.seed = tok, profile, seed
        by_title: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            if row["context"] not in by_title[row["title"]]:
                by_title[row["title"]].append(row["context"])
        self.articles = sorted(by_title.items())  # [(title, [contexts in order])]
        self.questions = [r["question"] for r in rows]
        self.system = self._system_prompt() if profile.system_tokens else ""

    def _system_prompt(self) -> str:
        # Instructions followed by a fixed reference text, identical for every request.
        reference = " ".join(self.articles[0][1] + self.articles[1][1])
        return fit(self.tok, SYSTEM_INSTRUCTIONS + reference, self.profile.system_tokens)

    def _passage(self, rng: random.Random, n_tokens: int) -> str:
        """Consecutive contexts from a random point; a short article continues with the next one."""
        a = rng.randrange(len(self.articles))
        title, contexts = self.articles[a]
        k = rng.randrange(len(contexts))
        text = ""
        while len(self.tok.encode(text)) < n_tokens:
            if k == len(contexts):  # article exhausted: move on to the next article
                a, k = (a + 1) % len(self.articles), 0
                contexts = self.articles[a][1]
            text += contexts[k] + " "
            k += 1
        return f"Extrait ({title}) : " + text

    def prompt(self, i: int) -> tuple[str, int]:
        """Prompt text of request `i` and its length in tokens."""
        rng = random.Random(f"{self.seed}:{self.profile.name}:{i}")
        p = self.profile
        parts = [self.system] if self.system else []
        for j in range(p.n_passages):
            parts.append(
                fit(
                    self.tok, f"[{j + 1}] " + self._passage(rng, p.passage_tokens), p.passage_tokens
                )
            )
        question = rng.choice(self.questions)
        if p.n_passages == 0:  # chat: one message made of a context and a question
            question = f"{question} Contexte : " + self._passage(rng, p.input_tokens)
        head = "\n\n".join(parts) + ("\n\n" if parts else "")
        tail = "Question : " + question + QUESTION_INSTRUCTIONS * 4
        # Give the question whatever room is left so that the whole prompt hits the target.
        room = p.input_tokens - len(self.tok.encode(head))
        text = head + fit(self.tok, tail, room)
        for _ in range(5):
            n = len(self.tok.encode(text))
            if n == p.input_tokens or room <= 0:
                break
            room += p.input_tokens - n
            text = head + fit(self.tok, tail, room)
        return text, len(self.tok.encode(text))


def poisson_arrivals(rate: float, duration_s: float, seed: int) -> list[float]:
    """Send times (seconds from the step start) of a Poisson process of `rate` requests/s."""
    rng = random.Random(f"arrivals:{seed}:{rate}")
    t, times = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            return times
        times.append(t)
