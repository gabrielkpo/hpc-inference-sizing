"""Public inputs of the bench, downloaded once into a local cache, at pinned revisions.

- PIAF (Etalab, French question answering, MIT licence): real French prose for the prompts.
- The Mistral Small 4 tokenizer (tekken.json): exact prompt lengths in the served model's tokens.

Nothing here is committed to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

HF = "https://huggingface.co"
PIAF_URL = (
    f"{HF}/datasets/AgentPublic/piaf/resolve/bda8c063bc7297180796cd835d1974c0bc71c521"
    "/plain_text/train-00000-of-00001.parquet"
)
TOKENIZER_URL = (
    f"{HF}/mistralai/Mistral-Small-4-119B-2603-NVFP4/resolve/"
    "45331841b631f4e281df8e959ea3cc9beb84298a/tekken.json"
)


def cache_dir() -> Path:
    path = Path(os.environ.get("HPC_SIZING_CACHE", Path.home() / ".cache" / "hpc-inference-sizing"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(url: str, name: str) -> Path:
    """Download `url` into the cache as `name` unless it is already there."""
    dest = cache_dir() / name
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def load_piaf(path: Path | None = None) -> list[dict[str, str]]:
    """PIAF rows as dicts with `title`, `context` and `question`."""
    import pyarrow.parquet as pq

    table = pq.read_table(path or fetch(PIAF_URL, "piaf-train.parquet"))
    cols = table.select(["title", "context", "question"]).to_pydict()
    return [
        {"title": t, "context": c, "question": q}
        for t, c, q in zip(cols["title"], cols["context"], cols["question"], strict=True)
    ]


class MistralTokenizer:
    """Mistral's official tokenizer (mistral-common), without BOS/EOS tokens."""

    def __init__(self, path: Path | None = None) -> None:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer

        self._tok = Tekkenizer.from_file(str(path or fetch(TOKENIZER_URL, "small4-tekken.json")))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, bos=False, eos=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
