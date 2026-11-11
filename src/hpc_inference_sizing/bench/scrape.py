"""Read the server's Prometheus `/metrics` every few seconds during a load step.

Only series whose name starts with one of `prefixes` are kept (`vllm:` for vLLM, `llamacpp:`
for llama.cpp), so the names do not need to be known in advance.
"""

from __future__ import annotations

import asyncio
import time

import httpx

DEFAULT_PREFIXES = ("vllm:", "llamacpp:")


def parse_prometheus(text: str, prefixes: tuple[str, ...] = DEFAULT_PREFIXES) -> dict[str, float]:
    """`name{labels}` -> value for the kept series; comments and other series are skipped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith(prefixes):
            continue
        key, _, value = line.rpartition(" ")
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


async def scrape_until(
    base_url: str,
    stop: asyncio.Event,
    t0: float,
    interval_s: float = 5.0,
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict]:
    """Samples `{"t": seconds since t0, "metrics": {...}}` until `stop` is set."""
    samples: list[dict] = []
    async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
        while not stop.is_set():
            try:
                resp = await client.get(f"{base_url}/metrics")
                if resp.status_code == 200:
                    samples.append(
                        {
                            "t": time.perf_counter() - t0,
                            "metrics": parse_prometheus(resp.text, prefixes),
                        }
                    )
            except httpx.HTTPError:
                pass  # a missing sample is not a failed step
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except TimeoutError:
                pass
    return samples
