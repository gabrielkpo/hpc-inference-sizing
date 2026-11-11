"""Streaming client for an OpenAI-compatible `/v1/completions` endpoint, and one load step.

Each request asks for exactly `output_tokens` tokens (`max_tokens` = `min_tokens`, end of
sequence ignored) so that every request does the same work, and is timed on its stream: time of
the first non-empty chunk, time of the last one, token counts from the final `usage` chunk.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import httpx

from hpc_inference_sizing.bench.stats import RequestRecord
from hpc_inference_sizing.bench.workload import poisson_arrivals


async def send(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    t0: float,
    timeout_s: float = 300.0,
) -> RequestRecord:
    """Send one streaming completion; times are seconds since `t0` (time.perf_counter)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = RequestRecord(sent_at=time.perf_counter() - t0)
    chunks = 0
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/completions", json=payload, timeout=timeout_s
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:200]
                rec.error = f"HTTP {resp.status_code}: {body}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:  # e.g. vLLM aborting a request mid-stream
                    rec.error = f"server error: {json.dumps(event['error'])[:200]}"
                    return rec
                choices = event.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter() - t0
                    if rec.first_token_at is None:
                        rec.first_token_at = now
                    rec.last_token_at = now
                    chunks += 1
                if usage := event.get("usage"):
                    rec.prompt_tokens = usage.get("prompt_tokens")
                    rec.completion_tokens = usage.get("completion_tokens")
    except (httpx.HTTPError, json.JSONDecodeError) as err:
        rec.error = f"{type(err).__name__}: {err}"
        return rec
    if rec.completion_tokens is None:  # server sent no usage: one chunk per token
        rec.completion_tokens = chunks
    if not rec.completion_tokens:
        rec.error = "no token received"
    return rec


async def run_step(
    base_url: str,
    model: str,
    prompt_for: Callable[[int], str],
    output_tokens: int,
    rate: float,
    duration_s: float,
    seed: int,
    first_index: int = 0,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = 300.0,
    t0: float | None = None,
) -> list[RequestRecord]:
    """Send Poisson arrivals at `rate` for `duration_s`, then wait for every answer.

    Prompts are built before the step starts so that tokenizing does not delay the send times.
    `t0` lets the caller share the time origin with other measurements (the /metrics scraper).
    """
    arrivals = poisson_arrivals(rate, duration_s, seed)
    prompts = [prompt_for(first_index + i) for i in range(len(arrivals))]
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(limits=limits, transport=transport) as client:
        if t0 is None:
            t0 = time.perf_counter()

        async def fire(at: float, prompt: str) -> RequestRecord:
            await asyncio.sleep(max(0.0, at - (time.perf_counter() - t0)))
            return await send(client, base_url, model, prompt, output_tokens, t0, timeout_s)

        return list(
            await asyncio.gather(*(fire(a, p) for a, p in zip(arrivals, prompts, strict=True)))
        )
