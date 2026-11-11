"""An in-memory OpenAI-compatible server with controlled delays, for the bench tests."""

from __future__ import annotations

import asyncio
import json

import httpx


class _Stream(httpx.AsyncByteStream):
    def __init__(self, first_delay: float, token_delay: float, n_tokens: int, usage: bool):
        self.first_delay, self.token_delay = first_delay, token_delay
        self.n_tokens, self.usage = n_tokens, usage

    async def __aiter__(self):
        await asyncio.sleep(self.first_delay)
        for i in range(self.n_tokens):
            if i:
                await asyncio.sleep(self.token_delay)
            event = {"choices": [{"index": 0, "text": f"t{i} "}]}
            yield f"data: {json.dumps(event)}\n\n".encode()
        if self.usage:
            event = {
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": self.n_tokens},
            }
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"


class FakeServer:
    """`transport` answers /v1/completions with a timed stream and /metrics with Prometheus text."""

    def __init__(self, first_delay=0.05, token_delay=0.01, status=200, usage=True):
        self.first_delay, self.token_delay = first_delay, token_delay
        self.status, self.usage = status, usage
        self.requests: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            text = (
                "# HELP x\nvllm:num_requests_running 3.0\nvllm:gpu_cache_usage_perc 0.5\nother 1\n"
            )
            return httpx.Response(200, text=text)
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, text="overloaded")
        stream = _Stream(self.first_delay, self.token_delay, body["max_tokens"], self.usage)
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
