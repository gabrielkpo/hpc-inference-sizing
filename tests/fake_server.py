"""An in-memory OpenAI-compatible server with controlled delays, for the bench tests."""

from __future__ import annotations

import asyncio
import json

import httpx


class _Stream(httpx.AsyncByteStream):
    def __init__(self, first_delay, token_delay, n_tokens, usage, error_after=None):
        self.first_delay, self.token_delay = first_delay, token_delay
        self.n_tokens, self.usage, self.error_after = n_tokens, usage, error_after

    async def __aiter__(self):
        await asyncio.sleep(self.first_delay)
        for i in range(self.n_tokens):
            if i:
                await asyncio.sleep(self.token_delay)
            if i == self.error_after:
                error = {"error": {"message": "out of KV cache", "code": 500}}
                yield f"data: {json.dumps(error)}\n\n".encode()
                return
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

    def __init__(
        self, first_delay=0.05, token_delay=0.01, status=200, usage=True, error_after=None
    ):
        self.first_delay, self.token_delay = first_delay, token_delay
        self.status, self.usage, self.error_after = status, usage, error_after
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
        stream = _Stream(
            self.first_delay, self.token_delay, body["max_tokens"], self.usage, self.error_after
        )
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
