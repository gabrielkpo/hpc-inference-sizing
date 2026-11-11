"""Measurement bench: load generator, streaming client, SLO statistics, sweep, /metrics scraper.

It only talks to an OpenAI-compatible `/v1/completions` endpoint, so the same code runs against a
small model served by llama.cpp on a laptop and against vLLM on rented GPUs.
"""
