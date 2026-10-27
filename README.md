# hpc-inference-sizing

[![CI](https://github.com/gabrielkpo/hpc-inference-sizing/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrielkpo/hpc-inference-sizing/actions/workflows/ci.yml)

Answer a client's infrastructure question with measurements on real GPUs: **how many GPUs to serve a model under a latency target**, and **what latency and capacity a fixed GPU budget gives**.

> **Status:** framing done, nothing measured yet. Expect breaking changes.

## Why

An organisation that hosts an open-weight model on its own GPUs has to size the hardware before it buys or reserves it. Too few GPUs and the assistant is too slow to be used; too many and it pays for idle cards. This project measures what one serving unit sustains under a latency objective, then turns those measurements into a sizing answer for two fictional clients:

- a **public administration** that buys its GPUs: the minimum number of GPUs for a time-to-first-token target, and what a stricter target costs;
- a **defence ministry** with a fixed, isolated GPU fleet: the best latency and the number of concurrent users before the target breaks.

## Plan

| Step | Content |
|---|---|
| Model | Mistral Small 4 (MoE, 119B parameters, 6.5B active per token), NVFP4 weights (70.8 GB), served with vLLM |
| Hardware | One rented node with 2 GPUs (48 GB each); the weights do not fit on one card |
| Compared | Tensor, expert and pipeline parallel across the 2 GPUs, then vLLM serving parameters |
| Workload | RAG-shaped requests (long prompts, short answers) and chat requests, Poisson arrivals, load raised until the target breaks |
| Checks | Extrapolation from one to two serving units on 4 GPUs; quality of NVFP4 against FP8 |
| Execution | The same SLURM job scripts run on a local Docker cluster first, then on the rented node |

The protocol (hypotheses, objectives, test matrix) is written in `docs/PROTOCOL.md` and frozen before the first paid run.

## Run it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest
```

## Development

Pull requests only, with Conventional Commits titles. CI runs ruff (lint and format), pytest, a full-history gitleaks scan and a PR title check. Install the hooks with `uv run pre-commit install` (or a global `pre-commit install`).
