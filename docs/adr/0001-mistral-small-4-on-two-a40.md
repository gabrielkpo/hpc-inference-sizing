# ADR 0001: Mistral Small 4 (NVFP4) on 2 × A40

- Status: accepted (2026-10-27)
- Decided by: the project owner, after comparing it with Mistral Small 3.2 on 2 × RTX A6000

## Context

The project measures how many GPUs a client needs to serve an open-weight Mistral model under a
latency target. The first plan used **Mistral Small 3.2 24B** (dense, June 2025) on a rented
2 × RTX A6000 node. Checking Hugging Face on 2026-10-27 showed that the current "Small" is
**Mistral Small 4** (January 2026): a MoE with 128 experts, 4 active plus 1 shared, 119B
parameters and 6.5B active per token, with MLA attention.

| | Small 3.2 24B | Small 4 119B |
|---|---|---|
| Weights | ~48 GB in bf16, ~25 GB in 8 bits | 70.8 GB in NVFP4 (~119 GB in FP8) |
| Fits on one 48 GB card | yes, in 8 bits | no |
| KV cache per token (bf16) | ~160 KB (40 layers × 8 KV heads × 128 × 2 × 2 bytes) | ~23 KB with MLA ((256 + 64) × 36 layers × 2 bytes) |
| What a client deploys in late 2026 | a 16-month-old model | the current one |

## Decision

Serve **`mistralai/Mistral-Small-4-119B-2603-NVFP4`** with vLLM on **2 × A40** (48 GB each, Ampere,
PCIe 4.0) rented on Vast.ai, and compare the three ways of splitting a MoE across the two GPUs:
tensor, expert and pipeline parallel. Small 3.2 is dropped, not kept as a fallback.

## Why

- **Multi-GPU becomes a necessity**, which is the case where high-performance computing skills
  matter: the weights do not fit on one card.
- **The comparison gets a real question**: how to split a MoE (by layer, by expert, by stage) is
  what serving large MoE models involves, rather than "one GPU against two".
- **A40 rather than A6000**: the A6000 offer found had 69 GB of disk, less than the weights. The
  A40 offers found on 2026-10-27 had 457 to 628 GB of disk and PCIe 4.0 for about 1.02 to 1.07 $/h.

## Consequences

- **Untested path.** Ampere has no FP4 tensor cores. vLLM 0.32.0 falls back to Marlin W4A16
  kernels for NVFP4 dense layers and MoE experts from SM 7.5 upwards (A40: SM 8.6), but no run of
  this model on Ampere was found. The 15-minute test session decides; if it fails, the fallback
  is Blackwell GPUs with native FP4 and a reduced matrix, decided by the owner.
- **Tight KV cache.** With about 35 GB of weights per GPU, only a few GB per card remain for the
  cache, if MLA keeps a full copy of the cache on each GPU (to verify). This limits concurrency
  and is itself a result to measure.
- **No independent units on 2 GPUs.** Checking the one-unit-to-many extrapolation needs a 4-GPU
  node for about an hour, which also hosts the FP8 checkpoint for the quality comparison.
