# Test session (2026-11-11)

Protocol §10: before the campaign, one short paid session checks that the model starts on the
rented hardware, that one unit answers under load, that the results leave the machine and that
the instance is destroyed. **It checks the path; its numbers are not results.** They come from
one short ladder (60 s steps instead of 180 s, no repeat, no bisection) and are not used by the
sizing tool or by any hypothesis. They are written here because they shape the campaign plan.

The protocol was still a draft (not frozen) when this session ran.
The run file and the server logs stay out of Git (`runs/scratch/`), like the dry runs.

## Setup

| Item | Value |
|---|---|
| Node | Vast.ai, host 682584, machine 151324, Belgium, verified, reliability 99.7 % |
| GPUs | 2 × NVIDIA A40 (46,068 MiB each), driver 595.91.07, PCIe 4.0 x16, no NVLink, both on one NUMA node |
| Server | `vllm/vllm-openai:v0.32.0`, vLLM 0.32.0, default `gpu_memory_utilization` (0.92 in this version) |
| Code | commit `1df13a1`, [`campaign/test-session.toml`](../campaign/test-session.toml), [`scripts/node-setup.sh`](../scripts/node-setup.sh) |
| Price | 1.348 USD/h for the node, disk included; bandwidth billed on top |

## Steps and outcomes

| Step | Outcome |
|---|---|
| 1. Backstop check: 1-GPU node, `python:3.12-slim`, no guard, deadline 0.1 h | The instance destroyed itself about 16 min after it started (960 s, as set), through the Python fallback of the start script (no curl in that image) and the per-instance key. `start_date` is the creation time. No SSH on that image. Cost 0.01 USD |
| 2. Rent the 2 × A40 with the guard (1.5 h, 2.5 USD) | Running about 80 s after the rental; SSH works on the vLLM image; the backstop is armed inside |
| 3. Node setup | The bench installed in seconds; all weights (13 shards, 70.85 GB) downloaded in 4 min 35 s |
| 4. One unit under load, then a start check of the other three | Below |
| 5. Destroy | Ctrl-C in the guard terminal: destroyed in 3 s; `vast list` empty |

Total spend: 1.22 USD (credit 10.00 → 8.78). The guard counted 1.03 USD: the difference is very
likely bandwidth (70 GB of weights and the vLLM image), which Vast.ai bills on top of the hourly
price and the guard does not count.

## Does the model start on Ampere?

Yes. vLLM runs the NVFP4 weights through its Marlin kernels (`MarlinNvFp4LinearKernel` for the
dense layers, the `MARLIN` NVFP4 MoE backend for the experts) and logs that the A40 has no native
FP4 support, so weights are FP4 and compute is 16-bit, as the protocol assumed (§2). Attention
runs on `TRITON_MLA`.

All four 2-GPU units start and answer a request ("Paris" to a one-word question):

| Unit | Start to healthy | Weights loaded per GPU |
|---|---|---|
| TP2 | 229 s (first start, with `torch.compile`) | 33.34 GiB |
| EP2 | 175 s | 33.34 GiB |
| PP2 | 145 s | 33.51 GiB |
| DP2+EP | 176 s | 35.72 GiB |

## Memory model against vLLM

Predictions are those of [`MEMORY.md`](MEMORY.md), unchanged; "measured" is what vLLM logged at
start-up (`Model loading took`, `GPU KV cache size`). For DP2+EP vLLM logs one cache per rank;
the unit cache is the sum of the two.

| Unit | Weights per GPU, predicted / measured (GB) | Unit cache, predicted / measured (tokens) | Gap |
|---|---|---|---|
| TP2 | 35.85 / 35.80 | 237,808 / 278,688 | +17 % |
| EP2 | 35.85 / 35.80 | 237,808 / 278,688 | +17 % |
| PP2 | 35.83 / 35.98 | 477,281 / 534,608 | +12 % |
| DP2+EP | 38.19 / 38.35 | 272,632 / 266,272 (2 × 133,136) | −2 % |

Reading:

- **Weights per GPU** are within 0.5 % of the prediction for all four splits, including the
  copies DP2+EP makes.
- **PP2 holds 1.92 times the cache of TP2**, against 2.0 predicted: splitting by layers does avoid
  copying the cache.
- **The cache gaps of TP2, EP2 and PP2 have one known cause.** `MEMORY.md` assumed vLLM's default
  `gpu_memory_utilization` of 0.90; vLLM 0.32 defaults to 0.92. Two points of a 45 GiB card are
  0.90 GiB, which is the gap between the predicted and logged free memory (TP2: 5.10 GiB
  predicted, 5.98 GiB logged).
- **DP2+EP only looks right.** With the same correction, its logged free memory is about 25 %
  below the prediction: it uses about 1 GiB more per GPU than the 2 GiB overhead the model assumes (two engines,
  plus expert-parallel communication buffers). As a result the predicted order is wrong: the model
  put DP2+EP ahead of TP2 (42 against 37 concurrent RAG requests); the logged caches put it
  slightly behind (about 41 against 43).

H6 is judged on the campaign, under the frozen protocol, not on these numbers. Whether to pin
`--gpu-memory-utilization` in the server command or to update the model's input is a decision for
the protocol freeze; the prediction table in `MEMORY.md` stays as it was published.

## One short ladder (TP2, RAG profile)

60 s per step, 6,100 input and 300 output tokens per request, seed 0.

| Arrival rate | Requests | Errors | TTFT p50 / p95 (s) | TPOT p50 / p95 (ms) | Output tokens/s |
|---|---|---|---|---|---|
| 0.25 req/s | 12 | 0 | 0.85 / 1.45 | 13 / 22 | 60 |
| 0.5 req/s | 33 | 0 | 0.83 / 1.99 | 28 / 37 | 158 |
| 1 req/s | 61 | 0 | 2.16 / 3.49 | 109 / 158 | 265 |

The ladder stopped at 1 req/s, the first step to break the loosest target. Indicative capacity:
0.5 req/s at T = 2 s and T = 3 s; no tested rate met T = 1 s (p95 1.45 s already at 0.25 req/s).
At 1 req/s the time per output token broke first (158 ms against the 67 ms bound), while the
cache was far from full. With 12 to 61 requests per step, the p95 values rest on few requests.

## What changes for the campaign

- The fallback of §10 is not needed: the four units run on Ampere.
- The host above (machine 151324) is a candidate for the whole 2-GPU matrix (one host rule, §10).
- The budget needs a margin for bandwidth, which the guard does not count.
- The memory setting of the server is to be fixed at the protocol freeze (above).
