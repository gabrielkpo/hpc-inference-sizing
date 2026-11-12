# Results: the 2-GPU campaign (2 × A40)

The 2-GPU matrix of the protocol (§5) ran on 2026-11-11 and 12 on one Vast.ai node (2 × NVIDIA
A40, PCIe 4.0, vLLM 0.32.0, Mistral Small 4 NVFP4): 18 runs, no error, 11.1 h, 13.81 USD. It ran
under protocol v1.0; amendments A7 to A9 ([`PROTOCOL.md`](PROTOCOL.md#12-amendments)) state what
it did not do as written. The 4-GPU part (H4, H5) has not run.

Every number below comes from [`results/a40-x2.json`](results/a40-x2.json), a summary of the raw
run files (15-40 MB each, kept out of Git) made by
`hpc-sizing results summarize --dir results/a40-x2 --out docs/results/a40-x2.json`.

## How to read the numbers

- **Capacity** of a unit at target T: the highest arrival rate whose 3-minute step met the SLO
  (TTFT p95 ≤ T, TPOT p95 ≤ 67 ms, no error), read from every step of the run (§4, A8). Rates
  are nominal; the rate actually sent in that step is shown in brackets. Arrivals are random, so
  the two differ by up to about 20 %.
- **Resolution.** A capacity is located between its last passing and first failing step, a
  quarter of a ladder interval apart: 0.5 and 0.53 req/s are the same number at this resolution.
- **Same arrivals in every sweep.** The arrivals of a step are drawn with a seed set by the
  step's position in the sweep. Two sweeps that reach a rate by the same path replay the same
  arrival sequence: good for comparing configurations, but a pass then says "passes this
  sequence". The repeats (§6.6) draw new sequences and show how much that matters.

## What a client should take from it

1. **A first token in under 1 s is out of reach for 6,100-token RAG prompts on this hardware, at
   any number of GPUs.** Even at 0.125 req/s, almost idle, TTFT p95 is 1.25 to 1.44 s on every
   unit (p50 0.78 to 0.96 s): that is the time to read the prompt (prefill) with FP4 weights
   computed in 16 bits on A40s. Adding units adds throughput, not speed. Getting there takes
   shorter prompts (the chat profile, 500 tokens, has a TTFT p95 of 0.14 to 0.18 s), faster GPUs
   with native low-precision compute, or fewer retrieved passages.
2. **Prompt length decides capacity far more than how the model is split.** A 2-GPU unit serves
   about 2.5 req/s of chat but about 0.4 to 0.5 req/s of RAG at T = 2 s: a factor of five for
   twelve times more prompt tokens.
3. **What breaks first is generation speed, not memory.** Above every capacity at 2 and 3 s, and
   at every target on chat, the step that fails breaks on TPOT (alone or with TTFT): with more
   concurrent requests, each token comes slower. Only at 1.5 s on RAG does TTFT break first. PP2 holds 1.9 times the KV cache of TP2 and gains nothing from it; on chat it is the
   slowest unit (1.56 against 2.58 req/s). On two PCIe GPUs, cutting every layer in two (TP2) or
   placing experts whole (EP2) serves best.
4. **Keep prefix caching on, and let a whole prompt fit one prefill chunk.** Every RAG request
   starts with the same 1,000-token system prompt: turning prefix caching off makes 1.5 s
   unreachable and lowers the 2 s capacity (0.375 against 0.5 req/s). Raising
   `max_num_batched_tokens` from the default (2,048 here) to 8,192, so that a 6,100-token prompt
   is read in one chunk, is the only lever that moved: 0.75 req/s at 2 s on one run each. That
   gain is one step and was not repeated (H3 below). `max_num_seqs` changed nothing.
5. **On rented GPUs, noise sets the resolution of a sizing study.** The same configuration at the
   same rate passes or fails from one arrival sequence to the next; differences of one ladder step
   between units are not real until repeated. A sizing should carry that range, not one number.

## Capacity of each unit (req/s, rate actually sent in brackets)

RAG profile (6,100 input and 300 output tokens):

| Unit | T = 1 s | T = 1.5 s | T = 2 s | T = 3 s | First failure above the 3 s capacity | KV cache (tokens) | Cold start (s) |
|---|---|---|---|---|---|---|---|
| TP2 | unreachable | 0.375 (0.33) | 0.531 (0.57) | 0.531 (0.57) | TPOT at 0.5625 | 237,280 | 186 |
| EP2 | unreachable | 0.375 (0.33) | 0.5 (0.47) | 0.75 (0.69) | TTFT and TPOT at 0.8125 | 237,280 | 168 |
| PP2 | unreachable | 0.375 (0.33) | 0.5 (0.47) | 0.5 (0.47) | TPOT at 0.531 | 451,808 | 141 |
| DP2+EP | unreachable | 0.4375 (0.38) | 0.4375 (0.38) | 0.5 (0.47) | TPOT at 0.625 | 183,456 | 176 |

Chat profile (500 input and 300 output tokens). TTFT never binds, so all four targets give the
same capacity, set by TPOT:

| Unit | Every target | First failure above it | KV cache (tokens) | Cold start (s) |
|---|---|---|---|---|
| TP2 | 2.58 (2.49) | TPOT at 2.586 | 237,280 | 151 |
| EP2 | 2.45 (2.36) | TPOT at 2.453 | 237,280 | 133 |
| DP2+EP | 2.17 (2.06) | TPOT at 2.180 | 183,456 | 147 |
| PP2 | 1.56 (1.39) | TPOT at 1.566 | 451,440 | 127 |

## Repeats (§6.6): how much a pass is worth

Each capacity point of the best unit of each profile was measured three more times, with new
arrival sequences. Passes out of three:

| Repeated point | 2 s and 3 s targets | Steps (TTFT p95 / TPOT p95) |
|---|---|---|
| EP2, RAG, 0.375 req/s | 3/3 | 1.50 / 34, 1.59 / 44, 1.91 / 38 ms (all fail 1.5 s) |
| EP2, RAG, 0.5 req/s | 1/3 | 1.68 / 47, 2.51 / 76, 2.41 / 86 ms |
| EP2, RAG, 0.75 req/s | 0/3 | 2.55 / 76, 2.63 / 93, 2.03 / 75 ms |
| TP2, chat, 2.5 req/s | 3/3 | TPOT p95 58, 66, 64 ms |
| TP2, chat, 2.5625 req/s | 1/3 | TPOT p95 68, 67, 63 ms |
| TP2, chat, 2.578 req/s | 2/3 | TPOT p95 64, 65, 68 ms |

EP2's 0.75 req/s at 3 s passed in the base run, in six of the seven lever runs and in the
combination run, which reach it by the same path and so replay the same arrival sequence (124
requests, 0.69 req/s sent). With three
other sequences it fails every time. A robust reading of RAG on 2 × A40 is **0.4 to 0.5 req/s at
2 to 3 s**, the same for every unit within the noise. On chat, TP2 sits on the TPOT bound at 2.5
to 2.58 req/s (the p95 drifts between 58 and 68 ms around the 67 ms limit).

## Levers (§5.2), on EP2, RAG profile

The lever phase ran on EP2, which the faulty capacities made the best unit by tie-break; with
the recomputed values TP2 is ahead at 2 s by less than the resolution (A8).

| EP2 + lever | T = 1.5 s | T = 2 s | T = 3 s | KV cache (tokens) |
|---|---|---|---|---|
| defaults | 0.375 (0.33) | 0.5 (0.47) | 0.75 (0.69) | 237,280 |
| `max_num_seqs` 32 / 64 / 128 | 0.375 | 0.5 | 0.75 | 268,768 / 262,304 / 250,480 |
| `max_num_batched_tokens` 2,048 | 0.375 (0.33) | 0.5 (0.47) | 0.75 (0.69) | 237,280 |
| `max_num_batched_tokens` 8,192 | 0.4375 (0.54) | 0.75 (0.69) | 0.75 (0.69) | 211,984 |
| `max_num_batched_tokens` 16,384 | 0.375 (0.33) | 0.75 (0.69) | 0.75 (0.69) | 161,104 |
| prefix caching off | unreachable | 0.375 (0.33) | 0.5 (0.47) | 237,280 |
| best combination (8,192) | 0.4375 (0.54) | 0.75 (0.69) | 0.75 (0.69) | 210,896 |

`max_num_batched_tokens` 2,048 gives exactly the defaults, so 2,048 is the default here. The KV
cache moves with the levers, most likely because vLLM sizes its CUDA graphs and activation
buffers from them:
fewer sequences leave more room for the cache, a bigger prefill chunk leaves less. None of these
changes the conclusion that memory is not what limits.

## Memory model against vLLM (H6)

| Unit | Predicted (tokens) | Reported by vLLM | Gap |
|---|---|---|---|
| TP2 | 237,808 | 237,280 | -0.2 % |
| EP2 | 237,808 | 237,280 | -0.2 % |
| PP2 | 477,281 | 451,808 | -5.3 % |
| DP2+EP | 272,632 | 183,456 (2 × 91,728) | -32.7 % |

The calculator in [`MEMORY.md`](MEMORY.md) is right to 5 % where one engine holds the model, and
wrong by a third for DP2+EP, whose two engines and expert-parallel buffers use about 1 GiB more
per GPU than its fixed 2 GiB overhead allows.

## Hypotheses (§7)

| ID | Verdict | Why |
|---|---|---|
| H1 price of latency | **Not resolved** | 1 s is unreachable for every unit, so H1 is judged at 1.5 s (A3). On single runs, the best unit needs 0.75 / 0.4375 = 1.7 times the GPUs at 1.5 s than at 3 s, above the 1.5 threshold, but the 0.75 req/s at 3 s it rests on failed all three repeats; with the robust 0.4 to 0.5 req/s the ratio falls to about 1.1-1.3. What holds is cruder and stronger: the price of latency is a wall at the prefill floor (TTFT p95 about 1.3 s), not a curve between 1.5 and 3 s |
| H2 best split depends on the target | **Not resolved; its mechanism refuted** | At 1.5 s TP2 is not ahead of the others (all within one step); at 3 s no split that avoids copying the cache beats TP2. The premise that cache space limits at loose targets is wrong here: TPOT limits everywhere |
| H3 levers add ≥ 20 % | **Not resolved** | The one lever that moved (`max_num_batched_tokens` 8,192) gained one step at 2 s on a single run, inside the repeat spread; +20 % is finer than the lever ladders resolve (A9). Turning prefix caching off clearly lowers capacity |
| H4 two units scale linearly | **Not tested** | Needs the 4-GPU session |
| H5 NVFP4 costs little quality | **Not tested** | Needs the 4-GPU session (FP8 reference) |
| H6 memory model within 10 % | **Refuted** | Three units within 5.3 %, DP2+EP off by 32.7 % |

## What it means for the two clients (indicative)

The sizing tool will do this properly; here are the orders of magnitude, with the robust RAG
capacity of 0.4 to 0.5 req/s per 2-GPU unit at T = 2 s, the response time measured at capacity
(about 10 s for RAG, 17 s for chat) and a think time of 60 s (§3.2).

- **Administration** (2,000 active agents, RAG): 2,000 / (10 + 60) ≈ 29 req/s at peak, so 58 to 72
  units, **about 115 to 145 A40s** at 2 s; at 1.5 s, near the prefill floor, far more and very
  sensitive to noise; **1 s is not reachable on A40s at any fleet size**. At the 2 × A40 market
  rate seen (1.2 to 1.35 USD/h per node), 115 to 145 A40s cost about 51,000 to 71,000 USD per
  month to rent. If `max_num_batched_tokens` 8,192 holds up when repeated, the fleet shrinks by about a
  third.
- **Defence** (fixed fleet of 8 A40s, so 4 units): about 1.6 to 2 req/s of RAG at 2 s, that is
  **about 110 to 140 active users** (one request per 70 s each), or about 10 req/s of chat at a
  TTFT p95 of 0.3 s, **about 750 active users**. A fixed fleet sized on chat is overwhelmed by
  RAG use: the mix of request lengths is the first thing to agree on with the client.

## Limits

- One host, one GPU class: A40s over PCIe without NVLink. The method and the trade-offs transfer;
  the absolute numbers do not (§11).
- Poisson arrivals with fixed lengths are smoother than real traffic; real peaks would lower the
  usable capacity.
- Capacities rest on single 3-minute steps; the repeats cover the best unit of each profile only.
- No warm-up step, provenance partly in logs, lever unit chosen with the faulty values (A7, A8).
