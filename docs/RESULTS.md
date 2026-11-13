# Results: the 2-GPU campaign and the 4-GPU session (A40)

## For decision-makers

**The question.** Two organisations want to run an AI assistant on their own graphics cards
(GPUs), the chips that run these models. A public administration plans a document assistant for
20,000 agents, about 2,000 of them using it at the same time at peak hours: how many cards does
it need, and can the assistant start answering in under one second? A defence ministry already
has 8 cards: how many people can use them at once? Both clients are fictional; the measurements
are real.

**What we measured.** We rented two NVIDIA A40 cards (a 2020 generation, common and cheap to
rent), ran Mistral Small 4 on them, and sent them traffic shaped like real use: questions about
documents (the assistant first reads a few pages, about 4,500 words with its instructions, then
answers) and plain chat (short questions). We raised the traffic step by step until answers
became too slow, with hundreds to thousands of test requests per configuration, for about 24 USD
of rented time in all, test runs and failed attempts included. A last session on four cards checked two assumptions the answers rest on:
that adding machines adds capacity, and that the compressed version of the model answers as well
as the full one.

**The answers.**

- **For the administration, with answers starting within 2 seconds: about 110 to 160 cards of
  this kind,** that is about 50,000 to 70,000 USD a month at the rental prices we paid. The range
  is honest: on rented cards, the same test does not give exactly the same result twice. Copies
  of the model placed behind a dispatcher did better still in the four-card test, about 80 cards
  for the same service, but that was measured once and is not yet a figure to budget on.
- **Starting to answer in under one second is possible, but it costs about three times as much as
  two seconds on the same setup:** about 250 cards (110,000 USD a month) against 80, measured
  once on copies behind a dispatcher. Reading the documents takes about 0.8 seconds; on a single copy of the model,
  questions that arrive together wait for each other and push the slowest past one second. A
  dispatcher sends them to different copies.
- **For the defence ministry, 8 cards serve about 100 to 150 people at once asking questions on
  documents, or about 750 to 800 people chatting.** The same fleet serves five times more people if they
  chat than if they work on documents.

**What drives the bill, in order.**

1. **How much the assistant has to read for each question.** Documents make each question about
   five times more expensive than a chat message.
2. **How fast it writes once it has started.** When too many people use it at once, each answer
   is written more slowly; that, not memory, is what sets the limit.
3. **Two settings.** One kind of reuse of already-read text must stay on (turning it off costs
   about a quarter of the capacity); a second setting looks able to cut the fleet by about a
   third, but was measured once and needs confirming.

- **The compressed version of the model answers as well as the full one** on 200 questions in
  French: the cheaper version, which fits on half the cards, loses nothing measurable.

**What we recommend.**

1. Agree with the future users on an acceptable wait: within 2 seconds is realistic on this
   hardware; under 1 second roughly triples the bill.
2. Measure what share of the questions will be about documents: it moves the fleet size more
   than any technical choice.
3. Budget a range, not a single figure, and confirm it on the cards you intend to buy before
   buying them: this study shows that a few days of measurement on rented cards cost tens of
   dollars, against a fleet costing tens of thousands a month.

**How sure we are.** Solid: the effect of document length, what limits the assistant, and that
the compressed model answers as well (all well above the measurement noise). Measured once, so to
be confirmed: the gain from a dispatcher and the cost of one second. Not settled: which of the
four ways of spreading the model over the two cards is best, and the exact gain from settings
(the differences are as small as the noise).

**Six words used below.** *First-token time (TTFT)*: how long before the first word of the
answer appears. *Time per output token (TPOT)*: how long each following word takes. *p95*: the
value 95 % of requests stay under, so a guarantee that holds for almost everyone. *RAG*: a
question answered from documents the assistant reads first. *Unit*: one copy of the model
running on two cards; a fleet is a number of units. *Dispatcher*: a program that hands each new
question to the next copy of the model in turn (here nginx, round robin).

## Technical details

The 2-GPU matrix of the protocol (§5) ran on 2026-11-11 and 12 on one Vast.ai node (2 × NVIDIA
A40, PCIe 4.0, vLLM 0.32.0, Mistral Small 4 NVFP4): 18 runs, no error, 11.1 h, 13.81 USD. It ran
under protocol v1.0; amendments A7 to A9 ([`PROTOCOL.md`](PROTOCOL.md#12-amendments)) state what
it did not do as written. The 4-GPU session (H4, H5, protocol §8 and §9 with amendment A10) ran on
2026-11-12 on one 4 × A40 node: 4 runs, no error, about 2 h, 7.13 USD including the weight
downloads; two earlier attempts on other hosts cost 2.09 USD without a measurement (one host went
offline during setup, one rental was stopped before the runs).

Every number below comes from [`results/a40-x2.json`](results/a40-x2.json), a summary of the raw
run files (15-40 MB each, kept out of Git) made by
`hpc-sizing results summarize --dir results/a40-x2 --out docs/results/a40-x2.json`, and for the
4-GPU session from [`results/a40-x4.json`](results/a40-x4.json), made the same way.

### How to read the numbers

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

### Findings, technically

1. **A first token in under 1 s is out of reach for one 2-GPU unit on 6,100-token RAG prompts,
   but not for units behind a balancer.** Reading the prompt (prefill, FP4 weights computed in 16
   bits on A40s) takes about 0.78 s (TTFT p50 at 0.125 req/s). The p95 is higher, 1.25 to 1.44 s
   on every unit, because even near idle some requests arrive while another is being read and
   wait: in the first step of the 4-GPU session, the two requests above 1 s are among those that
   arrived with another in flight. Two units behind round robin send those requests to different
   units; the same arrivals then stay under 0.8 s, and the pair holds 0.5 req/s at T = 1 s (4-GPU
   session below; one run). Shorter prompts also get there (the chat profile, 500 tokens, has a
   TTFT p95 of 0.14 to 0.18 s), as would GPUs with native low-precision compute.
2. **Prompt length decides capacity far more than how the model is split.** A 2-GPU unit serves
   about 2.5 req/s of chat but about 0.375 to 0.5 req/s of RAG at T = 2 s: a factor of five for
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

### Capacity of each unit (req/s, rate actually sent in brackets)

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

### Repeats (§6.6): how much a pass is worth

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
other sequences it fails every time. A robust reading of RAG on 2 × A40 is **0.375 to 0.5 req/s at
2 to 3 s**, the same for every unit within the noise. On chat, TP2 sits on the TPOT bound at 2.5
to 2.58 req/s (the p95 drifts between 58 and 68 ms around the 67 ms limit).

### Levers (§5.2), on EP2, RAG profile

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

### 4-GPU session (§8, §9, A10): two units and answer quality

One 4 × A40 node (PCIe 4.0, two NUMA nodes, GPUs 0-1 and 2-3 on separate ones), vLLM 0.32.0. Each
run was measured once.

**H4: one TP2 unit, then two TP2 units behind nginx round robin**, RAG profile, same procedure as
the campaign (30 s warm-up, ladder, two bisection steps):

| Setup | T = 1 s | T = 1.5 s | T = 2 s | T = 3 s | KV cache (tokens) | Cold start (s) |
|---|---|---|---|---|---|---|
| 1 unit, GPUs 0-1 | unreachable | 0.375 (0.33) | 0.5 (0.47) | 0.75 (0.69) | 237,280 | 202 |
| 2 units, GPUs 0-1 and 2-3 | 0.5 (0.47) | 1.375 (1.26) | 1.39 (1.41) | 1.39 (1.41) | 2 × 237,280 | 283 |

The single unit reproduces the campaign's TP2 on another host (0.5 against 0.531 req/s at 2 s).
At T = 2 s the pair holds 2.8 times the capacity of one unit; with the resolution (the single
unit's 0.625 req/s step failed on TPOT p95 by 1 ms, 68 against 67 ms), the ratio lies between
2.2 and 2.8, above the 1.8 of H4 either way. More than double is what queueing predicts: TTFT
p95 is set by requests that arrive while another is being prefilled, and round robin both halves
the load per unit and makes each unit's arrivals more regular than a Poisson stream (every other
arrival). TTFT p50 is the same in both setups (0.78 s at 0.125 req/s); only the tail moves. The
gain should grow, not shrink, with more units behind one balancer, but only two were measured.

**H5: answer quality of NVFP4 (2 GPUs) against FP8 (4 GPUs)**, 200 PIAF questions (seed 0),
chat endpoint, temperature 0, at most 32 tokens, French normalisation (A10):

| Model | Exact match | F1 | Answers cut at 32 tokens |
|---|---|---|---|
| NVFP4, TP2 | 36.0 | 66.3 | 26 |
| FP8, TP4 | 34.0 | 65.9 | 30 |

NVFP4 is 0.4 F1 point above FP8, inside the noise of 200 questions and well within the 2 points
H5 allows. The low exact match is a style effect, the same for both: the model answers in a
sentence, often with bold ("Jim rentre dans une **église**." for the reference "église"), and 13
to 15 % of the answers are cut at 32 tokens. F1 counts the overlapping words and is the fairer of
the two scores. Every answer is kept in the run files.

### Memory model against vLLM (H6)

| Unit | Predicted (tokens) | Reported by vLLM | Gap |
|---|---|---|---|
| TP2 | 237,808 | 237,280 | -0.2 % |
| EP2 | 237,808 | 237,280 | -0.2 % |
| PP2 | 477,281 | 451,808 | -5.3 % |
| DP2+EP | 272,632 | 183,456 (2 × 91,728) | -32.7 % |

The calculator in [`MEMORY.md`](MEMORY.md) is right to 5 % where one engine holds the model, and
wrong by a third for DP2+EP, whose two engines and expert-parallel buffers use about 1 GiB more
per GPU than its fixed 2 GiB overhead allows.

### Hypotheses (§7)

| ID | Verdict | Why |
|---|---|---|
| H1 price of latency | **Not resolved on 2-GPU units; supported behind a balancer (one run)** | 1 s is unreachable for every 2-GPU unit, so H1 is judged at 1.5 s (A3). On single runs, the best unit needs 0.75 / 0.4375 = 1.7 times the GPUs at 1.5 s than at 3 s, above the 1.5 threshold, but the 0.75 req/s at 3 s it rests on failed all three repeats; with the robust 0.375 to 0.5 req/s the ratio falls to about 1.1-1.3. Two units behind round robin reach 1 s, and there the administration needs 248 GPUs at 1 s against 80 at 3 s, a ratio of 3.1, on one run of each |
| H2 best split depends on the target | **Not resolved; its mechanism refuted** | At 1.5 s TP2 is not ahead of the others (all within one step); at 3 s no split that avoids copying the cache beats TP2. The premise that cache space limits at loose targets is wrong here: TPOT limits everywhere |
| H3 levers add ≥ 20 % | **Not resolved** | The one lever that moved (`max_num_batched_tokens` 8,192) gained one step at 2 s on a single run, inside the repeat spread; +20 % is finer than the lever ladders resolve (A9). Turning prefix caching off clearly lowers capacity |
| H4 two units scale linearly | **Confirmed, beyond linear (one run)** | Two units behind round robin hold 2.2 to 2.8 times one unit at T = 2 s (criterion ≥ 1.8); queueing explains the excess. Extrapolating from one unit is therefore prudent, not optimistic |
| H5 NVFP4 costs little quality | **Confirmed** | PIAF F1 66.3 for NVFP4 against 65.9 for FP8 (criterion: NVFP4 ≥ FP8 − 2) |
| H6 memory model within 10 % | **Refuted** | Three units within 5.3 %, DP2+EP off by 32.7 % |

### What it means for the two clients (sizing tool)

`hpc-sizing size` turns the capacities into the clients' answers (§3.2, §8): users send
N / (R + Z) requests per second, with R the response time measured at that capacity and Z = 60 s
of think time; units = ⌈load / capacity⌉, two GPUs each. It gives two bounds: optimistic, the
best single run of any unit; prudent, the highest rate a unit met in every repeat. Rental at
0.61 USD per GPU-hour (2 × A40 at 1.216 USD/h), 730 h a month.

| Question | Target | Optimistic | Prudent |
|---|---|---|---|
| Administration, 2,000 active agents, RAG: GPUs | 1 s | unreachable | unreachable |
| | 1.5 s | 136 GPUs (DP2+EP 0.44 req/s) | unknown: no repeated rate met 1.5 s |
| | 2 s | 108 GPUs, 48,000 USD/month (TP2 0.53 req/s) | 160 GPUs, 71,000 USD/month (0.375 req/s) |
| | 3 s | 72 GPUs (EP2 0.75 req/s, which failed 3 of 3 repeats) | 160 GPUs (0.375 req/s) |
| Defence, 8 GPUs: active users | RAG, 2 s | about 148 | about 100 |
| | chat, 2 s | about 798 | about 762 |

The tool extrapolates from one unit, with no gain from balancing: H4 shows that this is prudent.
Sized on the measured pair instead (4 GPUs per block, same formula, one run, R at that capacity):

| Question | Target | One unit, this node | Pairs behind round robin |
|---|---|---|---|
| Administration, 2,000 active agents, RAG: GPUs | 1 s | unreachable | 248 GPUs, 110,000 USD/month (0.5 req/s per pair) |
| | 2 s | 120 GPUs, 53,000 USD/month (0.5 req/s) | 80 GPUs, 36,000 USD/month (1.39 req/s per pair) |
| Defence, 8 GPUs, RAG, 2 s: active users | | about 135 | about 206 (two pairs) |

These are single runs, with two units behind the balancer; a fleet would put more units behind
one balancer, which queueing says helps further but nobody measured here. They show the direction
and the size of the effect, not a figure to budget on before repeats.

Reading: the prudent RAG capacity is the same at 2 s and 3 s, because 0.5 req/s passed one repeat
in three at both targets; the optimistic gap between 2 s and 3 s rests on a step that did not
reproduce. Between 1.5 and 3 s the fleet changes within the noise; 1 s needs
units behind a balancer and about three times the GPUs of 2 s. If `max_num_batched_tokens` 8,192 holds up when repeated, the fleet
shrinks by about a third. A fixed fleet sized on chat is overwhelmed by RAG use: the mix of
request lengths is the first thing to agree on with a client.

### Limits

- One host, one GPU class: A40s over PCIe without NVLink. The method and the trade-offs transfer;
  the absolute numbers do not (§11).
- Poisson arrivals with fixed lengths are smoother than real traffic; real peaks would lower the
  usable capacity.
- Capacities rest on single 3-minute steps; the repeats cover the best unit of each profile only.
  The 4-GPU runs were not repeated.
- Answer quality is one dataset (PIAF, extractive questions in French), 200 questions, short
  answers; it compares the two formats, not the model with others.
- 2-GPU campaign: no warm-up step, provenance partly in logs, lever unit chosen with the faulty
  values (A7, A8). The 4-GPU session has both warm-up and provenance in every run file.
- 4-GPU session: the FP8 repository holds its weights twice (two formats, 121 GB each); the node
  setup fetched both and filled the disk, so the FP8 weights were fetched again by hand in the
  format vLLM loads, before any measurement. The campaign caps were lowered to fit the guard.
