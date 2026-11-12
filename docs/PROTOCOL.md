# Measurement protocol

- **Status: frozen, version 1.0, 2026-11-11**, by the pull request `docs: freeze protocol`;
  amendments A7 to A9 (2026-11-12) record what the 2-GPU campaign did not do as written.
- **Freeze rule:** this file is frozen before the first run of the campaign (§10). The test
  session ran on 2026-11-11 under the draft, as §10 intends: it checks the path, and its numbers
  are not results ([`TEST-SESSION.md`](TEST-SESSION.md)). The draft said "before the first paid
  GPU run"; that wording was changed at the freeze (amendment A1). After the freeze, any change
  goes into [Amendments](#12-amendments) with its date and reason. Results produced under an
  earlier version keep the protocol version they ran under.

This protocol fixes, before any measurement, the question, the workload, the configurations,
what counts as "the target is met", the hypotheses and their pass criteria, and the stop rules.
Its purpose is that the results cannot be bent to fit the conclusion.

## 1. Question

A client hosts an open-weight model on its own GPUs and asks one of two questions:

| Client (fictional) | Inspired by | Question |
|---|---|---|
| **Administration**: document assistant for its agents, hosted in-house | public cases of national agencies equipping their staff with a GenAI assistant | *How many GPUs, at minimum, to serve the peak load with a given time-to-first-token target, and what does a stricter target cost?* |
| **Defence**: same kind of assistant on a fixed, isolated GPU fleet | public framework agreements requiring models to run on the ministry's own infrastructure | *With N GPUs, what latency do users get, and how many concurrent users before the target breaks?* |

Both are answered from the same measurement: the **capacity of one serving unit under a latency
target**. A serving unit is one vLLM deployment of the model on a fixed number of GPUs.

The clients are fictional. Their sizes are set in §3 and are inputs to the sizing tool, not
measured quantities.

## 2. System under test

| Item | Value |
|---|---|
| Model | `mistralai/Mistral-Small-4-119B-2603-NVFP4`: MoE, 128 experts, 4 active plus 1 shared, 119B parameters, 6.5B active per token, MLA attention, 70.8 GB of weights |
| Reasoning | Off (`reasoning_effort="none"`): the assistant answers directly |
| Server | vLLM 0.32.0 (`vllm/vllm-openai:v0.32.0`), recorded in every result, with `--gpu-memory-utilization 0.90` (§5.1) |
| Hardware | One rented node with **2 × NVIDIA A40 (48 GB, Ampere, SM 8.6)**, PCIe 4.0; 4 × A40 for §8 |
| Weight format on Ampere | NVFP4 weights run through vLLM's Marlin W4A16 fallback kernels (no FP4 tensor cores on Ampere) |
| Provider | Vast.ai, verified host, reliability ≥ 98 %, disk ≥ 120 GB, download ≥ 500 Mbit/s |

Every result records: git commit, protocol version, vLLM version and image digest, GPU model and
count, driver and CUDA versions, PCIe generation, host identifier, vLLM flags, workload profile,
seed, start and end timestamps.

## 3. Workload

### 3.1 Request profiles

| Profile | Input tokens | Output tokens | Shared prefix |
|---|---|---|---|
| **RAG** (main) | 6,100: system prompt 1,000 + 5 passages of 1,000 + question 100 | 300 | the 1,000-token system prompt, identical for every request |
| **Chat** (secondary) | 500 | 300 | none |

- Lengths are exact, counted with the model's tokenizer. Output length is forced (`max_tokens` =
  `min_tokens` = 300, end-of-sequence ignored) so every request does the same work. Checked on
  the test session: 106 requests out of 106 returned exactly 300 tokens.
- Passages and questions come from PIAF (§9), so the text is real French prose and not random
  tokens. PIAF contexts are Wikipedia paragraphs of a few hundred tokens, so a passage is built by
  concatenating consecutive contexts of the same article until it passes 1,000 tokens, then
  cutting it to exactly 1,000 tokens. Each request draws different passages: only the system
  prompt is shared.
- Seeds are fixed per run and recorded.

### 3.2 Arrivals and users

Requests arrive as a **Poisson process** at rate λ (requests per second).

A user does not send requests continuously. A user asks a question, waits for the answer (R) and
reads it (think time Z). By Little's law, N active users send

> λ = N / (R + Z)

| Parameter | Value |
|---|---|
| Think time Z | 60 s |
| Administration: agents / active at peak | 20,000 / 10 %, so N = 2,000 active users |
| Defence: fleet size | 8 GPUs of the measured class |

The bench measures capacity in requests per second. The sizing tool converts it into users with
this formula and states Z next to every user count.

## 4. Service-level objective (SLO)

A load step **meets the SLO** when all of these hold over its measured window:

| Metric | Bound |
|---|---|
| Time to first token (TTFT), p95 | ≤ T, with T ∈ {1 s, 1.5 s, 2 s, 3 s} |
| Time per output token (TPOT), p95 | ≤ 67 ms (at least 15 tokens per second per user) |
| Errors and timeouts | 0 |

The **capacity** C(unit, profile, T) is the highest arrival rate whose step meets the SLO.

A target that no step meets, down to the lowest rate of the ladder, is **unreachable** for that
unit and profile: C = 0, and the result says "unreachable on this hardware with this profile".
Adding units does not make it reachable, since a unit's latency at low load does not depend on
the number of units. The sizing tool (§8) then reports "unreachable" instead of a GPU count. The
1.5 s target was added at the freeze (amendment A3).

## 5. Configurations

### 5.1 Serving units on 2 GPUs

The weights (70.8 GB) do not fit on one 48 GB card, so every unit spans both GPUs. Four ways of
splitting a MoE across them are the main comparison (calculator names in brackets):

| Unit | vLLM flags (indicative) | What is split |
|---|---|---|
| **TP2** (`tp`) | `--tensor-parallel-size 2` | every layer, attention and experts, across both GPUs |
| **EP2** (`ep`) | `--tensor-parallel-size 2 --enable-expert-parallel` | experts placed whole on one GPU or the other |
| **PP2** (`pp`) | `--pipeline-parallel-size 2` | the first half of the layers on one GPU, the second half on the other |
| **DP2+EP** (`dp_ep`) | `--data-parallel-size 2 --enable-expert-parallel` | attention, shared expert, embedding and output head copied on both GPUs, each GPU serving its own requests; routed experts placed whole on one GPU or the other |

With MLA attention the KV cache is copied on both GPUs under TP2 and EP2, split by layers under
PP2 and split by requests under DP2+EP. The predicted capacity of each unit is in
[`MEMORY.md`](MEMORY.md) (2 x A40, RAG profile: TP2 and EP2 37 concurrent requests, PP2 74,
DP2+EP 42).

All units use `--attention-backend TRITON_MLA`, the Mistral formats (`--tokenizer-mode
mistral --config-format mistral --load-format mistral`) and `--gpu-memory-utilization 0.90`, the
value the memory model assumes, so that H6 tests the predictions as published (vLLM 0.32 defaults
to 0.92; amendment A2). A unit that does not start on the test
session (§10) is recorded as **not supported on this hardware**: that is a result, and it is
not debugged on paid time.

### 5.2 Serving parameters (levers)

Tested on the best unit only, RAG profile, T = 2 s, one lever at a time from vLLM's defaults.
A lever ladder starts two steps below that unit's baseline capacity, not at the bottom:

| Lever | Values |
|---|---|
| `max_num_seqs` | 32, 64, 128 |
| `max_num_batched_tokens` (chunked prefill size) | 2,048, 8,192, 16,384 |
| Prefix caching | on, off |

The best combination is then measured once at T = 1 s and T = 3 s.

## 6. Procedure

1. **Start** the unit, wait for `/health`, record the cold-start time and the KV-cache capacity
   vLLM reports.
2. **Warm up** for 30 s at a low rate; the warm-up is not measured.
3. **Load ladder**: steps of 3 min at λ = 0.125, 0.25, 0.5, 1, 2, 4, … req/s, stopped at
   the first step that breaks the **loosest** target (T = 3 s). One ladder serves all four
   targets: the capacity for each T is read from the same steps. The ladder starts at 0.125 req/s
   so that a strict target is measured near idle (amendment A3).
4. **Refine** each target's capacity with two bisection steps between its last passing and first
   failing rate.
5. **Record** every request (send time, first-token time, token times, lengths, error) and the
   vLLM `/metrics` scraped every 5 s (queue length, KV-cache use, running and waiting requests).
6. **Repeat** the final capacity point 3 times on the best unit of each profile; report the
   median and the range.
7. **Save** the run's JSON off the machine as soon as it ends; the campaign resumes from the
   last saved run.

Each step is one SLURM job; the same `sbatch` scripts run on the local Docker cluster against a
small model (dry run) and on the rented node against the real one.

## 7. Hypotheses and pass criteria

Each hypothesis is confirmed or refuted by the numbers, and both outcomes are published.

| ID | Hypothesis | Pass criterion |
|---|---|---|
| **H1** | The price of latency is non-linear: a stricter TTFT target costs disproportionately more GPUs | GPUs(T = 1 s) / GPUs(T = 3 s) ≥ 1.5 for the administration scenario, RAG profile. If T = 1 s is unreachable for every unit (§4), that is reported, and H1 is judged on T = 1.5 s instead |
| **H2** | The best split depends on the target: under a strict TTFT target per-request speed limits first and TP2 wins; under a loose target cache space limits first and a split that does not copy the cache (PP2 or DP2+EP) wins | C(TP2) is the highest of the four units at T = 1 s **and** C(PP2) or C(DP2+EP) exceeds C(TP2) at T = 3 s, beyond the repeat range. If T = 1 s is unreachable for every unit, the first half is judged on T = 1.5 s |
| **H3** | Tuning the levers raises capacity over vLLM defaults | ≥ +20 % capacity, best unit, RAG, T = 2 s |
| **H4** | Capacity scales linearly with the number of units, so extrapolating from one unit is valid | 2 units behind round-robin reach ≥ 1.8 × the capacity of 1 unit (§8) |
| **H5** | NVFP4 costs little quality against FP8 | PIAF F1 of NVFP4 ≥ F1 of FP8 − 2 points (§9) |
| **H6** | The memory calculator predicts the KV-cache capacity vLLM reports | within 10 % of the predictions in [`MEMORY.md`](MEMORY.md) for every unit, vLLM's log read per split as described there |

## 8. Extrapolation and its check (4 GPUs)

The sizing tool assumes independent units and perfect load spreading:

> units = ⌈ λ_peak / C(unit, profile, T) ⌉ (C = 0: unreachable, §4), GPUs = units × GPUs per unit, cost = GPUs × price
> per GPU-hour × 730 h per month

H4 tests this assumption on the only setup within budget: a 4-GPU node running **two TP2 units**
behind a round-robin balancer (nginx), against one unit, same workload, same procedure. The same
node then serves the **FP8** checkpoint (about 119 GB) on all 4 GPUs for §9 and for an FP8
capacity point.

Not tested, and written as a limit: units on separate machines, smarter routing, autoscaling.

## 9. Quality check

- Dataset: **PIAF** (French question answering built by Etalab, `AgentPublic/piaf` on Hugging
  Face, MIT licence, 3,835 questions), 200 questions drawn with a fixed seed.
- Prompt: the context passage and the question, answer in a few words; temperature 0.
- Score: exact match and token-level F1 against the reference answers, same normalisation for
  both formats.
- Compared: NVFP4 on 2 GPUs against FP8 on 4 GPUs.

## 10. Runs, budget and stop rules

| Run | GPUs | Content | Estimated cost |
|---|---|---|---|
| Test session | 2 | the model loads, one unit answers, one short ladder, results leave the machine, the instance is destroyed | done on 2026-11-11: 1.22 $ |
| Main campaign | 2 | §5.1 on both profiles (8 ladders, ≈ 6 h), §5.2 (≈ 7 short ladders, ≈ 3 h), repeats (≈ 1.5 h) | ≈ 10.5 h at 1.35 $/h, ≈ 14.2 $ |
| Scaling and quality | 4 | §8 and §9; disk ≥ 200 GB for both checkpoints (NVFP4 71 GB and FP8 121 GB) | ≈ 2 h at 3.00 $/h, ≈ 6 $ |
| Bandwidth | | billed on top of the hourly price: weights (NVFP4 71 GB, FP8 121 GB), image | ≈ 0.8 $ |

Prices are those seen on 2026-11-11 ([`VAST.md`](VAST.md)); the draft assumed 1.07 $/h. The
estimate of 10.5 h predates the test session and is kept as the planning figure; the extra low
step of each ladder (A3) adds about 50 min, about 1.1 $, for a total near 23 $.

- **No rental** until the whole chain (SLURM jobs, load generator, client, sweep, save, resume)
  has run end to end locally.
- **Hard budget: 20 €** in total; the plan above comes to ≈ 23 $, so the cap acts as a stop
  rule (amendment A4). The campaign script prints the amount spent after each block; if the
  projection passes the cap, repeats are dropped first, then lever values, never the base
  ladders. The guard does not count bandwidth: its `--budget-usd` keeps a margin for it.
- Prepaid credit, added by the owner. Vast.ai has no switch to turn off automatic billing: with a
  card on file it charges the card only for a negative balance, which the time and budget caps
  of every rental prevent (owner's choice to keep the card, 2026-11-11). Keys and card are
  entered by the owner, never by a script or an agent.
- Every campaign script has a **time limit** and **destroys** the instance when it ends or fails
  (destroy, not stop: a stopped instance keeps billing its disk).
- **One host for the whole 2-GPU matrix**: numbers from different hosts are not compared. The
  matrix runs in one session (amendment A5), preferably on the test session's machine (Vast.ai
  machine 151324), with the results copied off the node after every run.
- **Fallback.** If the model does not start on Ampere during the test session, the run stops, the
  instance is destroyed and the next step (Blackwell GPUs with native FP4 and a reduced matrix)
  waits for the owner's decision. No automatic spending on a fallback.

## 11. Threats to validity

- **One host, one GPU class.** A40 over PCIe is not an H100 or B200 with NVLink. The method and
  the trade-offs transfer; the absolute numbers do not.
- **Synthetic traffic.** Poisson arrivals with fixed lengths are smoother than real peaks;
  real traffic would lower the usable capacity.
- **Rented hardware.** Neighbours, thermal state and host network add noise; repeats measure it,
  they do not remove it.
- **Weight-only FP4 on Ampere.** The Marlin path is slower than native FP4; Blackwell numbers
  would differ.

## 12. Amendments

Changes made at the freeze (2026-11-11), against the draft. Those marked *after the test
session* were motivated by its numbers ([`TEST-SESSION.md`](TEST-SESSION.md)); they were decided
before any campaign run.

| ID | Change | Reason |
|---|---|---|
| A1 | Freeze rule: "before the first run of the campaign" instead of "before the first paid GPU run" | The test session is paid and ran under the draft, as §10 intends |
| A2 | `--gpu-memory-utilization 0.90` pinned in every server command (§5.1) | *After the test session.* vLLM 0.32 defaults to 0.92, while the memory model assumes 0.90; pinning the model's value keeps H6 a test of the predictions as published, instead of recomputing them after seeing the logs |
| A3 | Target T = 1.5 s added (§4); unreachable targets defined (§4); H1 and the first half of H2 fall back to 1.5 s if 1 s is unreachable for every unit; the ladder starts at 0.125 req/s (§6) | *After the test session.* TP2 already had a TTFT p95 of 1.45 s at 0.25 req/s (12 requests), so 1 s may be unreachable. The 1 s target is kept; the extra target costs no run time, the extra low step about 3 min per ladder (about 50 min, 1.1 $ over the campaign). An unreachable target gives "unreachable" in the sizing tool, not a GPU count. The draft's 0.1 req/s start became 0.125 to keep the ladder's doubling |
| A4 | Budget table at the prices seen on 2026-11-11, bandwidth added; the 20 € cap acts as a stop rule; the 4-GPU node needs ≥ 200 GB of disk | *After the test session.* 2 × A40 cost 1.35 $/h, not 1.07; bandwidth is billed on top. The owner adds credit; the cut order of §10 is unchanged |
| A5 | The 2-GPU matrix runs in one session on one host | Enforces the one-host rule; splitting risks losing the host between sessions |
| A6 | Values marked *(proposed)* in the draft are kept unchanged (§3.2, §5.2, §6 apart from A3, §7 apart from A3, §9); the output-length check of §3.1 is marked done | The test session gave no reason to change them |

### After the 2-GPU campaign (2026-11-12)

The 2-GPU matrix ran on 2026-11-11 and 12 under version 1.0. An audit of the repository and the
run files then found the deviations below. None of them is fixed by rerunning: each is recorded
here, and the analysis says where it matters. The measurements are not changed; how they are read
is stated before any result is published.

| ID | Deviation | What it changes |
|---|---|---|
| A7 | Not implemented as written: the 30 s warm-up (§6.2; the first measured step, 0.125 req/s, came right after the server start); provenance in every result (§2: vLLM version, KV-cache size and server flags are in the server logs, driver, PCIe generation and vLLM version in the node setup log, the image by tag; the image digest and CUDA version were not recorded); per-token times (§6.5: first and last token only, so TPOT is a per-request mean); SLURM jobs (§6: the matrix ran as one `campaign run --chain` process, see [ADR 0003](adr/0003-campaign-runs-as-one-chained-process.md)) | No warm-up effect is visible (on TP2 × RAG the 0.125 and 0.25 req/s steps have the same TTFT p95, 1.32 and 1.33 s). Before the 4-GPU session, the bench is to run the warm-up and record provenance automatically; until then, these gaps hold for every run |
| A8 | The code fixed each target's capacity when that target's bisection ended, so steps measured later for looser targets did not count for stricter ones: 12 of the 32 base capacities in the run files are below the highest step that met them. The realized arrival rate also differs from the nominal rate by up to about 20 % (Poisson arrivals over 180 s) | Every capacity is recomputed from the steps of the run files as §4 defines it (highest rate whose step meets the SLO), and the realized rate is shown next to it. The lever phase picked its unit from the faulty values: EP2, by tie-break; with the recomputed values TP2 is the best unit at 2 s (0.53 against 0.5 req/s). The levers were measured on EP2 and are reported as such |
| A9 | Resolution and noise: two bisection steps resolve a capacity to a quarter of a ladder interval, and the repeats (§6.6) show that a step at the same nominal rate can pass or fail from one repetition to the next | Capacity differences between units, and lever gains, smaller than one ladder step or than the spread of the repeats, are reported as not resolved. This applies to H2 ("beyond the repeat range", already in its criterion) and to H3 (+20 % is finer than the lever ladders can resolve) |
