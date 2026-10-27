# Measurement protocol

- **Status: draft.** Values marked *(proposed)* are defaults to confirm before the freeze.
- **Freeze rule:** this file is frozen by a pull request titled `docs: freeze protocol`, merged
  before the first paid GPU run. After the freeze, any change goes into
  [Amendments](#12-amendments) with its date and reason. Results produced under an earlier
  version keep the protocol version they ran under.

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
| Server | vLLM, one pinned version (0.32.0 at the time of writing), recorded in every result |
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
  `min_tokens` = 300, end-of-sequence ignored) so every request does the same work. *(to verify
  that the pinned vLLM version honours both for this model)*
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
| Think time Z | 60 s *(proposed)* |
| Administration: agents / active at peak | 20,000 / 10 %, so N = 2,000 active users *(proposed)* |
| Defence: fleet size | 8 GPUs of the measured class *(proposed)* |

The bench measures capacity in requests per second. The sizing tool converts it into users with
this formula and states Z next to every user count.

## 4. Service-level objective (SLO)

A load step **meets the SLO** when all of these hold over its measured window:

| Metric | Bound |
|---|---|
| Time to first token (TTFT), p95 | ≤ T, with T ∈ {1 s, 2 s, 3 s} |
| Time per output token (TPOT), p95 | ≤ 67 ms (at least 15 tokens per second per user) |
| Errors and timeouts | 0 |

The **capacity** C(unit, profile, T) is the highest arrival rate whose step meets the SLO.

## 5. Configurations

### 5.1 Serving units on 2 GPUs

The weights (70.8 GB) do not fit on one 48 GB card, so every unit spans both GPUs. The three
ways of splitting a MoE across them are the main comparison:

| Unit | vLLM flags (indicative) | What is split |
|---|---|---|
| **TP2** | `--tensor-parallel-size 2` | every layer, attention and experts, across both GPUs |
| **EP2** | `--tensor-parallel-size 2 --enable-expert-parallel` | experts placed whole on one GPU or the other |
| **PP2** | `--pipeline-parallel-size 2` | the first half of the layers on one GPU, the second half on the other |

All units use `--attention-backend TRITON_MLA` and the Mistral formats (`--tokenizer-mode
mistral --config-format mistral --load-format mistral`). A unit that does not start on the test
session (§10) is recorded as **not supported on this hardware**: that is a result, and it is
not debugged on paid time.

### 5.2 Serving parameters (levers)

Tested on the best unit only, RAG profile, T = 2 s, one lever at a time from vLLM's defaults.
A lever ladder starts two steps below that unit's baseline capacity, not at the bottom:

| Lever | Values |
|---|---|
| `max_num_seqs` | 32, 64, 128 *(proposed)* |
| `max_num_batched_tokens` (chunked prefill size) | 2,048, 8,192, 16,384 *(proposed)* |
| Prefix caching | on, off |

The best combination is then measured once at T = 1 s and T = 3 s.

## 6. Procedure

1. **Start** the unit, wait for `/health`, record the cold-start time and the KV-cache capacity
   vLLM reports.
2. **Warm up** for 30 s at a low rate; the warm-up is not measured.
3. **Load ladder**: steps of 3 min at λ = 0.25, 0.5, 1, 2, 4, … req/s *(proposed)*, stopped at
   the first step that breaks the **loosest** target (T = 3 s). One ladder serves all three
   targets: the capacity for each T is read from the same steps.
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
| **H1** | The price of latency is non-linear: a stricter TTFT target costs disproportionately more GPUs | GPUs(T = 1 s) / GPUs(T = 3 s) ≥ 1.5 for the administration scenario, RAG profile *(proposed)* |
| **H2** | The best split depends on the target: TP2 wins at a strict TTFT, EP2 at a loose one | C(TP2) > C(EP2) at T = 1 s **and** C(EP2) > C(TP2) at T = 3 s, beyond the repeat range |
| **H3** | Tuning the levers raises capacity over vLLM defaults | ≥ +20 % capacity, best unit, RAG, T = 2 s *(proposed)* |
| **H4** | Capacity scales linearly with the number of units, so extrapolating from one unit is valid | 2 units behind round-robin reach ≥ 1.8 × the capacity of 1 unit (§8) |
| **H5** | NVFP4 costs little quality against FP8 | PIAF F1 of NVFP4 ≥ F1 of FP8 − 2 points (§9) *(proposed)* |
| **H6** | The memory calculator predicts the KV-cache capacity vLLM reports | within 10 % for every unit |

## 8. Extrapolation and its check (4 GPUs)

The sizing tool assumes independent units and perfect load spreading:

> units = ⌈ λ_peak / C(unit, profile, T) ⌉, GPUs = units × GPUs per unit, cost = GPUs × price
> per GPU-hour × 730 h per month

H4 tests this assumption on the only setup within budget: a 4-GPU node running **two TP2 units**
behind a round-robin balancer (nginx), against one unit, same workload, same procedure. The same
node then serves the **FP8** checkpoint (about 119 GB) on all 4 GPUs for §9 and for an FP8
capacity point.

Not tested, and written as a limit: units on separate machines, smarter routing, autoscaling.

## 9. Quality check

- Dataset: **PIAF** (French question answering built by Etalab, `AgentPublic/piaf` on Hugging
  Face, MIT licence, 3,835 questions), 200 questions drawn with a fixed seed *(proposed)*.
- Prompt: the context passage and the question, answer in a few words; temperature 0.
- Score: exact match and token-level F1 against the reference answers, same normalisation for
  both formats.
- Compared: NVFP4 on 2 GPUs against FP8 on 4 GPUs.

## 10. Runs, budget and stop rules

| Run | GPUs | Content | Estimated cost |
|---|---|---|---|
| Test session | 2 | the model loads, one unit answers, one short ladder, results leave the machine, the instance is destroyed | ≈ 1 $ |
| Main campaign | 2 | §5.1 on both profiles (6 ladders, ≈ 4.5 h), §5.2 (≈ 7 short ladders, ≈ 3 h), repeats (≈ 1.5 h) | ≈ 9 h, ≈ 10 $ |
| Scaling and quality | 4 | §8 and §9 | ≈ 2 h, ≈ 4 $ |

- **No rental** until the whole chain (SLURM jobs, load generator, client, sweep, save, resume)
  has run end to end locally.
- **Hard budget: 20 €** in total; the plan above comes to ≈ 15 $. The campaign script prints the
  amount spent after each block; if the projection passes the cap, repeats are dropped first,
  then lever values, never the base ladders.
- Prepaid credit only, no automatic top-up. Keys and card are entered by the owner, never by a
  script or an agent.
- Every campaign script has a **time limit** and **destroys** the instance when it ends or fails
  (destroy, not stop: a stopped instance keeps billing its disk).
- **One host for the whole 2-GPU matrix**: numbers from different hosts are not compared.
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

None yet.
