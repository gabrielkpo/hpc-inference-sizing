# Renting the GPU node (Vast.ai)

`hpc-sizing vast` finds an offer that matches the protocol (§2), rents it only on an explicit
`--yes`, watches it and destroys it. It is the only code in this repository that spends money,
so its rules are enforced in code ([`vast.py`](../src/hpc_inference_sizing/vast.py)) and tested
against a stand-in for the API ([`test_vast.py`](../tests/test_vast.py)); nothing was rented to
write it.

## Rules

| Rule (protocol §10) | How |
|---|---|
| Keys and card are entered by the owner | The API key is read from `VAST_API_KEY` only; it is never printed, logged or written. Credit is prepaid, without automatic top-up, set by the owner on the website |
| No rental by accident | `vast rent` prints the plan; it rents only with `--yes`, and only if the live price of the offer is within `--max-price` |
| Every rental has a time limit and is destroyed | `vast rent --yes` hands over to the **guard**, which destroys the instance at `--max-hours`, at `--budget-usd`, on Ctrl-C, or when the instance stays dead (exited, offline) for 10 minutes |
| ... even if the laptop sleeps | A **backstop** in the instance start script destroys the instance from inside, 10 minutes after the deadline, with the per-instance key Vast.ai sets in every container (`CONTAINER_API_KEY`, `CONTAINER_ID`); the owner's key never goes to the instance |
| Destroy, never stop | A stopped instance keeps billing its disk; the code only calls destroy |
| A bug in the guard does not leave the instance running | Any unexpected error in the guard loop destroys the instance before the guard exits |
| A failed destroy is loud | Five attempts, then `DESTROY FAILED ... destroy it by hand at https://cloud.vast.ai/instances/` |

What is not proven until the test session (steps 1 and 4 below):

- that the per-instance key is allowed to destroy its own instance (Vast.ai documents `vastai
  destroy instance $CONTAINER_ID` with it, not the REST call): checked first, on a cheap short
  rental (below);
- that an instance's `start_date` is its creation time: a guard restarted with `vast guard`
  counts the instance's age from it, while the guard started by `vast rent` counts from zero
  until this is checked (`vast show` on the test instance).

## Commands

The key lives in `.env` (ignored by Git, see `.env.example`); the CLI does not read that file
by itself, so commands that need the key load it with `uv run --env-file .env` (or the key is
exported in the shell).

```bash
uv run hpc-sizing vast offers --gpu A40 --num-gpus 2           # no key needed; --gpu any works
uv run --env-file .env hpc-sizing vast balance                  # key needed from here on
uv run --env-file .env hpc-sizing vast rent --offer ID --max-price 1.5 --max-hours 1 --budget-usd 2   # plan only
uv run --env-file .env hpc-sizing vast rent --offer ID --max-price 1.5 --max-hours 1 --budget-usd 2 --yes
uv run --env-file .env hpc-sizing vast show ID      # status, price, ssh command
uv run --env-file .env hpc-sizing vast list
uv run --env-file .env hpc-sizing vast destroy ID
```

On a laptop, run the guard under `caffeinate -i` (macOS) so it does not sleep during the rental.

## Test session (protocol §10, ≈ 1 h 15, ≈ 2 USD)

Owner, before (done on 2026-11-11): Vast.ai account, prepaid credit with automatic top-up off,
API key in `.env`, SSH public key on the website. Each step below is run, its result reported to
the owner, and the next step starts only on the owner's go.

`$V` below stands for `uv run --env-file .env hpc-sizing vast`.

**1. Backstop check (≈ 20 min, ≈ 0.02 USD).** Proves that an instance destroys itself if the
laptop sleeps, before any expensive rental relies on it. A small image, the cheapest node, no
guard:

```bash
uv run hpc-sizing vast offers --gpu any --num-gpus 1 --disk 10 --limit 3
$V rent --offer ID --image python:3.12-slim --disk 10 --max-price 0.15 \
  --max-hours 0.1 --budget-usd 0.1 --no-guard --yes
$V show INSTANCE   # once running: note start_date against the creation time
$V list            # every few minutes: the instance must be gone ~16 min after it started
```

The backstop counts from the start script, which runs once the image is pulled (status
`running`), not from the creation. Pass: the instance disappears by itself (`python:3.12-slim` has no curl, so this also checks the
Python path of the script). Fail: destroy it by hand (`$V destroy INSTANCE`), report, and every
later rental keeps the guard running in the foreground.

**2. Rent the 2 × A40 (guard in that terminal).**

```bash
uv run hpc-sizing vast offers --gpu A40 --num-gpus 2
caffeinate -i uv run --env-file .env hpc-sizing vast rent --offer ID --max-price 1.6 \
  --max-hours 1.5 --budget-usd 2.5 --yes
```

**3. Prepare the node (≈ 20 to 30 min, mostly the 76 GB of weights).** In a second terminal,
with the ssh command from `$V show INSTANCE`:

```bash
ssh -p PORT root@HOST 'curl -fsSL https://raw.githubusercontent.com/gabrielkpo/hpc-inference-sizing/main/scripts/node-setup.sh -o node-setup.sh && nohup bash node-setup.sh > node-setup.log 2>&1 &'
ssh -p PORT root@HOST 'tail -n 20 node-setup.log'   # again until "== ready"
```

Long steps run under `nohup` with a log, so an ssh drop (laptop sleep, network) does not kill
them halfway through paid time.

[`scripts/node-setup.sh`](../scripts/node-setup.sh) prints the GPUs, driver, PCIe generation and
vLLM version, installs git and uv, clones the repository, installs the bench, fetches PIAF and
the tokenizer, then the weights (anonymous download; the model is not gated). It can be run
again: done steps are skipped.

**4. Does the model start? Then one short ladder (≈ 15 min).**
[`campaign/test-session.toml`](../campaign/test-session.toml): unit TP2, RAG profile, steps of
60 s at 0.25, 0.5, 1 and 2 req/s, the same server command as the campaign.

```bash
ssh -p PORT root@HOST 'cd hpc-inference-sizing && nohup uv run hpc-sizing campaign run \
  --matrix campaign/test-session.toml --max-hours 0.5 --meta gpu=2xA40 --meta host=HOST_ID \
  > ../campaign.log 2>&1 &'
ssh -p PORT root@HOST 'tail -n 30 campaign.log'   # again until the run is written
scp -P PORT -r root@HOST:hpc-inference-sizing/runs/scratch/test-session runs/scratch/
```

Report: cold-start time, the KV-cache size vLLM logs (H6), TTFT and TPOT per step, errors. If the
server does not start (`tp-rag.failed.json`, server log next to it), stop: protocol §10 fallback,
owner's decision.

**5. Destroy.** Ctrl-C in the guard terminal (or the guard's deadline), then `$V list` must say
"no instance on the account".

## Prices seen on 2026-11-11 (public offers, disk of 150 GB included)

| Offer | USD/h | Note |
|---|---|---|
| 2 × A40, verified, reliability ≥ 0.98 | 1.35 to 1.54 | five matching offers, all on hosts in Belgium |
| 4 × A40, verified | 3.00 | one offer |

The protocol budget (§10) assumed 1.07 USD/h for 2 × A40. At these prices the plan comes to about
1.35 (test) + 14.2 (2-GPU campaign) + 6.0 (4-GPU, 2 h) ≈ 21.5 USD, above the 16 USD of the plan
and close to the 20 EUR cap. Whether to cut the plan (fewer repeats or levers, per §10) is decided
at the protocol freeze.
