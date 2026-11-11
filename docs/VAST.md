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

What is not proven until the test session:

- that the per-instance key is allowed to destroy its own instance (Vast.ai documents `vastai
  destroy instance $CONTAINER_ID` with it, not the REST call): checked first, on a cheap short
  rental (below);
- that an instance's `start_date` is its creation time: a guard restarted with `vast guard`
  counts the instance's age from it, while the guard started by `vast rent` counts from zero
  until this is checked (`vast show` on the test instance).

## Commands

```bash
uv run hpc-sizing vast offers --gpu A40 --num-gpus 2           # no key needed
uv run hpc-sizing vast balance                                  # key needed from here on
uv run hpc-sizing vast rent --offer ID --max-price 1.5 --max-hours 1 --budget-usd 2   # plan only
uv run hpc-sizing vast rent --offer ID --max-price 1.5 --max-hours 1 --budget-usd 2 --yes
uv run hpc-sizing vast show ID      # status, price, ssh command
uv run hpc-sizing vast list
uv run hpc-sizing vast destroy ID
```

On a laptop, run the guard under `caffeinate -i` (macOS) so it does not sleep during the rental.

## Test session (protocol §10, ≈ 1 h)

Owner, before: create the Vast.ai account, add prepaid credit (no automatic top-up), create an
API key, put it in `.env` or the shell (`VAST_API_KEY`), add an SSH public key on the website.

1. **Backstop check** (≈ 0.5 $): rent the cheapest 1-GPU offer with `--max-hours 0.1 --no-guard`
   and watch `vast list`: the instance must disappear by itself about 16 minutes later. If it
   does not, destroy it by hand and keep the guard running for every rental after that.
2. **2 × A40**: `vast offers`, pick the cheapest verified host, `vast rent ... --max-hours 1
   --budget-usd 2 --yes` (the guard runs in that terminal).
3. On the node (`vast show ID` gives the ssh command): clone the repository, `bash
   slurm/single-node.sh`, run one unit and one short ladder (`campaign run --only tp-rag` with a
   short matrix), copy the run file back (`scp`).
4. The guard destroys the instance at the deadline, or Ctrl-C in its terminal destroys it now.
   `vast list` must show nothing left.
5. If the model does not start on Ampere: stop there (protocol §10 fallback, owner's decision).

## Prices seen on 2026-11-11 (public offers, disk of 150 GB included)

| Offer | USD/h | Note |
|---|---|---|
| 2 × A40, verified, reliability ≥ 0.98 | 1.35 to 1.54 | five matching offers, all on hosts in Belgium |
| 4 × A40, verified | 3.00 | one offer |

The protocol budget (§10) assumed 1.07 USD/h for 2 × A40. At these prices the plan comes to about
1.35 (test) + 14.2 (2-GPU campaign) + 6.0 (4-GPU, 2 h) ≈ 21.5 USD, above the 16 USD of the plan
and close to the 20 EUR cap. Whether to cut the plan (fewer repeats or levers, per §10) is decided
at the protocol freeze.
