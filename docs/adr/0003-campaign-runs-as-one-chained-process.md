# ADR 0003: on the rented node, the campaign runs as one chained process

- Status: accepted (2026-11-12); supersedes the rented-node part of ADR 0002
- Decided by: the project owner

## Context

ADR 0002 planned to run the campaign as SLURM jobs, locally and on the rented node. The frozen
protocol (amendment A5) then required the whole 2-GPU matrix in one session on one host, and its
second half depends on the first: the levers run on the best unit, and the repeats on each
profile's best unit, which are known only once the base runs are measured. A SLURM array runs a
fixed list of runs; deriving the next phase needs a step between arrays.

## Decision

- On the rented node, the campaign runs as **one `hpc-sizing campaign run --chain` process**,
  started under `nohup`. It runs the base matrix, derives and runs the levers, the best
  combination and the repeats, and writes each phase's choices to `plan-<phase>.json`.
- SLURM stays for what it ran: the **local dry run** (a Docker cluster, a job array of runs) and
  any fixed list of runs (`slurm/submit.sh`). It was not used on the rented node.

## Why

- One process holds the whole chain and resumes from the run files; no scheduler step is needed
  between phases on a node that belongs to one user.
- ADR 0002 already allowed running the campaign directly on the rented node.

## Consequences

- The 2-GPU campaign of 2026-11-11 ran this way (protocol amendment A7). Claims that it ran as
  SLURM jobs are removed from the README.
- A SLURM version of the chain (one array per phase, `--dependency=afterany` between them) is
  possible but not built.
