# ADR 0002: the campaign runs as SLURM jobs, locally and on the rented node

- Status: accepted (2026-10-27)
- Decided by: the project owner

## Context

SLURM schedules jobs on shared clusters: it hands GPUs to jobs and queues the rest. It does not
split a model across GPUs; vLLM does that through tensor, expert or pipeline parallelism. A
rented Vast.ai node belongs to one user, so nothing needs scheduling there.

The campaign, however, is a series of finite jobs (start a unit, run a load ladder, save the
results), which is what SLURM is built for. And no paid GPU time may be spent before the whole
chain has run end to end at no cost.

## Decision

- Write each campaign step as an **`sbatch` script**.
- Run those scripts first on a **local SLURM cluster in Docker** (one controller, two nodes)
  against a small model on CPU: the proof of concept.
- Run **the same scripts** on the rented node, with a single-node SLURM inside the container
  declaring its GPUs.
- If SLURM does not start on the rented node during the test session, run the campaign script
  directly. Nothing is debugged on paid time.

## Why

- The dry run exercises the exact path that will run on the GPUs, not a separate wrapper.
- The job scripts work unchanged on any SLURM cluster, which is how GPU time is obtained in
  research and on most HPC systems.

## Consequences

- SLURM in an unprivileged container needs a minimal setup (no cgroup plugins, process tracking
  by Linux process IDs); it is built and tested locally first.
- The local cluster produces no performance numbers: a CPU and a small model say nothing about
  GPUs. Its results never enter the report.
