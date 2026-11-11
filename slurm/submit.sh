#!/bin/bash
# Submit every run of a campaign matrix as a SLURM job array, one task at a time.
#   slurm/submit.sh campaign/a40-x2.toml [extra campaign run args...]
set -euo pipefail

MATRIX=${1:?usage: slurm/submit.sh MATRIX [campaign run args...]}
shift
N=$(uv run --no-dev -q hpc-sizing campaign list --matrix "$MATRIX" | wc -l | tr -d ' ')
mkdir -p runs/slurm
# Arguments go to the batch script as arguments, not through --export (which splits on commas).
sbatch --parsable --array="0-$((N - 1))%1" slurm/campaign.sbatch "$MATRIX" "$@"
