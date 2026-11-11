#!/bin/bash
# Submit every run of a campaign matrix as a SLURM job array, one task at a time.
#   slurm/submit.sh campaign/a40-x2.toml [extra campaign run args...]
set -euo pipefail

MATRIX=${1:?usage: slurm/submit.sh MATRIX [campaign run args...]}
shift
N=$(uv run --no-dev -q hpc-sizing campaign list --matrix "$MATRIX" | wc -l | tr -d ' ')
mkdir -p runs/slurm
sbatch --parsable --array="0-$((N - 1))%1" \
  --export=ALL,MATRIX="$MATRIX",CAMPAIGN_ARGS="$*" slurm/campaign.sbatch
