#!/bin/bash
# One run of a campaign matrix per array task. Submit with slurm/submit.sh, which sets the array
# size and throttles it to one task at a time (%1): a node serves one model at a time.
#SBATCH --job-name=hpc-sizing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=runs/slurm/%x-%A_%a.out
set -euo pipefail

: "${MATRIX:?set MATRIX to a campaign matrix, e.g. campaign/a40-x2.toml}"
RUN_ID=$(uv run --no-dev -q hpc-sizing campaign list --matrix "$MATRIX" \
  | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "array task ${SLURM_ARRAY_TASK_ID} on $(hostname): run ${RUN_ID}"

# The runner skips a run whose result already exists, so a resubmitted array resumes.
uv run --no-dev -q hpc-sizing campaign run --matrix "$MATRIX" --only "$RUN_ID" ${CAMPAIGN_ARGS:-}
