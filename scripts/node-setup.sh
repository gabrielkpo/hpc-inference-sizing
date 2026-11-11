#!/bin/bash
# Prepare a rented GPU node (vllm/vllm-openai image) for the bench, so no paid minute is spent
# typing commands. Run as root on the node, from the home directory:
#   curl -fsSL https://raw.githubusercontent.com/gabrielkpo/hpc-inference-sizing/main/scripts/node-setup.sh -o node-setup.sh
#   nohup bash node-setup.sh > node-setup.log 2>&1 < /dev/null &   # survives an ssh drop
# or, once the repository is cloned: bash scripts/node-setup.sh
# Steps: report the node, install git and uv, clone the repository, install the bench, fetch
# the bench data and the model weights. Each step is skipped when already done.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/gabrielkpo/hpc-inference-sizing.git}
REPO_DIR=${REPO_DIR:-$HOME/hpc-inference-sizing}
MODEL=${MODEL:-mistralai/Mistral-Small-4-119B-2603-NVFP4}

step() { printf '\n== %s (%s)\n' "$1" "$(date -u +%H:%M:%S)"; }

step "node"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.current \
  --format=csv,noheader || echo "WARNING: nvidia-smi failed"
python3 -c 'import vllm; print("vllm", vllm.__version__)' || echo "WARNING: vllm not importable"
df -h "$HOME" | tail -1

step "tools"
if ! command -v git >/dev/null; then
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q git
fi
if ! command -v uv >/dev/null; then
  # pip may refuse a system install (PEP 668): fall back to the official installer, on PATH
  python3 -m pip install -q uv 2>/dev/null ||
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
uv --version

step "repository"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -q "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1

step "bench"
cd "$REPO_DIR"
uv sync -q
# PIAF and the tokenizer (a few MB): fetched now so the first run does not wait on them.
uv run python -c 'from hpc_inference_sizing.bench import data
data.load_piaf(); data.MistralTokenizer(); print("PIAF and tokenizer ok")'

step "weights"
# About 76 GB. vLLM would fetch them on the first start; fetching them here shows the
# download speed and fails early if the disk is too small.
if command -v hf >/dev/null; then
  hf download "$MODEL" >/dev/null
else
  uvx --from huggingface_hub hf download "$MODEL" >/dev/null
fi
du -sh "${HF_HOME:-$HOME/.cache/huggingface}" | tail -1
df -h "$HOME" | tail -1

step "ready"
echo "next: cd $REPO_DIR && uv run hpc-sizing campaign run --matrix campaign/test-session.toml"
