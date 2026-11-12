#!/bin/bash
# Two serving units on one 4-GPU node behind a round-robin balancer (protocol §8, H4): the same
# vLLM command on GPUs 0-1 (port 8001) and GPUs 2-3 (port 8002), then nginx on port 8000 once
# both answer /health. The campaign runner starts this script in its own process group and stops
# the whole group, so the two servers and nginx stop together.
#   bash scripts/serve-two-units.sh MODEL [EXTRA VLLM FLAGS...]
set -euo pipefail
model=$1
shift
common=(--host 127.0.0.1 --served-model-name small4 --tokenizer-mode mistral --config-format mistral
  --load-format mistral --attention-backend TRITON_MLA --max-model-len 8192
  --gpu-memory-utilization 0.90 --tensor-parallel-size 2 "$@")

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$model" --port 8001 "${common[@]}" &
first=$!
CUDA_VISIBLE_DEVICES=2,3 vllm serve "$model" --port 8002 "${common[@]}" &
second=$!
trap 'kill 0' EXIT

for port in 8001 8002; do
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
    # a unit that died at start would leave this loop waiting until the runner's timeout
    { kill -0 "$first" && kill -0 "$second"; } 2>/dev/null || { echo "a vLLM unit exited during startup" >&2; exit 1; }
    sleep 5
  done
done

conf=$(mktemp -d)/nginx.conf
cat > "$conf" <<NGINX
worker_processes 1;
pid $(dirname "$conf")/nginx.pid;
error_log stderr warn;
events { worker_connections 4096; }
http {
  access_log off;
  upstream units { server 127.0.0.1:8001; server 127.0.0.1:8002; }
  server {
    listen 127.0.0.1:8000;
    location / {
      proxy_pass http://units;
      proxy_http_version 1.1;
      proxy_buffering off;
      proxy_read_timeout 600s;
    }
  }
}
NGINX
nginx -c "$conf" -g 'daemon off;'
