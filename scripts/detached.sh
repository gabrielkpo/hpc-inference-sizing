#!/bin/bash
# Run a long command detached from this terminal and from the Claude app, so it survives a
# closed window or a crashed app: its own session (no hang-up signal reaches it), output to a
# log file, and on macOS the laptop kept awake while it runs (keep it plugged in, lid open).
#   scripts/detached.sh LOG_FILE COMMAND [ARGS...]
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 LOG_FILE COMMAND [ARGS...]" >&2
  exit 2
fi
log=$1
shift
mkdir -p "$(dirname "$log")"
awake=()
if command -v caffeinate >/dev/null; then awake=(caffeinate -i); fi
nohup python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  "${awake[@]}" "$@" > "$log" 2>&1 < /dev/null &
echo "started pid $! (log: $log)"
