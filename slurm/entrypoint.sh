#!/bin/bash
# Role of a container of the local SLURM cluster: slurmctld (controller) or slurmd (node).
set -euo pipefail

mkdir -p /run/munge && chown munge:munge /run/munge
gosu_munge() { su -s /bin/bash munge -c "$*"; }
gosu_munge "/usr/sbin/munged --force"

case "${1:-}" in
  slurmctld)
    exec /usr/sbin/slurmctld -D -i
    ;;
  slurmd)
    # Python environment for the jobs, built once per node from the mounted repository.
    (cd /work && uv sync --frozen --no-dev -q) || echo "uv sync failed" >&2
    exec /usr/sbin/slurmd -D -N "$(hostname)"
    ;;
  *)
    exec "$@"
    ;;
esac
