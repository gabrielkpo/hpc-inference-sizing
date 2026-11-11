#!/bin/bash
# Single-node SLURM inside the rented container (ADR 0002): the same job array as on the local
# cluster, one node that owns all the GPUs. Run as root once the repository is cloned:
#   bash slurm/single-node.sh && slurm/submit.sh campaign/a40-x2.toml
# If SLURM does not come up, run the campaign directly instead (no debugging on paid time):
#   uv run hpc-sizing campaign run --matrix campaign/a40-x2.toml
set -euo pipefail

if ! command -v slurmctld >/dev/null; then
  apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends slurm-wlm munge
fi

NODE=$(hostname -s)
CPUS=$(nproc)
MEM_MB=$(($(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 - 2048))
GPUS=$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ')
GRES=""
if [ "${GPUS}" -gt 0 ]; then GRES="Gres=gpu:${GPUS}"; fi

mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /run/munge
cat > /etc/slurm/slurm.conf <<EOF
ClusterName=hpcsizing
SlurmctldHost=${NODE}
AuthType=auth/munge
SlurmUser=root
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
JobAcctGatherType=jobacct_gather/none
AccountingStorageType=accounting_storage/none
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
MpiDefault=none
ReturnToService=2
GresTypes=gpu
NodeName=${NODE} CPUs=${CPUS} RealMemory=${MEM_MB} ${GRES} State=UNKNOWN
PartitionName=gpu Nodes=${NODE} Default=YES MaxTime=INFINITE State=UP
EOF
printf 'CgroupPlugin=disabled\n' > /etc/slurm/cgroup.conf  # containers: no cgroups, no systemd
if [ "${GPUS}" -gt 0 ]; then
  echo "Name=gpu File=/dev/nvidia[0-$((GPUS - 1))]" > /etc/slurm/gres.conf
else
  : > /etc/slurm/gres.conf  # no GPU (local test of this script)
fi

[ -f /etc/munge/munge.key ] || (mungekey --create --force 2>/dev/null || create-munge-key -f)
chown -R munge:munge /etc/munge /run/munge && chmod 400 /etc/munge/munge.key
su -s /bin/bash munge -c "/usr/sbin/munged --force"

slurmctld -i
slurmd -N "${NODE}"
sleep 3
scontrol update nodename="${NODE}" state=resume 2>/dev/null || true
sinfo
