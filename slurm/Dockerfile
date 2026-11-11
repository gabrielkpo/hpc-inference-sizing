# Local SLURM cluster for dry runs of the campaign (ADR 0002): one image, three roles
# (controller, two compute nodes). Built and used only on a laptop; no performance number
# comes from it.
FROM python:3.12-slim

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      slurm-wlm munge ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.12.5

# One munge key baked into the image, shared by every container of this local cluster.
RUN mkdir -p /etc/munge && (mungekey --create --force || create-munge-key -f) \
 && chown -R munge:munge /etc/munge && chmod 400 /etc/munge/munge.key \
 && mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm \
 && chown -R slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

COPY slurm.conf cgroup.conf /etc/slurm/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The Python environment lives outside the mounted repository (the host's .venv is for macOS).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
