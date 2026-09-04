#!/usr/bin/env -S -i /bin/bash --noprofile --norc
set -euo pipefail
umask 077

PHASE1_LAUNCHER_DIR="$(/usr/bin/readlink -f -- "$(/usr/bin/dirname -- "${BASH_SOURCE[0]}")")"
exec /usr/bin/env -i \
  PATH=/usr/bin:/bin LC_ALL=C TZ=UTC \
  SLURM_CONF=/project/compute/slurm/etc/slurm.conf \
  /engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python -I -S -B \
  "${PHASE1_LAUNCHER_DIR}/submit_phase1_confirmation.py" "$@"
