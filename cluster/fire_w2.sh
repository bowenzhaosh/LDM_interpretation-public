#!/usr/bin/env bash
# Amendment F -- fire the W2 Track-B retrain. THREE seeds, seven eps each,
# one GPU per seed job (the seven processes share the card; see w2_fleet.sbatch).
#
# PRECONDITION, checked here and not assumed: AMENDMENT_F.sha256 must exist on
# the commit the mirror is sitting on. W2 is the measurement the amendment
# registers; firing it against an unlocked instrument produces a number with no
# gate behind it, which is the whole failure mode the prereg discipline exists
# to prevent. This script refuses rather than warns.
#
# Usage:  bash cluster/fire_w2.sh [OUTROOT]
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
OUTROOT="${1:-$REPO/campaigns/corrected_20260812/raw/postF/w2}"
SEEDS="${SEEDS:-0 1 2}"

echo "[fire_w2] syncing the mirror to origin/main"
ssh washu "cd $REPO && git fetch --quiet origin && git checkout --quiet main && git pull --ff-only --quiet origin main && git rev-parse --short HEAD"

echo "[fire_w2] checking the lock"
if ! ssh washu "test -f $REPO/AMENDMENT_F.sha256"; then
  echo "REFUSING: AMENDMENT_F.sha256 is absent on the mirror. The amendment is" >&2
  echo "not locked, so W2 has no registered gate to report into. Lock first." >&2
  exit 2
fi
# The lock file follows AMENDMENT_E.sha256's prose format ("<name> sha256 <hex>"
# then human lines), NOT sha256sum's "<hex>  <name>" -- so `sha256sum -c` cannot
# parse it. Pull the recorded hex out and compare it ourselves.
RECORDED=$(ssh washu "cd $REPO && head -1 AMENDMENT_F.sha256 | awk '{print \$3}'")
ACTUAL=$(ssh washu "cd $REPO && sha256sum AMENDMENT_F.md | awk '{print \$1}'")
if [ -z "$RECORDED" ] || [ "$RECORDED" != "$ACTUAL" ]; then
  echo "REFUSING: the amendment does not match its recorded digest." >&2
  echo "  recorded $RECORDED" >&2
  echo "  actual   $ACTUAL" >&2
  exit 3
fi
echo "[fire_w2] lock verified: $ACTUAL"

echo "[fire_w2] launching seeds: $SEEDS -> $OUTROOT"
for S in $SEEDS; do
  ssh washu "cd $REPO && mkdir -p w2_out '$OUTROOT' && \
    SEEDVAL=$S OUTROOT='$OUTROOT' sbatch --parsable cluster/w2_fleet.sbatch"
done
echo "[fire_w2] submitted. Reconcile with: ssh washu 'squeue -u zhao.b'"
