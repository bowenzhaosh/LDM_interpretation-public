#!/bin/bash
#SBATCH --job-name=ldm-xscore
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=dose_out/%x-%j.out
#SBATCH --error=dose_out/%x-%j.out
# EXT campaign scoring (REPORTED): cluster/mech_score.sbatch's contract, run through
# scripts/mech_predgain_ext.py so the world (DVAL, KVAL) can differ from the
# registered (3, 8). CPU only. The registered gate panel must be given explicitly.
# Env: STEPS EPS NETS_DIR OUT_DIR SCALE_VAL NROWS_VAL SEEDS_VAL KIND_VAL IV_DIR_VAL
#      PANEL_SEED_VAL SPLIT_SEED_VAL NPH_VAL DVAL (3) KVAL (8)
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
STEPS="${STEPS:?STEPS is required}"
EPS="${EPS:?EPS is required}"
NETS_DIR="${NETS_DIR:?NETS_DIR is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

export PYTHONPATH="$REPO/src"
export MECH_NETS="$REPO/$NETS_DIR"
export MECH_OUT="$REPO/$OUT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MECH_SCALE="${SCALE_VAL:-base}" MECH_N_ROWS="${NROWS_VAL:-20}"
export MECH_D="${DVAL:-3}" MECH_K="${KVAL:-8}"
[ -n "${WSVAL:-}" ] && export MECH_WORLD_SEED="$WSVAL"
export MECH_PANEL_SEED="${PANEL_SEED_VAL:?PANEL_SEED_VAL is required}"
export MECH_SPLIT_SEED="${SPLIT_SEED_VAL:?SPLIT_SEED_VAL is required}"
export MECH_N_PER_HALF="${NPH_VAL:?NPH_VAL is required}"
[ -n "${IV_DIR_VAL:-}" ] && export MECH_IV_DIR="$REPO/$IV_DIR_VAL"
export MECH_LADDER=0
# A scored EXT cell is written once, like a registered one (no silent overwrite).
export MECH_CONFIRM=1
KIND="${KIND_VAL:-orig}"
SEEDS="${SEEDS_VAL:-3 4 5}"

# Registered-campaign guard (fail-closed). A bare pattern test on the raw string is
# fail-open for a relative path, for the campaign directory itself, and for a '..'
# traversal — all three were reachable (cluster/tests/test_ext_drivers.sh T6).
guard_not_registered() {   # NAME VALUE -> exit 7 unless VALUE is safely outside campaigns/mech_20260827
  local name="$1" p="$2"
  case "$p" in /*) : ;; *) echo "REFUSING: $name must be an absolute path (got '$p')"; exit 7 ;; esac
  case "$p" in *..*) echo "REFUSING: $name must not contain '..' (got '$p')"; exit 7 ;; esac
  if command -v realpath >/dev/null 2>&1; then p=$(realpath -m -- "$p"); fi
  case "${p%/}/" in
    */campaigns/mech_20260827/*) echo "REFUSING: $name is inside the registered campaign ($p)"; exit 7 ;;
  esac
}
guard_not_registered MECH_OUT "$MECH_OUT"
if [ "$KIND" = "n40ext" ] && [ "$MECH_N_ROWS" != "20" ]; then echo "n40ext scoring replays the n20 panel: NROWS_VAL must be 20"; exit 2; fi

cd "$REPO" && mkdir -p dose_out "$MECH_OUT"
git rev-parse --short HEAD
for st in $STEPS; do
  "$PY" scripts/mech_predgain_ext.py run --eps $EPS --step "$st" --m-q 8 --jobs 14 --kind "$KIND" --seeds $SEEDS
  "$PY" scripts/mech_predgain_ext.py summarise --step "$st"
done
echo "[xscore] done"
