#!/bin/bash
#SBATCH --job-name=ldm-xtrain
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=dose_out/%x-%j.out
#SBATCH --error=dose_out/%x-%j.out
# EXT campaign (REPORTED, never gated): cluster/mech_train.sbatch with the world
# dimension and atom count forwarded (scripts/mech_train.py already exposes --d
# and --K; the locked driver does not pass them). Identical trainer call otherwise.
# Env: EPSVAL (required) SCALEVAL (base) NROWS (20) STEPSVAL (500000)
#      SEEDSVAL ("3 4 5") CKPTS OUTROOT LRVAL (1e-3) DVAL (3) KVAL (8)
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTROOT="${OUTROOT:?OUTROOT is required (never a registered directory)}"
EPSVAL="${EPSVAL:?EPSVAL is required}"
SCALEVAL="${SCALEVAL:-base}"
NROWS="${NROWS:-20}"
STEPSVAL="${STEPSVAL:-500000}"
CKPTS="${CKPTS:-0}"
SEEDSVAL="${SEEDSVAL:-3 4 5}"
LRVAL="${LRVAL:-1e-3}"
DVAL="${DVAL:-3}"
KVAL="${KVAL:-8}"
WSVAL="${WSVAL:-}"          # world seed -> scripts/mech_train_ws.py; empty = the registered atom library via mech_train.py
TRAINER=scripts/mech_train.py; WSARG=""
[ -n "$WSVAL" ] && { TRAINER=scripts/mech_train_ws.py; WSARG="--world-seed $WSVAL"; }

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
guard_not_registered OUTROOT "$OUTROOT"

OMPVAL="${OMPVAL:-4}"   # the locked driver uses 4; 1 is right for >3 seeds per card (python-bound trainer)
export OMP_NUM_THREADS=$OMPVAL MKL_NUM_THREADS=$OMPVAL OPENBLAS_NUM_THREADS=$OMPVAL
export PYTHONPATH="$REPO/src"

mkdir -p "$OUTROOT" dose_out
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO" && git rev-parse --short HEAD
sha256sum $TRAINER src/pfn_dag_verify/corrected_models.py src/pfn_dag_verify/corrected_world.py

pids=()
for S in $SEEDSVAL; do
  "$PY" $TRAINER --eps "$EPSVAL" --seed "$S" --steps "$STEPSVAL" \
      --scale "$SCALEVAL" --n-rows "$NROWS" --ckpt-steps $CKPTS --peak-lr "$LRVAL" \
      --d "$DVAL" --K "$KVAL" $WSARG \
      --outroot "$OUTROOT" \
      > "$OUTROOT/train_${SCALEVAL}_lr${LRVAL}_n${NROWS}_d${DVAL}_K${KVAL}_eps${EPSVAL}_s${S}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; fi
done
echo "[xtrain] ${SCALEVAL} n${NROWS} d${DVAL} K${KVAL} ws${WSVAL:-reg} lr${LRVAL} eps ${EPSVAL} seeds ${SEEDSVAL} done, fail=${fail}"
exit "$fail"
