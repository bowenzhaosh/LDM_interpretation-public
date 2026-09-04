#!/bin/bash
#SBATCH --job-name=ldm-mtrain
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
# Mechanism-phase contrasts (DIAGNOSTIC): scale and context-length variants of
# the W2 recipe via scripts/mech_train.py. SEEDSVAL seeds share the card.
# Env: EPSVAL (required) SCALEVAL (base) NROWS (20) STEPSVAL (500000)
#      SEEDSVAL ("0 1 2") CKPTS OUTROOT
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTROOT="${OUTROOT:-$REPO/campaigns/mech_20260827/contrasts}"
EPSVAL="${EPSVAL:?EPSVAL is required}"
SCALEVAL="${SCALEVAL:-base}"
NROWS="${NROWS:-20}"
STEPSVAL="${STEPSVAL:-500000}"
CKPTS="${CKPTS:-0 100000 500000}"
SEEDSVAL="${SEEDSVAL:-0 1 2}"
LRVAL="${LRVAL:-1e-3}"      # registered recipe 1e-3; other values tag the net filenames

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$REPO/src"

mkdir -p "$OUTROOT" dose_out
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO" && git rev-parse --short HEAD

pids=()
for S in $SEEDSVAL; do
  "$PY" scripts/mech_train.py --eps "$EPSVAL" --seed "$S" --steps "$STEPSVAL" \
      --scale "$SCALEVAL" --n-rows "$NROWS" --ckpt-steps $CKPTS --peak-lr "$LRVAL" \
      --outroot "$OUTROOT" \
      > "$OUTROOT/train_${SCALEVAL}_lr${LRVAL}_n${NROWS}_eps${EPSVAL}_s${S}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; fi
done
echo "[mtrain] ${SCALEVAL} n${NROWS} eps ${EPSVAL} done, fail=${fail}"
exit "$fail"
