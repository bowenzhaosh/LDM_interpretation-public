#!/bin/bash
#SBATCH --job-name=ldm-dose
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=dose_out/%x-%j.out
#SBATCH --error=dose_out/%x-%j.out
# Mechanism phase 3 -- DOSE EXTENSION. DIAGNOSTIC, not registered.
#
# Question (WORKLOG 2026-08-27): the W2 dose surface shows in-task regret
# falling as ~dose^-0.55 at eps>=0.5 with no plateau by 100k, so the order-
# capture shortfall reads as undertraining. Does eta_order keep closing on that
# line (pure undertraining) or plateau (structural)? Same world, same recipe,
# same seeds as W2 -- only `steps` changes -- so the 100k checkpoint of this run
# reproduces W2's trajectory and the 150k..500k checkpoints extend it.
#
# One job per EPS, its three seeds sharing the card (W2 packed seven).
# Outputs go to a NEW outroot; nothing under raw/postF/ is touched, and every
# artifact is stamped DIAGNOSTIC via --diagnostic.
# Env:  EPSVAL (required)  STEPSVAL (default 500000)  OUTROOT
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTROOT="${OUTROOT:-$REPO/campaigns/mech_20260827/dose_ext}"
EPSVAL="${EPSVAL:?EPSVAL is required}"
STEPSVAL="${STEPSVAL:-500000}"
CKPTS="${CKPTS:-0 10000 25000 50000 100000 150000 200000 250000 300000 400000 500000}"
# SEEDSVAL: which seeds this job trains concurrently on its card. The 500k pass
# packed three per job; the 2M pass runs ONE per job (2M steps alone ~9-10 h on
# an a6000; three sharing would be ~15 h) and passes --time on the sbatch line.
SEEDSVAL="${SEEDSVAL:-0 1 2}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$REPO/src"

mkdir -p "$OUTROOT" dose_out
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO" && git rev-parse --short HEAD

pids=()
for S in $SEEDSVAL; do
  "$PY" -m pfn_dag_verify.w2_run \
      --eps "$EPSVAL" --seed "$S" --steps "$STEPSVAL" \
      --ckpt-steps $CKPTS --diagnostic \
      --n-per-half 500 --outroot "$OUTROOT" \
      > "$OUTROOT/dose_eps${EPSVAL}_s${S}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; fi
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
echo "[dose] eps ${EPSVAL} done, fail=${fail}"
exit "$fail"
