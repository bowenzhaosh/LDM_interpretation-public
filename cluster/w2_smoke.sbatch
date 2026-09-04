#!/bin/bash
#SBATCH --job-name=ldm-w2smoke
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=w2_out/%x-%j.out
#SBATCH --error=w2_out/%x-%j.out
#
# W2 fleet PLUMBING smoke. Same script, same packing, same sbatch shape as the
# real fleet, at 1000 steps instead of 100000. It answers three questions the
# real launch should not be the first to ask: does seven-way packing fit, what
# GPU utilisation does it actually reach (the firing directive treats under 50%
# as mis-sized), and what does a cell cost end to end.
#
# Every artifact is stamped DIAGNOSTIC, so the verdict loader refuses it. The
# smoke and the measurement share a filename pattern and differ only in step
# count, which is exactly the kind of difference nobody notices in a listing.
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTROOT="${OUTROOT:-$REPO/campaigns/corrected_20260812/raw/postF/w2_smoke}"
SEEDVAL="${SEEDVAL:-0}"
STEPSVAL="${STEPSVAL:-1000}"
NHALFVAL="${NHALFVAL:-500}"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src"
mkdir -p "$OUTROOT" w2_out

# Sample GPU utilisation while the cells run, rather than reading it once at the
# end when everything has already finished and the card is idle.
( while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader \
      >> "$OUTROOT/gpu_util_s${SEEDVAL}.csv" 2>/dev/null || true
    sleep 15
  done ) &
SMPL=$!

pids=()
for E in 0.0 0.1 0.25 0.35 0.5 0.75 1.0; do
  "$PY" -m pfn_dag_verify.w2_run --diagnostic \
      --eps "$E" --seed "$SEEDVAL" --steps "$STEPSVAL" \
      --n-per-half "$NHALFVAL" --ckpt-steps 0 500 1000 \
      --outroot "$OUTROOT" > "$OUTROOT/smoke_s${SEEDVAL}_eps${E}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do if ! wait "$p"; then fail=1; fi; done
kill "$SMPL" 2>/dev/null || true

echo "[w2-smoke] utilisation samples:"
sort -n "$OUTROOT/gpu_util_s${SEEDVAL}.csv" 2>/dev/null | tail -3 || true
awk -F'[ %,]+' '{s+=$1;n++} END{if(n)printf "[w2-smoke] mean GPU util %.1f%% over %d samples\n",s/n,n}' \
    "$OUTROOT/gpu_util_s${SEEDVAL}.csv" 2>/dev/null || true
echo "[w2-smoke] seed ${SEEDVAL} done, fail=${fail}"
exit "$fail"
