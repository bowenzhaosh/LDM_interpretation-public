#!/bin/bash
#SBATCH --job-name=ldm-c4floor
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/engrfs/project/class/zhao.b/LDM_interpretation/c4_out/%x-%j.out
#SBATCH --error=/engrfs/project/class/zhao.b/LDM_interpretation/c4_out/%x-%j.out
#
# Amendment F.2c-tol rule (4) — the pre-lock floor readout calibration.
#
# CPU only. The work is projection LPs (scipy/HiGHS) plus a 279k-parameter
# forward pass; no GPU is requested because none would be used.
#
# WashU is the SIGNED venue (Bo, 2026-08-26) substituting for the Mac, which
# rule (4) names but which refuses SSH. It is the other environment F.4's
# invariance table already characterizes, so the record gains no new one.
# Anywhere else this script would have to run --status diagnostic.
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
export PYTHONPATH="$REPO/src"

# One BLAS thread per worker; the pool sets these too, but a job that inherits
# 16 threads per worker on 16 cores thrashes and the measured cost went from
# 0.1 s per LP to minutes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p "$REPO/c4_out"
cd "$REPO"
"$PY" scripts/f2c_clause4_floor.py \
    --status "${STATUSVAL:-live}" \
    --jobs "${JOBSVAL:-14}" \
    --seed "${SEEDVAL:-0}" \
    --outdir "$REPO/campaigns/corrected_20260812/raw/postF/floor_calib"
echo "[c4] done"
