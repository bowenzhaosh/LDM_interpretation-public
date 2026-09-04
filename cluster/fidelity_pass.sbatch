#!/bin/bash
#SBATCH --job-name=ldm-fidelity
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=a60-2209
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/engrfs/project/class/zhao.b/LDM_interpretation/fid_out/%x-%j.out
#SBATCH --error=/engrfs/project/class/zhao.b/LDM_interpretation/fid_out/%x-%j.out
#
# Amendment F.2c / F.2c-tol rule (1) — the three-curve fidelity pass.
#
# VENUE IS FORCED, NOT CHOSEN (decision G / errata 1). `panel_sha256` hashes raw
# float64 bytes of a LAPACK-dependent draw, and FOUR hashes are already on record
# for the same eps=0.5 panel, so "WashU" is not specific enough. This pins
# a60-2209 — the single node all three W2 jobs (226981/2/3) ran on, recovered
# from `sacct -j 226981,226982,226983 -o JobID,NodeList` — and the script's
# mandatory pre-flight refuses unless the rebuilt panel hash equals every
# w2_eps{tag}_s*.json panel.panel_sha256. Wherever the hashes match IS the venue.
#
# No GPU. The work is projection LPs (scipy/HiGHS) plus a 279k-parameter forward
# pass on CPU; requesting a card would idle it. The node is pinned for its BLAS,
# not for its accelerator.
#
# THREADS. W2 built its panels under OMP/MKL/OPENBLAS_NUM_THREADS=2
# (w2_fleet.sbatch:39-41), so the parent inherits 2 for the panel-construction
# step, which is the step whose bytes must match. The pool workers then pin
# themselves to 1 in _init (LPs cannot touch panel bytes, and unpinned workers
# took the measured cost from 0.1 s/LP to minutes).
#
# Env:
#   STATUSVAL   diagnostic (default) | live
#   PREFLIGHT   1 to run the pre-flight only, no LP time
#   JOBSVAL     pool size (default 14)
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

mkdir -p "$REPO/fid_out"
cd "$REPO"

ARGS=(--status "${STATUSVAL:-diagnostic}"
      --jobs "${JOBSVAL:-14}"
      --repo "$REPO"
      --w2-dir "$REPO/campaigns/corrected_20260812/raw/postF/w2"
      --deficit-dir "$REPO/campaigns/corrected_20260812/raw/postF/deficit_w2"
      --raw-dir "$REPO/campaigns/corrected_20260812/raw/postF/fidelity")
if [[ "${PREFLIGHT:-0}" == "1" ]]; then
  ARGS+=(--preflight-only)
fi
# EXTRA is for the DIAGNOSTIC smoke only (--eps / --limit / --raw-dir into a
# scratch path). The script itself refuses `--status live` together with
# `--limit`, so a smoke cannot be promoted to the measurement by an env var.
if [[ -n "${EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(${EXTRA})
fi

hostname
"$PY" scripts/f2c_fidelity_pass.py "${ARGS[@]}"
echo "[fidelity] done"
