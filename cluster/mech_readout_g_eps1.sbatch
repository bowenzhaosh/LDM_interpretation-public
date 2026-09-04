#!/bin/bash
#SBATCH --job-name=ldm-G-readout-eps1
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=dose_out/%x-%j.out
#SBATCH --error=dose_out/%x-%j.out
# FIGURE_PLAN.md C1 (2026-09-01): the Amendment F LP-projection readout at eps = 1.0
# on the registered gate panel and the registered 500k fleet (seeds 3 4 5) -- the
# third rung of G4's dial. REPORTED, NOT GATED: G4 is registered at eps 0.5 and
# 0.75 only and scripts/mech_gates.py reads only those two files, so nothing this
# job writes can enter the verdict. This file is NOT in the G.0 digest set; it is
# cluster/mech_readout_g.sbatch (digested, --eps 0.5 0.75 hardcoded) with the eps
# list changed and its own job name. Same env, same stages, same write-once guards
# (stage_oracles refuses if eps1p0.npz exists; stage_project refuses to re-project).
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
CAMP=campaigns/mech_20260827

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MECH_CONFIRM=1
export MECH_PANEL_SEED=770000101 MECH_SPLIT_SEED=880000101 MECH_N_PER_HALF=1000
export MECH_PHASE1_OUT="$CAMP/confirm/phase1"
export MECH_NETS="$CAMP/confirm/base_lr0.001_d500000/nets"
export MECH_SCALE=base MECH_N_ROWS=20

cd "$REPO" && mkdir -p dose_out "$MECH_PHASE1_OUT"
git rev-parse --short HEAD

"$PY" scripts/mech_phase1.py oracles --eps 1.0 --sets
"$PY" scripts/mech_phase1.py project --steps 500000 --seeds 3 4 5 --eps 1.0
echo "[G-readout-eps1] done"
