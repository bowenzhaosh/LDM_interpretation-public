#!/bin/bash
#SBATCH --job-name=ldm-G-readout
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
# Amendment G gate G4: the Amendment F LP-projection readout, run AS LOCKED on
# models it never saw. Two stages, in order, on the registered gate panel:
#   oracles  -- the exact order posterior and the uniform reference per context
#   project  -- corrected_tomography._lp_project_or_ranges (solver highs, mode
#               'project') of each model's Q2 readout, seeds 3 4 5 at 500k
# Both stages are CPU-heavy over 1000 contexts (exact posteriors, then one LP per
# context per seed), so they run here and never on the login node.
#
# The panel env is passed EXPLICITLY. Under MECH_CONFIRM=1 mech_phase1.py refuses
# to run without it, because a missing MECH_PANEL_SEED silently rebuilds the
# EXPLORATION panel (770000000 / 880000000 / 500 contexts) and exits 0.
# --sets is passed with no fitted ladders: G4 needs w_exact and w_uniform only.
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

"$PY" scripts/mech_phase1.py oracles --eps 0.5 0.75 --sets
"$PY" scripts/mech_phase1.py project --steps 500000 --seeds 3 4 5 --eps 0.5 0.75
echo "[G-readout] done"
