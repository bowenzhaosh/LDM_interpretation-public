#!/bin/bash
#SBATCH --job-name=ldm-mscore
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=dose_out/%x-%j.out
#SBATCH --error=dose_out/%x-%j.out
# Mechanism phase 3 scoring -- projection-free eta_order on the dose-extension
# checkpoints. CPU only (exact posteriors + a 279k-param forward pass per query).
# Runs on the cluster so the controller box stays quiet at night.
# Env:  STEPS (default "100000 150000 200000 300000 400000 500000")
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
STEPS="${STEPS:-100000 150000 200000 300000 400000 500000}"
EPS="${EPS:-0.5 0.75 1.0}"

export PYTHONPATH="$REPO/src"
export MECH_NETS="${NETS_DIR:-$REPO/campaigns/mech_20260827/dose_ext/nets}"
export MECH_OUT="${OUT_DIR:-$REPO/campaigns/mech_20260827/predgain_doseext}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# contrast fleets: pass the architecture / context length the nets were trained with
export MECH_SCALE="${SCALE_VAL:-base}" MECH_N_ROWS="${NROWS_VAL:-20}"
# Amendment G (confirmatory panel): pass PANEL_SEED_VAL / SPLIT_SEED_VAL / NPH_VAL to
# score on the registered fresh panel; LADDER_VAL=0 disables the exploratory
# fitted ladders (they are indexed by context on the exploration's panel);
# KIND_VAL=n40ext scores the paired n40 extension (build file under IV_DIR_VAL).
[ -n "${PANEL_SEED_VAL:-}" ] && export MECH_PANEL_SEED="$PANEL_SEED_VAL"
[ -n "${SPLIT_SEED_VAL:-}" ] && export MECH_SPLIT_SEED="$SPLIT_SEED_VAL"
[ -n "${NPH_VAL:-}" ] && export MECH_N_PER_HALF="$NPH_VAL"
[ -n "${IV_DIR_VAL:-}" ] && export MECH_IV_DIR="$IV_DIR_VAL"
export MECH_LADDER="${LADDER_VAL:-1}"
KIND="${KIND_VAL:-orig}"
SEEDS="${SEEDS_VAL:-0 1 2}"
# CONFIRM_VAL=1: Amendment G scoring. The registered panel MUST be given explicitly
# (no silent fallback to the exploration panel), cells are scored exactly once,
# and n40ext scoring replays the n_rows-20 panel (MECH_N_ROWS must stay 20).
if [ "${CONFIRM_VAL:-0}" = "1" ]; then
  export MECH_CONFIRM=1
  : "${PANEL_SEED_VAL:?CONFIRM_VAL=1 requires PANEL_SEED_VAL}"
  : "${SPLIT_SEED_VAL:?CONFIRM_VAL=1 requires SPLIT_SEED_VAL}"
  : "${NPH_VAL:?CONFIRM_VAL=1 requires NPH_VAL}"
  [ "$MECH_LADDER" = "0" ] || { echo "CONFIRM_VAL=1 requires LADDER_VAL=0"; exit 2; }
fi
if [ "$KIND" = "n40ext" ] && [ "$MECH_N_ROWS" != "20" ]; then echo "n40ext scoring replays the n20 panel: NROWS_VAL must be 20"; exit 2; fi

cd "$REPO" && mkdir -p dose_out "$MECH_OUT"
for st in $STEPS; do
  "$PY" scripts/mech_predgain.py run --eps $EPS --step "$st" --m-q 8 --jobs 14 --kind "$KIND" --seeds $SEEDS
  "$PY" scripts/mech_predgain.py summarise --step "$st"
done
echo "[mscore] done"
