#!/bin/bash
#SBATCH --job-name=ldm-preflight
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=preflight_out/%x-%j.out
#SBATCH --error=preflight_out/%x-%j.out
#
# Amendment F.2a -- the W2 pre-flight. Four gates at 100 steps, before the fleet
# fires, so a persistence or readout failure is found here rather than at 100k.
#
# a6000 rather than rtx4000: the d=3 exact oracle OOMs on 20GB. The job is
# oracle-bound rather than GPU-bound, which is why it asks for 8 CPUs.
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
# Namespaced BY EPS. Two concurrent pre-flights at different eps shared one
# nets/ directory and the same tag_prefix (seed 0 both), so the second job
# overwrote the first's checkpoints mid-run and Gate C could reload another
# job's weights. Caught by inspection on 2026-08-23; both runs were discarded.
OUTROOT="${OUTROOT:-$REPO/campaigns/corrected_20260812/raw/postF/preflight/eps${EPSTAG:-0p5}}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$REPO/src"

mkdir -p "$OUTROOT"
nvidia-smi || true

# -u so the per-gate ticks reach the log as they happen. A gate that prints
# nothing until it finishes is indistinguishable from a gate that has hung,
# which is how two pre-flights burned 2h42m of CPU before anyone could say
# where the time went.
srun "$PY" -u "$REPO/scripts/w2_preflight.py" \
  --eps "${EPSVAL:-0.5}" \
  --steps "${STEPSVAL:-100}" \
  --n-per-half "${NHALFVAL:-24}" \
  --n-tom-contexts "${NTOMVAL:-6}" \
  --lp-solver "${LPSOLVER:-highs}" \
  --ckpt-steps 0 50 100 \
  --outdir "$OUTROOT/nets" \
  --out "$OUTROOT/preflight_eps${EPSTAG:-0p5}.json"

echo "[preflight] done"
