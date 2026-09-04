#!/bin/bash
#SBATCH --job-name=ldm-f4
#SBATCH --partition=linuxlab-project
#SBATCH --account=engr-class-any
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=ident_out/%x-%j.out
#SBATCH --error=ident_out/%x-%j.out
#
# Amendment F.4 -- one cell of the eps=0.5 attribution 2x2.
# Required env: SEMVAL {postfix,prefix}, XOVAL {current,legacy}, TAGVAL
set -euo pipefail
REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTDIR="${OUTDIR:-$REPO/campaigns/corrected_20260812/raw/postF/f4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONPATH="$REPO/src"
mkdir -p "$OUTDIR"
echo "[f4] sem=${SEMVAL} crossover=${XOVAL} host=$(hostname) job=${SLURM_JOB_ID:-none}"
srun "$PY" "$REPO/scripts/f4_attribution_2x2.py" \
  --sem "${SEMVAL}" --crossover "${XOVAL}" --eps 0.5 --solver highs-ipm \
  --out "$OUTDIR/f4_${TAGVAL}.json"
