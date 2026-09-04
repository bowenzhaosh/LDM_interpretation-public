#!/bin/bash
#SBATCH --job-name=ldm-deficit
#SBATCH --partition=linuxlab-project
#SBATCH --account=engr-class-any
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=ident_out/%x-%j.out
#SBATCH --error=ident_out/%x-%j.out
#
# Amendment F.2b -- produce the PRIMARY x-axis artifact for one eps cell.
# Required env: EPSVAL, TAGVAL
set -euo pipefail
REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTDIR="${OUTDIR:-$REPO/campaigns/corrected_20260812/raw/postF/deficit}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONPATH="$REPO/src"
mkdir -p "$OUTDIR"
echo "[deficit] eps=${EPSVAL} host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(cd $REPO && git rev-parse --short HEAD)"
srun "$PY" -m pfn_dag_verify.corrected_deficit_run \
  --eps "${EPSVAL}" --out "$OUTDIR/deficit_eps${TAGVAL}.json"
