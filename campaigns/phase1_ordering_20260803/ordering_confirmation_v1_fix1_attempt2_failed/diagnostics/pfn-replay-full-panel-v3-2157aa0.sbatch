#!/bin/bash
#SBATCH --job-name=p1-replay-sweep
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-run2/slurm/p1-replay-sweep-%j.out
#SBATCH --error=/engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-run2/slurm/p1-replay-sweep-%j.err

set -euo pipefail
export PATH="/usr/bin:/bin"
export LC_ALL="C"
export TZ="UTC"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="1"
unset LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
exec /engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python -I -S -B \
  /engrfs/project/class/zhao.b/pfn-confirm-diagnostics-2157aa0/pfn-replay-full-panel-v3-2157aa0.py \
  /engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-2157aa0 \
  /engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-run2/panel
