#!/bin/bash
#SBATCH --job-name=p1-replay-debug
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-run2/slurm/p1-replay-debug-%j.out
#SBATCH --error=/engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-run2/slurm/p1-replay-debug-%j.err

set -euo pipefail
exec /engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python -I -S -B \
  /engrfs/project/class/zhao.b/pfn-confirm-diagnostics-2157aa0/pfn-replay-diagnose-2157aa0.py \
  /engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-2157aa0 \
  /engrfs/project/class/zhao.b/pfn-dag-phase1-confirm-v1-fix1-run2/panel
