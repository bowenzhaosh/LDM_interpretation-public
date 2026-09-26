#!/bin/bash
#SBATCH --job-name=ldm-w4
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel/slurm_logs/w4_%A_%a.out
#
# W4 -- d=2 consolidation scoring over the dose x scale x prior grid.
#
# The checkpoints for the whole grid already exist: 1092 .pt files under
# config.DOSE_NETS, which is 4 scales x 3 priors x 7 doses x {16 seeds at base,
# 12 elsewhere}. What has never run is the SCORING: results/exp1 holds 48 files,
# all at prior=AL40 and dose=12000. This array closes that gap.
#
# One task per (scale, prior, seed); the task loops the seven doses in
# config.DEFAULT_DOSE_STEPS. exp1's --resume skips a cell whose JSON is already
# on disk, so re-firing this array after a partial drain costs nothing and is
# the intended way to backfill.
#
# Sizing, from the wallclock_s recorded in the 52 existing results (median per
# cell): base 511 s, mid 724 s, large 1224 s, xl 2358 s. Seven doses per task
# puts base near 1.0 h and xl near 4.6 h, both inside the 16 h wall with room
# for a slow node.
#
# Required env (--export; NO COMMAS -- Slurm splits on them):
#   PRIORVAL   AL40 | AL20 | N
#   SCALEVAL   base | mid | large | xl
set -euo pipefail

D=/engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel
P=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
cd "$D"
mkdir -p slurm_logs results/exp1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SEED=$SLURM_ARRAY_TASK_ID
DOSES="0 100 300 1000 3000 6000 12000"

echo "[w4] prior=${PRIORVAL} scale=${SCALEVAL} seed=${SEED} host=$(hostname) start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DOSE in $DOSES; do
  # The eval seed must be a deterministic function of the cell, not of the array
  # index, so a re-fire that lands on a different node reproduces the same panel.
  EVAL=$(( 700 + SEED * 7 + DOSE / 100 ))
  echo "[w4] --- dose=${DOSE} eval_seed=${EVAL} $(date -Is)"
  "$P" exp1_evidence_tracking.py \
      --scale "${SCALEVAL}" --prior "${PRIORVAL}" --seeds "${SEED}" \
      --dose "${DOSE}" --eval-seed-base "${EVAL}" --resume
done

echo "[w4] done prior=${PRIORVAL} scale=${SCALEVAL} seed=${SEED} rc=$? $(date -Is)"
