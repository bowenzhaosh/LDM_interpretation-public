#!/bin/bash
#SBATCH --job-name=ldm-w4cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel/slurm_logs/w4cpu_%j.out
#
# CPU variant of w4_consolidation.sbatch (2026-08-24, right-sizing).
# exp1_evidence_tracking.py never places the model on a device (core.load_pfn has
# no .to(device)), so the GPU-array tasks ran at 0% GPU util / 4 MiB used with one
# process at 99% of ONE core each (sampled 6x over 60 s on a40-2205, job 224490_0).
# Same code, same env, same eval-seed formula (EVAL = 700 + SEED*7 + DOSE/100), so
# the numbers are identical; this file only drops the GPU request and packs
# several seeds per job as parallel processes, one process per seed so every seed
# keeps its own --eval-seed-base exactly as the GPU array passed it.
# Partition/account/QOS are passed at submit (condo-cse5100 or app-interactive).
#
# Required env (--export; NO COMMAS -- Slurm splits on them):
#   PRIORVAL   AL40 | AL20 | N
#   SCALEVAL   base | mid | large | xl
#   SEEDLIST   colon-separated seeds, e.g. 0:1:2:3
set -euo pipefail

D=/engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel
P=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
cd "$D"
mkdir -p slurm_logs results/exp1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
DOSES="0 100 300 1000 3000 6000 12000"

echo "[w4cpu] prior=${PRIORVAL} scale=${SCALEVAL} seeds=${SEEDLIST} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} start $(date -Is)"

run_seed () {
  local SEED=$1
  local DOSE EVAL
  for DOSE in $DOSES; do
    EVAL=$(( 700 + SEED * 7 + DOSE / 100 ))
    echo "[w4cpu] --- seed=${SEED} dose=${DOSE} eval_seed=${EVAL} $(date -Is)"
    "$P" exp1_evidence_tracking.py \
        --scale "${SCALEVAL}" --prior "${PRIORVAL}" --seeds "${SEED}" \
        --dose "${DOSE}" --eval-seed-base "${EVAL}" --resume
  done
  echo "[w4cpu] seed=${SEED} done $(date -Is)"
}

PIDS=()
for S in ${SEEDLIST//:/ }; do
  run_seed "$S" > "slurm_logs/w4cpu_${SLURM_JOB_ID}_s${S}.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
echo "[w4cpu] done prior=${PRIORVAL} scale=${SCALEVAL} seeds=${SEEDLIST} rc=${RC} $(date -Is)"
exit $RC
