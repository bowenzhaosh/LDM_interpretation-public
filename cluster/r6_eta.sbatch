#!/bin/bash
#SBATCH --job-name=ldm-r6
#SBATCH --partition=linuxlab-project
#SBATCH --account=engr-class-any
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel/slurm_logs/r6_%A_%a.out
#
# R6 -- Amendment E.7 (D7), the eta ceiling. One (d, prior, seed) cell per array task.
#
# CPU on purpose. The nets are small and the arithmetic is bin-grid work, so this
# runs perfectly well off the GPU -- and the GPUs are carrying the W4 consolidation
# fleet, which genuinely needs them.
#
# Required env (--export; NO COMMAS):
#   DVAL       2 | 3
#   PRIORVAL   d=2: AL40 | AL20 | N        d=3: A | C  (N is the Gaussian-null arm
#                                          and is NOT supported here -- exp4 never
#                                          passes gaussian=True to its generator, so
#                                          that arm generates AL data against Gaussian
#                                          anchors; r6_run raises rather than produce
#                                          a finite meaningless number)
#   SCALEVAL   d=2 only: base | mid | large | xl
#   DOSEVAL    d=2: 12000    d=3: 20000
set -euo pipefail

D=/engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel
P=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
cd "$D"
mkdir -p slurm_logs r6_out

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SEED=$SLURM_ARRAY_TASK_ID
SCALE="${SCALEVAL:-d3}"
OUT="r6_out/r6_d${DVAL}_${PRIORVAL}_${SCALE}_s${SEED}_dose${DOSEVAL}.json"
EVAL=$(( 900 + SEED ))

echo "[r6] d=${DVAL} prior=${PRIORVAL} scale=${SCALE} seed=${SEED} host=$(hostname) start $(date -Is)"

if [ "${DVAL}" = "3" ]; then
  srun "$P" r6_run.py --d 3 --prior "${PRIORVAL}" --seed "${SEED}" --dose "${DOSEVAL}" \
       --n-pairs 800 --M 20 --M-select 500 --eval-seed "${EVAL}" \
       --nets-dir /engrfs/project/class/zhao.b/pfn-dag-e18b/nets3 --out "$OUT"
else
  srun "$P" r6_run.py --d 2 --prior "${PRIORVAL}" --scale "${SCALEVAL}" \
       --seed "${SEED}" --dose "${DOSEVAL}" \
       --n-pairs 800 --M 20 --eval-seed "${EVAL}" --out "$OUT"
fi

echo "[r6] done -> $OUT rc=$? $(date -Is)"
