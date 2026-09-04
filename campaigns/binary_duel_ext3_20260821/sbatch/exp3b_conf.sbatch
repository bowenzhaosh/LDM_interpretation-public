#!/bin/bash
#SBATCH -J au_bd_e3c
#SBATCH -p condo-cse5100
#SBATCH -A engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o /engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel/slurm_logs/exp3c_%a.out
#SBATCH --array=0-7

cd /engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python

i=$SLURM_ARRAY_TASK_ID
SITES=(resid0 query1)
s=$((i/2)); site=$((i%2))
SEED=$s; SITE=${SITES[$site]}
EVAL=$((801 + i))

echo "exp3b_conf cell idx=$i seed=$SEED site=$SITE eval=$EVAL host=$(hostname) start $(date)"
"$P" exp3b_phaseB_conf.py --prior AL40 --seeds "$SEED" --dose 12000 --site "$SITE" --eval-seed-base "$EVAL" --resume
echo "exp3b_conf cell idx=$i done rc=$? $(date)"
