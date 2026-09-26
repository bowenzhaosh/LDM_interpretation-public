#!/bin/bash
#SBATCH -J au_bd_e4
#SBATCH -p condo-cse5100
#SBATCH -A engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o /engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel/slurm_logs/exp4_%a.out
#SBATCH --array=0-11

cd /engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel
export OMP_NUM_THREADS=8
P=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python

i=$SLURM_ARRAY_TASK_ID
PRIORS=(A C N)
p=$((i/4)); s=$((i%4))
PRIOR=${PRIORS[$p]}; SEED=$s
EVAL=$((601 + i))

echo "exp4 d3 cell idx=$i prior=$PRIOR seed=$SEED eval=$EVAL host=$(hostname) start $(date)"
"$P" exp4_d3_readout.py --prior "$PRIOR" --seeds "$SEED" --dose 20000 --nets-dir /engrfs/project/class/zhao.b/pfn-dag-e18b/nets3 --eval-seed-base "$EVAL" --resume
echo "exp4 d3 cell idx=$i done rc=$? $(date)"
