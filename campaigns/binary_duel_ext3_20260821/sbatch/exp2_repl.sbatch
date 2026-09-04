#!/bin/bash
#SBATCH -J au_bd_e2
#SBATCH -p condo-cse5100
#SBATCH -A engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o /engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel/slurm_logs/exp2_%a.out
#SBATCH --array=0-47

cd /engrfs/project/class/zhao.b/pfn-dag-e18b/experiments/binary-duel
export OMP_NUM_THREADS=8
P=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python

i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 12 ]; then
  SCALE=base; SEED=$((i+4))
elif [ $i -lt 24 ]; then
  SCALE=mid; SEED=$((i-12))
elif [ $i -lt 36 ]; then
  SCALE=large; SEED=$((i-24))
else
  SCALE=xl; SEED=$((i-36))
fi
EVAL=$((551 + i))

echo "exp2 repl cell idx=$i scale=$SCALE seed=$SEED eval=$EVAL host=$(hostname) start $(date)"
"$P" exp2_mislead.py --scale "$SCALE" --prior AL40 --seeds "$SEED" --dose 12000 --eval-seed-base "$EVAL" --resume
echo "exp2 repl cell idx=$i done rc=$? $(date)"
