#!/bin/bash
#SBATCH --job-name=ldm-w2
#SBATCH --partition=condo-cse5100
#SBATCH --account=engr-acad-cse5100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=w2_out/%x-%j.out
#SBATCH --error=w2_out/%x-%j.out
#
# Amendment F -- W2, the retrain on the repaired instrument. One job per SEED,
# all seven eps running CONCURRENTLY on one GPU.
#
# Packing, not laziness. The base PFN is 279k parameters, so a single training
# process leaves an a6000 almost entirely idle and the per-step cost is kernel
# launch rather than arithmetic. Seven of them share the card comfortably (well
# under 49 GB) and the job's wall becomes the slowest eps rather than their sum.
# The eval stage is CPU-bound -- an exact posterior per scored query -- which is
# why this asks for 16 cores rather than the 2 an LP job would want.
#
# Env:
#   SEEDVAL   the seed this job owns
#   OUTROOT   where the artifacts and nets go
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTROOT="${OUTROOT:-$REPO/campaigns/corrected_20260812/raw/postF/w2}"
SEEDVAL="${SEEDVAL:?SEEDVAL is required}"
STEPSVAL="${STEPSVAL:-100000}"
NHALFVAL="${NHALFVAL:-500}"

# Seven concurrent processes on 16 cores: give each a small, fixed thread budget
# rather than letting each grab the whole node and thrash.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src"

mkdir -p "$OUTROOT" w2_out
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

pids=()
for E in 0.0 0.1 0.25 0.35 0.5 0.75 1.0; do
  "$PY" -m pfn_dag_verify.w2_run \
      --eps "$E" --seed "$SEEDVAL" --steps "$STEPSVAL" \
      --n-per-half "$NHALFVAL" --outroot "$OUTROOT" \
      > "$OUTROOT/w2_s${SEEDVAL}_eps${E}.log" 2>&1 &
  pids+=($!)
done

# Wait on each explicitly so ONE failing cell fails the job. `wait` with no
# arguments returns 0 whatever the children did, which would report a fleet that
# lost a cell as a clean run.
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; fi
done

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
echo "[w2] seed ${SEEDVAL} done, fail=${fail}"
exit "$fail"
