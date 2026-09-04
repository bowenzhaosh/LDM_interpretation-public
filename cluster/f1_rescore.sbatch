#!/bin/bash
#SBATCH --job-name=ldm-f1rescore
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/engrfs/project/class/zhao.b/LDM_interpretation/c4_out/%x-%j.out
#SBATCH --error=/engrfs/project/class/zhao.b/LDM_interpretation/c4_out/%x-%j.out
#
# F.1 / E.3 -- the REGISTERED bin-summed estimand, over W2's persisted 100k
# checkpoints. Does NOT modify the locked scorer; see the script's header.
#
# CPU only: a 279k-parameter forward plus an exact posterior per query. Runs as a
# batch job rather than on the login node because one cell exceeds two minutes and
# the login node is shared -- and because node-local /tmp is not shared storage,
# which is how an earlier interactive attempt wrote its output into the void.
set -euo pipefail
R=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
export PYTHONPATH="$R/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$R/c4_out"
cd "$R"
OUT="$R/campaigns/corrected_20260812/raw/postF/f1_rescore"
mkdir -p "$OUT"

# One process per seed, three in parallel; each walks all seven eps.
pids=()
for S in 0 1 2; do
  "$PY" scripts/f1_exact_expectation_rescore.py \
      --seeds "$S" --device cpu --status "${STATUSVAL:-live}" \
      --netroot "$R/campaigns/corrected_20260812/raw/postF/w2" \
      --outdir "$OUT" > "$OUT/f1_s${S}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do if ! wait "$p"; then fail=1; fi; done
echo "[f1rescore] done fail=${fail}"
exit "$fail"
