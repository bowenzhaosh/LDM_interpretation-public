#!/bin/bash
#SBATCH --job-name=ldm-ident
#SBATCH --partition=linuxlab-project
#SBATCH --account=engr-class-any
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=ident_out/%x-%j.out
#SBATCH --error=ident_out/%x-%j.out
#
# Amendment E / D2 -- exact identified-set ladder, one (eps, solver) cell per job.
#
# Moved off the Mac at Bo's instruction (2026-08-21). The eps=0.5 x highs cell
# ran 3.8h wall on the Mac without finishing; highs-ipm finished the same cell in
# 21 min, so the simplex path degenerates on that cell's conditioning
# (cond_mean 4.1e11 there vs 2.0-2.3e10 at eps 0.1/0.25).
#
# Cell config is left at the script defaults on purpose (d=3, K=8, n_contexts=20,
# tol=1e-3, seed=990200000) so a cluster cell is configuration-identical to the
# Mac cell of the same (eps, solver) and the two are directly comparable.
#
# Required env (pass via --export; values must contain NO commas -- Slurm splits):
#   EPSVAL     eps for this cell
#   SOLVERVAL  highs | highs-ipm
#   TAGVAL     filename tag, e.g. 0p5_highs
#   OUTDIR     directory for the result JSON
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
OUTDIR="${OUTDIR:-$REPO/campaigns/corrected_20260812/raw/postE_washu}"

# scipy/HiGHS is single-threaded here; pin BLAS so it cannot oversubscribe the
# 2 cores we asked for.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONPATH="$REPO/src"

mkdir -p "$OUTDIR"
OUT="$OUTDIR/ident_eps${TAGVAL}.json"

# Environment provenance next to every result. The cluster env is NOT the Mac
# env (scipy 1.15.3 / numpy 2.2.6 vs 1.12.0 / 1.26.4), which means a different
# HiGHS build under a preregistered LP instrument. Recording it is what makes
# the Mac-vs-cluster comparison a control instead of an assumption.
"$PY" - <<PYEOF > "$OUTDIR/ident_eps${TAGVAL}.env.json"
import json, platform, sys
import numpy, scipy
print(json.dumps({
    "host": platform.node(),
    "slurm_job_id": "${SLURM_JOB_ID:-none}",
    "partition": "${SLURM_JOB_PARTITION:-none}",
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "eps": float("${EPSVAL}"),
    "solver": "${SOLVERVAL}",
    "git_head": "$(cd $REPO && git rev-parse HEAD)",
    "amendment": "E",
}, indent=2))
PYEOF

echo "[ident] eps=${EPSVAL} solver=${SOLVERVAL} -> $OUT"
cat "$OUTDIR/ident_eps${TAGVAL}.env.json"

srun "$PY" -m pfn_dag_verify.corrected_identifiability_run \
  --eps "${EPSVAL}" --solver "${SOLVERVAL}" --out "$OUT"

echo "[ident] done eps=${EPSVAL} solver=${SOLVERVAL}"
