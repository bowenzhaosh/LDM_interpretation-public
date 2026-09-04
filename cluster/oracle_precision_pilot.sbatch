#!/bin/bash
set -euo pipefail
umask 077

# Oracle-precision pilot shard runner.
# usage: sbatch oracle_precision_pilot.sbatch --source-root ABS --out-root ABS --row-start N --row-count M

SRC=""
OUTROOT=""
ROW_START=""
ROW_COUNT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root) SRC="$2"; shift 2 ;;
    --out-root) OUTROOT="$2"; shift 2 ;;
    --row-start) ROW_START="$2"; shift 2 ;;
    --row-count) ROW_COUNT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -n "$SRC" && -n "$OUTROOT" && -n "$ROW_START" && -n "$ROW_COUNT" ]] || { echo "missing args" >&2; exit 2; }
: "${SLURM_JOB_ID:?must run inside a Slurm allocation}"
SRC="$(/usr/bin/readlink -f -- "$SRC")"
OUTROOT="$(/usr/bin/readlink -f -- "$OUTROOT")"
SHARD_DIR="${OUTROOT}/rows_${ROW_START}_${ROW_COUNT}"
if [[ -e "$SHARD_DIR" ]]; then
  echo "shard output already exists: $SHARD_DIR" >&2
  exit 2
fi
mkdir -p "$SHARD_DIR"

# locked interpreter and environment
PY="/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python"
export PYTHONPATH="${SRC}/src"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

PANEL="${SRC}/campaigns/phase1_ordering_20260803/ordering_confirmation_v1_fix2_run1/panel"
CONFIG="${SRC}/config/oracle_precision_pilot_v1.json"

cd "$SRC"
"$PY" -m pfn_dag_verify.pilot_score \
  --config "$CONFIG" \
  --panel "$PANEL" \
  --out "$SHARD_DIR" \
  --device cuda \
  --row-start "$ROW_START" \
  --row-count "$ROW_COUNT" \
  > "$SHARD_DIR/stdout.log" 2> "$SHARD_DIR/stderr.log"

echo "pilot shard complete: $ROW_START + $ROW_COUNT rows"
