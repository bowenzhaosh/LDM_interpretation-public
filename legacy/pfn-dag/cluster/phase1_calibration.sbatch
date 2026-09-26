#!/bin/bash
set -euo pipefail

: "${PHASE1_ROOT:?set PHASE1_ROOT to the exact source checkout}"
: "${PHASE1_OUT:?set PHASE1_OUT to a persistent output path}"
: "${PHASE1_MODE:?set PHASE1_MODE to inventory, calibrate, or qualify}"

PHASE1_PYTHON="/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="1"
export PYTHONPATH="${PHASE1_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${PHASE1_OUT}"
cd "${PHASE1_ROOT}"

"${PHASE1_PYTHON}" -m pip check
nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total --format=csv \
  > "${PHASE1_OUT}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 30 > "${PHASE1_OUT}/gpu_utilization.csv" &
PHASE1_MONITOR_PID=$!
trap 'kill "${PHASE1_MONITOR_PID}" 2>/dev/null || true' EXIT

if [[ "${PHASE1_MODE}" == "inventory" ]]; then
  "${PHASE1_PYTHON}" -m pfn_dag_verify.phase1_ordering runtime-inventory \
    --device cuda --out "${PHASE1_OUT}/phase1-washu-binary-inventory.json"
elif [[ "${PHASE1_MODE}" == "calibrate" ]]; then
  : "${PHASE1_CONFIG:?set PHASE1_CONFIG for calibration}"
  : "${PHASE1_RUN_DIR:?set PHASE1_RUN_DIR for calibration}"
  PHASE1_RECOVERY_ARGS=()
  if [[ "${PHASE1_RECOVER_STALE:-0}" == "1" ]]; then
    PHASE1_RECOVERY_ARGS+=(--recover-stale-lease)
  elif [[ "${PHASE1_RECOVER_STALE:-0}" != "0" ]]; then
    echo "PHASE1_RECOVER_STALE must be 0 or 1" >&2
    exit 2
  fi
  "${PHASE1_PYTHON}" -m pfn_dag_verify.phase1_ordering calibrate \
    --device cuda --config "${PHASE1_CONFIG}" --out "${PHASE1_RUN_DIR}" \
    "${PHASE1_RECOVERY_ARGS[@]}"
elif [[ "${PHASE1_MODE}" == "qualify" ]]; then
  : "${PHASE1_CONFIG:?set PHASE1_CONFIG for qualification}"
  : "${PHASE1_RUN_DIR:?set PHASE1_RUN_DIR for qualification}"
  PHASE1_RECOVERY_ARGS=()
  if [[ "${PHASE1_RECOVER_STALE:-0}" == "1" ]]; then
    PHASE1_RECOVERY_ARGS+=(--recover-stale-lease)
  elif [[ "${PHASE1_RECOVER_STALE:-0}" != "0" ]]; then
    echo "PHASE1_RECOVER_STALE must be 0 or 1" >&2
    exit 2
  fi
  "${PHASE1_PYTHON}" -m pfn_dag_verify.phase1_ordering qualify \
    --device cuda --config "${PHASE1_CONFIG}" --out "${PHASE1_RUN_DIR}" \
    "${PHASE1_RECOVERY_ARGS[@]}"
else
  echo "unsupported PHASE1_MODE=${PHASE1_MODE}" >&2
  exit 2
fi
