#!/bin/bash
set -euo pipefail
umask 077

usage() {
  echo "usage: $0 --source-root ABS --attempt-root ABS --mode panel|pfn|oracle|join|verify [--bank 0|1|2]" >&2
  exit 2
}

PHASE1_ROOT_ARG=""
PHASE1_ATTEMPT_ARG=""
PHASE1_MODE=""
PHASE1_BANK_INDEX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root) [[ $# -ge 2 ]] || usage; PHASE1_ROOT_ARG="$2"; shift 2 ;;
    --attempt-root) [[ $# -ge 2 ]] || usage; PHASE1_ATTEMPT_ARG="$2"; shift 2 ;;
    --mode) [[ $# -ge 2 ]] || usage; PHASE1_MODE="$2"; shift 2 ;;
    --bank) [[ $# -ge 2 ]] || usage; PHASE1_BANK_INDEX="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "${PHASE1_ROOT_ARG}" && -n "${PHASE1_ATTEMPT_ARG}" && -n "${PHASE1_MODE}" ]] || usage
[[ "${PHASE1_ROOT_ARG}" == /* && "${PHASE1_ATTEMPT_ARG}" == /* ]] || usage
: "${SLURM_JOB_ID:?confirmation wrapper must run inside a Slurm allocation}"

PHASE1_ROOT="$(/usr/bin/readlink -f -- "${PHASE1_ROOT_ARG}")"
PHASE1_ATTEMPT_ROOT="$(/usr/bin/readlink -f -- "${PHASE1_ATTEMPT_ARG}")"
[[ -d "${PHASE1_ROOT}" && -d "${PHASE1_ATTEMPT_ROOT}" ]] || usage
if [[ "${PHASE1_ATTEMPT_ROOT}/" == "${PHASE1_ROOT}/"* ]]; then
  echo "attempt root must be outside the source checkout" >&2
  exit 2
fi

PHASE1_CONFIG="${PHASE1_ROOT}/config/phase1_ordering_confirmation.json"
PHASE1_PANEL_DIR="${PHASE1_ATTEMPT_ROOT}/panel"
case "${PHASE1_MODE}" in
  panel) PHASE1_STAGE="panel"; PHASE1_STAGE_OUT="${PHASE1_PANEL_DIR}" ;;
  pfn) PHASE1_STAGE="pfn"; PHASE1_STAGE_OUT="${PHASE1_ATTEMPT_ROOT}/pfn" ;;
  oracle)
    [[ "${PHASE1_BANK_INDEX}" =~ ^[012]$ ]] || usage
    PHASE1_STAGE="oracle${PHASE1_BANK_INDEX}"
    PHASE1_STAGE_OUT="${PHASE1_ATTEMPT_ROOT}/oracle_bank${PHASE1_BANK_INDEX}"
    ;;
  join) PHASE1_STAGE="join"; PHASE1_STAGE_OUT="${PHASE1_ATTEMPT_ROOT}/join" ;;
  verify)
    PHASE1_STAGE="verify"
    PHASE1_STAGE_OUT="${PHASE1_ATTEMPT_ROOT}/verification"
    ;;
  *) usage ;;
esac
PHASE1_LOG_DIR="${PHASE1_ATTEMPT_ROOT}/telemetry/${PHASE1_STAGE}-${SLURM_JOB_ID}"

if [[ -e "${PHASE1_STAGE_OUT}" ]]; then
  echo "stage output already exists: ${PHASE1_STAGE_OUT}" >&2
  exit 2
fi
if ! /usr/bin/mkdir "${PHASE1_LOG_DIR}"; then
  echo "stage telemetry directory must be new: ${PHASE1_LOG_DIR}" >&2
  exit 2
fi

PHASE1_PYTHON="/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python"
PHASE1_SITE_PACKAGES="/engrfs/project/class/zhao.b/conda_envs/tidpo/lib/python3.10/site-packages"
PHASE1_PYCACHE_PREFIX="${PHASE1_LOG_DIR}/pycache"
export PATH="/usr/bin:/bin"
export LC_ALL="C"
export TZ="UTC"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="1"
unset LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
cd "${PHASE1_ROOT}"

verify_submission_binding() {
  "${PHASE1_PYTHON}" -I -S -X "pycache_prefix=${PHASE1_PYCACHE_PREFIX}" -c \
    'import json,pathlib,sys,time; manifest=pathlib.Path(sys.argv[1]); stage=sys.argv[2]; job=sys.argv[3]; expected=pathlib.Path(sys.argv[4]).resolve(); pathlib.Path(sys.pycache_prefix).resolve()==expected or (_ for _ in ()).throw(RuntimeError("submission verifier pycache prefix drifted")); deadline=time.monotonic()+120.0
while True:
 value=json.loads(manifest.read_text()); rows=[row for row in value.get("jobs",[]) if row.get("stage")==stage]; len(rows)==1 and str(rows[0].get("job_id"))==job or (_ for _ in ()).throw(RuntimeError("Slurm job is not bound to the submission manifest")); state=value.get("state")
 if state=="SUBMITTED": break
 state=="READY_TO_RELEASE" or (_ for _ in ()).throw(RuntimeError(f"submission manifest is not runnable: {state}")); time.monotonic()<deadline or (_ for _ in ()).throw(RuntimeError("submission release receipt timed out")); time.sleep(0.1)' \
    "${PHASE1_ATTEMPT_ROOT}/SUBMISSION.json" "${PHASE1_STAGE}" \
    "${SLURM_JOB_ID}" "${PHASE1_PYCACHE_PREFIX}"
}

verify_submission_binding

run_locked_module() {
  local module="$1"
  shift
  "${PHASE1_PYTHON}" -I -S -X "pycache_prefix=${PHASE1_PYCACHE_PREFIX}" -c \
    'import pathlib,runpy,sys; root=pathlib.Path(sys.argv[1]).resolve(); site=pathlib.Path(sys.argv[2]).resolve(); expected_cache=pathlib.Path(sys.argv[3]).resolve(); module=sys.argv[4]; prefix=pathlib.Path(sys.base_prefix).resolve(); outside=[value for value in sys.path if value and not pathlib.Path(value).resolve().is_relative_to(prefix)]; forbidden=[name for name in ("site","sitecustomize","usercustomize") if name in sys.modules]; cache_ok=sys.pycache_prefix is not None and pathlib.Path(sys.pycache_prefix).resolve()==expected_cache; (not outside and not forbidden and cache_ok and sys.flags.isolated and sys.flags.no_site and sys.flags.no_user_site) or (_ for _ in ()).throw(RuntimeError(f"non-isolated Python startup: outside={outside}, forbidden={forbidden}, cache={sys.pycache_prefix}, flags={sys.flags}")); sys.path[:0]=[str(root / "src"),str(site)]; sys.argv=[module,*sys.argv[5:]]; runpy.run_module(module,run_name="__main__")' \
    "${PHASE1_ROOT}" "${PHASE1_SITE_PACKAGES}" "${PHASE1_PYCACHE_PREFIX}" \
    "${module}" "$@"
}

run_locked_module pip check
if [[ "${PHASE1_MODE}" != "verify" ]]; then
  "${PHASE1_PYTHON}" -I -S -X "pycache_prefix=${PHASE1_PYCACHE_PREFIX}" -c \
    'import json,os,pathlib,platform,socket,sys; site=pathlib.Path(sys.argv[1]).resolve(); expected_cache=pathlib.Path(sys.argv[2]).resolve(); forbidden=[name for name in ("site","sitecustomize","usercustomize") if name in sys.modules]; cache_ok=sys.pycache_prefix is not None and pathlib.Path(sys.pycache_prefix).resolve()==expected_cache; (sys.flags.isolated and sys.flags.no_site and sys.flags.no_user_site and not forbidden and cache_ok) or (_ for _ in ()).throw(RuntimeError(f"non-isolated telemetry interpreter: {forbidden}, cache={sys.pycache_prefix}, {sys.flags}")); sys.path.insert(0,str(site)); import torch; d=torch.cuda.current_device(); p=torch.cuda.get_device_properties(d); value={"mode":"cuda","hostname":socket.gethostname(),"python_executable":sys.executable,"python":platform.python_version(),"isolated":bool(sys.flags.isolated),"no_site":bool(sys.flags.no_site),"no_user_site":bool(sys.flags.no_user_site),"pycache_prefix":sys.pycache_prefix,"startup_customization_modules":forbidden,"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"slurm_job_gpus":os.environ.get("SLURM_JOB_GPUS"),"slurm_step_gpus":os.environ.get("SLURM_STEP_GPUS"),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch_current_device":d,"torch_device_name":p.name,"torch_device_total_memory":int(p.total_memory),"torch_device_uuid":str(getattr(p,"uuid",None))}; pathlib.Path(sys.argv[3]).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")' \
    "${PHASE1_SITE_PACKAGES}" "${PHASE1_PYCACHE_PREFIX}" \
    "${PHASE1_LOG_DIR}/assigned_device.json"
  /usr/bin/nvidia-smi --query-gpu=timestamp,index,name,uuid,driver_version,memory.total --format=csv \
    > "${PHASE1_LOG_DIR}/gpu_identity.csv"
  /usr/bin/nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 30 > "${PHASE1_LOG_DIR}/gpu_utilization.csv" &
  PHASE1_MONITOR_PID=$!
  trap 'kill "${PHASE1_MONITOR_PID}" 2>/dev/null || true' EXIT
else
  printf '%s\n' '{"mode":"cpu","stage":"verify"}' > "${PHASE1_LOG_DIR}/assigned_device.json"
fi

if [[ "${PHASE1_MODE}" == "panel" ]]; then
  run_locked_module pfn_dag_verify.phase1_panel \
    --device cuda --config "${PHASE1_CONFIG}" --out "${PHASE1_PANEL_DIR}"
elif [[ "${PHASE1_MODE}" == "pfn" ]]; then
  run_locked_module pfn_dag_verify.phase1_pfn \
    --device cuda --config "${PHASE1_CONFIG}" --panel "${PHASE1_PANEL_DIR}" \
    --out "${PHASE1_STAGE_OUT}"
elif [[ "${PHASE1_MODE}" == "oracle" ]]; then
  run_locked_module pfn_dag_verify.phase1_oracle_confirm \
    --device cuda --config "${PHASE1_CONFIG}" --panel "${PHASE1_PANEL_DIR}" \
    --bank "${PHASE1_BANK_INDEX}" --out "${PHASE1_STAGE_OUT}"
elif [[ "${PHASE1_MODE}" == "join" ]]; then
  run_locked_module pfn_dag_verify.phase1_join \
    --device cuda --config "${PHASE1_CONFIG}" --panel "${PHASE1_PANEL_DIR}" \
    --pfn "${PHASE1_ATTEMPT_ROOT}/pfn" \
    --oracle "${PHASE1_ATTEMPT_ROOT}/oracle_bank0" \
    --oracle "${PHASE1_ATTEMPT_ROOT}/oracle_bank1" \
    --oracle "${PHASE1_ATTEMPT_ROOT}/oracle_bank2" --out "${PHASE1_STAGE_OUT}"
elif [[ "${PHASE1_MODE}" == "verify" ]]; then
  PHASE1_SOURCE_COMMIT="$(/usr/bin/git -C "${PHASE1_ROOT}" rev-parse HEAD)"
  run_locked_module pfn_dag_verify.phase1_confirmation_verify \
    --root "${PHASE1_ATTEMPT_ROOT}/join" --repo "${PHASE1_ROOT}" \
    --commit "${PHASE1_SOURCE_COMMIT}" \
    --attempt-tag phase1-ordering-confirmation-v1-fix2 \
    --out "${PHASE1_STAGE_OUT}/independent_verification.json"
else
  usage
fi

verify_submission_binding
