#!/usr/bin/env bash
# Internal arms (PRESPEC_internal) on the cluster's CPU `debug` partition (36 cores, 4 h, no QOS cap).
#   bash cluster/fire_int.sh stage     # rsync scripts + prespec + the small inputs to the mirror, verify
#   bash cluster/fire_int.sh fire      # submit: builds -> acts -> fits (afterok), and Arm C runs + analyses
#   bash cluster/fire_int.sh pull      # pull acts, patch runs, reported JSONs back (never the 170 MB train sets)
# Registered inputs already on the mirror: predgain_confirm/*, confirm/*/nets/*, mech_ext_20260902/coarse/*.
set -euo pipefail
REPO=/engrfs/project/class/zhao.b/LDM_interpretation
SSH="ssh -o ControlPath=none -o ConnectTimeout=20 -o BatchMode=yes"
HOST=washu1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
INT=campaigns/mech_int_20260905
MODE="${1:-stage}"
LEDGER="$ROOT/.claude/int_jobs.tsv"

FILES="scripts/mech_probe_acts.py scripts/mech_probe_fit.py scripts/mech_patch_build.py scripts/mech_patch_run.py scripts/mech_patch_analyse.py scripts/mech_k2_verdict.py scripts/mech_component_control.py scripts/mech_coarse_oracle.py scripts/mech_ext_report.py $INT/PRESPEC_internal.md $INT/PRESPEC_internal.digest $INT/PRESPEC_internal.errata.md campaigns/mech_ext_20260902/reported/component_control.json"

stage() {
  echo "[I] staging"
  # shellcheck disable=SC2086
  timeout 300 rsync -R -e "$SSH" --checksum $(for f in $FILES; do printf '%s ' "$ROOT/./$f"; done) "$HOST:$REPO/" >/dev/null
  (cd "$ROOT" && sha256sum $FILES) | timeout 120 $SSH $HOST "cd $REPO && sha256sum -c --quiet -" || { echo "REFUSING: staged files differ"; exit 4; }
  timeout 60 $SSH $HOST "cd $REPO && sha256sum -c --quiet $INT/PRESPEC_internal.digest && mkdir -p $INT/probes $INT/patch $INT/reported dose_out && ls campaigns/mech_ext_20260902/coarse/coarse_eps0p75.npz campaigns/mech_20260827/confirm/base_lr0.001_d500000/nets/eps0p75/base_s3_ck0.pt >/dev/null" \
    || { echo "REFUSING: prespec digest or registered inputs missing on the mirror"; exit 4; }
  echo "[I] staged and verified"
}

sub() {  # NAME TIME DEPS(or -) CMD...
  local name="$1" tl="$2" dep="$3"; shift 3
  local depopt=""; [ "$dep" != "-" ] && depopt="--dependency=afterok:$dep"
  local jid
  jid=$(timeout 120 $SSH $HOST "cd $REPO && sbatch --parsable -A engr-class-any -p debug --cpus-per-task=36 --mem=100G --time=$tl $depopt --job-name=$name --output=dose_out/%x-%j.out --wrap='set -e; export PYTHONPATH=$REPO/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; cd $REPO; $*; echo $name-DONE' 2>/dev/null" | tr -d '\r' | grep -E '^[0-9]+' | head -1)
  echo "  submitted $jid  $name  ($tl, dep ${dep})"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$jid" "$name" "$dep" >> "$LEDGER"
  echo "$jid"
}

fire() {
  local jb ja jc jd
  # A: training sets, one job per eps (5 draws each, 34 workers)
  local A=""
  for e in 0.75 1.0 0.5; do
    jb=$(sub int-build-$e 03:30:00 - "for ts in 0 1 2 3 4; do $PY scripts/mech_probe_acts.py build --eps $e --train-seed \$ts --jobs 34; done" | tail -1)
    A="${A:+$A:}$jb"
  done
  # B: activations (24 invocations, torch 8 threads)
  ja=$(sub int-acts 03:30:00 - "export OMP_NUM_THREADS=8; for e in 0.75 1.0 0.5; do for s in 3 4 5; do $PY scripts/mech_probe_acts.py acts --eps \$e --seed \$s; $PY scripts/mech_probe_acts.py acts --eps \$e --seed \$s --cell base_lr0.001_d100000 --step 100000 --scored campaigns/mech_20260827/predgain_confirm/base_lr0.001_d100000 --nets campaigns/mech_20260827/confirm/base_lr0.001_d100000/nets; done; done; for e in 0.75 1.0; do for s in 3 4 5; do $PY scripts/mech_probe_acts.py acts --eps \$e --seed \$s --cell large_lr0.0003_d500000 --scored campaigns/mech_20260827/predgain_confirm/large_lr0.0003_d500000 --nets campaigns/mech_20260827/confirm/large_lr0.0003_d500000/nets; done; done" | tail -1)
  # C: fits, one job per (cell, eps), after A and B
  for e in 0.75 1.0 0.5; do
    sub int-fit-base500k-$e 03:50:00 "$A:$ja" "export OMP_NUM_THREADS=16; $PY scripts/mech_probe_fit.py --eps $e" >/dev/null
    sub int-fit-base100k-$e 03:50:00 "$A:$ja" "export OMP_NUM_THREADS=16; $PY scripts/mech_probe_fit.py --eps $e --cell base_lr0.001_d100000 --step 100000 --scored campaigns/mech_20260827/predgain_confirm/base_lr0.001_d100000 --nets campaigns/mech_20260827/confirm/base_lr0.001_d100000/nets" >/dev/null
  done
  for e in 0.75 1.0; do
    sub int-fit-large500k-$e 03:50:00 "$A:$ja" "export OMP_NUM_THREADS=16; $PY scripts/mech_probe_fit.py --eps $e --cell large_lr0.0003_d500000 --scored campaigns/mech_20260827/predgain_confirm/large_lr0.0003_d500000 --nets campaigns/mech_20260827/confirm/large_lr0.0003_d500000/nets" >/dev/null
  done
  # D: Arm C — builds (3 eps), runs (3 eps x 3 seeds), analyses
  sub int-armC 03:50:00 - "for e in 0.75 1.0 0.5; do $PY scripts/mech_patch_build.py --eps \$e --jobs 34; for s in 3 4 5; do $PY scripts/mech_patch_run.py --eps \$e --seed \$s --jobs 34; done; $PY scripts/mech_patch_analyse.py --eps \$e; done" >/dev/null
  echo "[I] fired; ledger $LEDGER"
}

fits() {   # the 8 fit jobs only (builds + acts already on the mirror)
  for e in 0.75 1.0 0.5; do
    sub int-fit-base500k-$e 03:50:00 - "export OMP_NUM_THREADS=16; $PY scripts/mech_probe_fit.py --eps $e" >/dev/null
    sub int-fit-base100k-$e 03:50:00 - "export OMP_NUM_THREADS=16; $PY scripts/mech_probe_fit.py --eps $e --cell base_lr0.001_d100000 --step 100000 --scored campaigns/mech_20260827/predgain_confirm/base_lr0.001_d100000 --nets campaigns/mech_20260827/confirm/base_lr0.001_d100000/nets" >/dev/null
  done
  for e in 0.75 1.0; do
    sub int-fit-large500k-$e 03:50:00 - "export OMP_NUM_THREADS=16; $PY scripts/mech_probe_fit.py --eps $e --cell large_lr0.0003_d500000 --scored campaigns/mech_20260827/predgain_confirm/large_lr0.0003_d500000 --nets campaigns/mech_20260827/confirm/large_lr0.0003_d500000/nets" >/dev/null
  done
}

armc_runs() {   # Arm C builds + runs for the given eps list (analyses submitted separately, afterok)
  for e in "$@"; do
    local jr
    jr=$(sub int-armC-run-$e 02:00:00 - "$PY scripts/mech_patch_build.py --eps $e --jobs 34 || true; for s in 3 4 5; do $PY scripts/mech_patch_run.py --eps $e --seed \$s --jobs 34 || true; done" | tail -1)
    sub int-armC-an-$e 03:50:00 "$jr" "export OMP_NUM_THREADS=8; $PY scripts/mech_patch_analyse.py --eps $e" >/dev/null
  done
}

pull() {
  mkdir -p "$ROOT/$INT/probes" "$ROOT/$INT/patch" "$ROOT/$INT/reported"
  rsync -r -e "$SSH" --checksum --ignore-existing --include='acts_*.npz' --exclude='*' "$HOST:$REPO/$INT/probes/" "$ROOT/$INT/probes/" 2>/dev/null
  rsync -r -e "$SSH" --checksum --ignore-existing --include='run_*.npz' --include='build_*.npz' --exclude='*' "$HOST:$REPO/$INT/patch/" "$ROOT/$INT/patch/" 2>/dev/null
  rsync -r -e "$SSH" --checksum "$HOST:$REPO/$INT/reported/" "$ROOT/$INT/reported/" 2>/dev/null
  ls "$ROOT/$INT/reported/"
}

case "$MODE" in stage) stage ;; fire) stage; fire ;; fits) stage; fits ;; armc) stage; shift; armc_runs "$@" ;; pull) pull ;; *) echo "usage: $0 {stage|fire|fits|armc EPS...|pull}"; exit 64 ;; esac
