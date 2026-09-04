#!/usr/bin/env bash
# EXT campaign (2026-09-02, REPORTED, never gated) — fire and score the extension
# rows. Run from bo-computer. Modelled on cluster/fire_confirm_g.sh's fail-closed
# pattern; that file and every G.0-digested file are left untouched.
#
#   bash cluster/fire_ext.sh check                 # probes + all 25 G.0 digests + stage the EXT drivers; submit nothing
#   bash cluster/fire_ext.sh E4|E7|E6|E5|E2|E3|WS|E8   # submit one row set (see .claude/EXT_PLAN_20260902.md, revised 09-03)
#   RESUME=1 bash cluster/fire_ext.sh E4           # same, skipping cells already queued/trained
#   bash cluster/fire_ext.sh score CELL "EPS…" "SEEDS" [KIND]   # one CPU job PER EPS on the gate panel
#
# Cell: campaigns/mech_ext_20260902/train/<arch>[_n<rows>]_lr<lr>_d<steps>[_K<K>][_dim<d>][_ws<seed>]/
# Seeds live in the checkpoint filenames. Every row uses cluster/mech_train_ext.sbatch
# (the locked trainer call with --d/--K forwarded, OMPVAL=1; --world-seed routes to
# scripts/mech_train_ws.py). Recipe (packing benchmark 09-03): 6 seeds per card, two lanes —
#   gpu    : --gres=gpu:a6000:1          (cap gres/gpu=6)
#   shard6 : --gres=shard:a6000:6        (an exclusive card; cap gres/shard=16, Slurm packs shards)
#   shard4 : --gres=shard:a6000:4        (the remaining 4 units, alone on a card while nobody else shards)
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
SSH="ssh -o ControlPath=none -o ConnectTimeout=20 -o BatchMode=yes"
HOST=washu1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CAMP=campaigns/mech_ext_20260902
TRAIN=$CAMP/train
SCORE=$CAMP/predgain
LEDGER="${EXT_LEDGER:-$ROOT/.claude/ext_jobs.tsv}"   # EXT_LEDGER: offline tests only
GATE_PANEL="PANEL_SEED_VAL=770000101,SPLIT_SEED_VAL=880000101,NPH_VAL=1000"
MODE="${1:-check}"
LANES=(shard6 gpu shard4 gpu gpu)      # round-robin lane assignment for new rows
LANE_I=0

# ---------------------------------------------------------------- preflight
echo "[X] probing /engrfs"
timeout 40 $SSH $HOST "timeout 20 stat -c %y $REPO/cluster/mech_train.sbatch" >/dev/null \
  || { echo "REFUSING: /engrfs stalled"; exit 2; }
echo "[X] probing Slurm"
timeout 60 $SSH $HOST "timeout 30 squeue -u \$USER -h | wc -l" >/dev/null \
  || { echo "REFUSING: Slurm not answering"; exit 2; }

NDIG=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["digests"]))' \
  "$ROOT/AMENDMENT_G.lock.json") || { echo "REFUSING: AMENDMENT_G.lock.json is unreadable"; exit 4; }
[ "$NDIG" = "25" ] || { echo "REFUSING: AMENDMENT_G.lock.json lists $NDIG digests, expected all 25 G.0 files"; exit 4; }
echo "[X] verifying all $NDIG G.0-digested files on the mirror"
python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))["digests"]
sys.stdout.write("".join(f"{h}  {p}\n" for p,h in sorted(d.items())))' "$ROOT/AMENDMENT_G.lock.json" \
  | timeout 300 $SSH $HOST "cd $REPO && sha256sum -c --quiet -" \
  || { echo "REFUSING: the mirror's locked code does not match G.0"; exit 4; }

echo "[X] staging the EXT drivers to the mirror"
NEW="cluster/mech_train_ext.sbatch cluster/mech_score_ext.sbatch scripts/mech_predgain_ext.py scripts/mech_train_ws.py scripts/mech_coarse_oracle.py"
# shellcheck disable=SC2086
timeout 120 rsync -R -e "$SSH" --checksum $(for f in $NEW; do printf '%s ' "$ROOT/./$f"; done) "$HOST:$REPO/" >/dev/null \
  || { echo "REFUSING: rsync of the EXT drivers failed"; exit 4; }
(cd "$ROOT" && sha256sum $NEW) | timeout 120 $SSH $HOST "cd $REPO && sha256sum -c --quiet -" \
  || { echo "REFUSING: EXT drivers differ on the mirror after rsync"; exit 4; }
echo "[X] preflight clean"
[ "$MODE" = "check" ] && { echo "[X] check only — nothing submitted"; exit 0; }

# ---------------------------------------------------------------- helpers
lrtag_of() {
  case "$1" in
    3e-3) echo 0.003 ;; 1e-3) echo 0.001 ;; 3e-4) echo 0.0003 ;; 1e-4) echo 0.0001 ;;
    *) echo "unregistered learning rate $1" >&2; exit 65 ;;
  esac
}
etag_of() { "$ROOT/.venv/bin/python" -c 'import sys; sys.path.insert(0,"'"$ROOT/src"'"); from pfn_dag_verify.corrected_verdict import eps_tag; print(eps_tag(float(sys.argv[1])))' "$1"; }
prefix_of() {  # ARCH LR NROWS  -> the net prefix mech_train.py writes
  local p="$1" lt=""
  [ "$2" = "1e-3" ] || lt="_lr$("$ROOT/.venv/bin/python" -c 'import sys; print(format(float(sys.argv[1]), "g"))' "$2")"
  [ "$3" = "20" ] || lt="_n$3$lt"
  echo "$p$lt"
}
cell_of() {  # ARCH LR NROWS STEPS K D WS
  local lrtag rowtag cell
  lrtag=$(lrtag_of "$2")
  if [ "$3" = "20" ]; then rowtag="$1"; else rowtag="${1}_n${3}"; fi
  cell="${rowtag}_lr${lrtag}_d$4"
  [ "$5" = "8" ] || cell="${cell}_K$5"
  [ "$6" = "3" ] || cell="${cell}_dim$6"
  [ "$7" = "-" ] || cell="${cell}_ws$7"
  echo "$cell"
}
seedtag_of() {  # "3 4 5" -> s3-5; a non-contiguous set keeps every seed, so two seed
  local -a s; local i                            # sets of one cell can never share a tag (B1)
  read -r -a s <<< "$1"
  [ "${#s[@]}" -gt 0 ] || { echo "empty seed list" >&2; exit 65; }
  for ((i = 1; i < ${#s[@]}; i++)); do
    if [ $(( ${s[i]} - ${s[i-1]} )) -ne 1 ]; then local IFS=_; echo "s${s[*]}"; return; fi
  done
  echo "s${s[0]}-${s[${#s[@]} - 1]}"
}
nseeds_of() { set -- $1; echo $#; }
lane_next() { LANE=${LANES[$((LANE_I % ${#LANES[@]}))]}; LANE_I=$((LANE_I + 1)); }   # sets $LANE (a $(...) would lose the counter in a subshell)
lane_opts() {  # LANE NSEEDS -> sbatch resource options
  local gres extra cpus mem
  case "$1" in
    gpu)    gres=gpu:a6000:1;   extra="" ;;
    shard6) gres=shard:a6000:6; extra="--exclude=a60-2208" ;;
    shard4) gres=shard:a6000:4; extra="--exclude=a60-2208" ;;
    *) echo "unknown lane $1" >&2; exit 67 ;;
  esac
  if [ "$2" -le 3 ]; then cpus=6; mem=8G; else cpus=12; mem=16G; fi
  echo "--gres=$gres --cpus-per-task=$cpus --mem=$mem $extra"
}

# fire ARCH LR NROWS STEPS TIME "SEEDS" K D WS EPS...
fire() {
  local arch="$1" lr="$2" nrows="$3" steps="$4" tlimit="$5" seeds="$6" K="$7" D="$8" WS="$9"; shift 9
  local cell prefix stag outroot e etag rc njob jid extra first lane ropts ns
  cell=$(cell_of "$arch" "$lr" "$nrows" "$steps" "$K" "$D" "$WS")
  prefix=$(prefix_of "$arch" "$lr" "$nrows")
  stag=$(seedtag_of "$seeds"); first=${seeds%% *}; ns=$(nseeds_of "$seeds")
  outroot="$TRAIN/$cell"
  extra=",DVAL=$D,KVAL=$K,OMPVAL=1"
  [ "$WS" = "-" ] || extra="$extra,WSVAL=$WS"
  for e in "$@"; do
    etag=$(etag_of "$e")
    set +e
    timeout 60 $SSH $HOST "ls $REPO/$outroot/nets/eps$etag/${prefix}_s${first}.pt $REPO/$outroot/nets/eps$etag/${prefix}_s${first}_ck*.pt 2>/dev/null | grep -q ." ; rc=$?
    set -e
    case "$rc" in
      0) if [ "${RESUME:-0}" = "1" ]; then echo "  skip (checkpoint exists) ${cell} eps=${e} seeds='${seeds}'"; continue; fi
         echo "REFUSING: $outroot/nets/eps$etag has seed $first — trained or in progress."; exit 6 ;;
      1) : ;;
      *) echo "REFUSING: could not probe $outroot (rc $rc) — not submitting blind."; exit 6 ;;
    esac
    local jname="ldm-X-${cell}-e${e}-${stag}"
    set +e
    njob=$(timeout 60 $SSH $HOST "squeue -u \$USER -h -n $jname -o %i | wc -l"); rc=$?
    set -e
    [ "$rc" = "0" ] || { echo "REFUSING: could not probe the queue (rc $rc)."; exit 6; }
    if [ "$njob" != "0" ]; then
      if [ "${RESUME:-0}" = "1" ]; then echo "  skip (queued) $jname"; continue; fi
      echo "REFUSING: $jname is already queued/running."; exit 6
    fi
    lane_next; lane=$LANE; ropts=$(lane_opts "$lane" "$ns")
    jid=$(timeout 120 $SSH $HOST "cd $REPO && mkdir -p $outroot dose_out && \
      sbatch --parsable --time=$tlimit $ropts --job-name=$jname \
        --export=ALL,EPSVAL=$e,SCALEVAL=$arch,NROWS=$nrows,STEPSVAL=$steps,LRVAL=$lr,SEEDSVAL='$seeds',CKPTS='0',OUTROOT=$REPO/$outroot$extra \
        cluster/mech_train_ext.sbatch" | tr -d '\r' | grep -E '^[0-9]+' | head -1) \
      || { echo "REFUSING: sbatch failed for $jname"; exit 5; }
    echo "  submitted ${jid}  ${cell} eps=${e} seeds='${seeds}' lane=${lane} --time=${tlimit} -> ${outroot}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$jid" "$MODE" "$cell" "$e" "$seeds" "$tlimit" "$lane" >> "$LEDGER"
  done
}

# score CELL "EPS…" "SEEDS" [KIND] — one CPU job PER EPS; output under <cell>/<seedtag>/
score() {
  local cell="$1" epslist="$2" seeds="$3" kind="${4:-orig}"
  local arch lr nrows steps K D WS scaleval ivdir tlimit jid stag e extra jname njob rc
  arch=${cell%%_*}; K=8; D=3; WS=""; nrows=20
  [[ "$cell" =~ _n([0-9]+)_ ]] && nrows=${BASH_REMATCH[1]}
  [[ "$cell" =~ _lr([0-9.]+)_d([0-9]+) ]] && { lr=${BASH_REMATCH[1]}; steps=${BASH_REMATCH[2]}; }
  [[ "$cell" =~ _K([0-9]+) ]] && K=${BASH_REMATCH[1]}
  [[ "$cell" =~ _dim([0-9]+) ]] && D=${BASH_REMATCH[1]}
  [[ "$cell" =~ _ws([0-9]+) ]] && WS=${BASH_REMATCH[1]}
  case "$lr" in 0.003) lr=3e-3 ;; 0.001) lr=1e-3 ;; 0.0003) lr=3e-4 ;; 0.0001) lr=1e-4 ;; esac
  scaleval=$(prefix_of "$arch" "$lr" "$nrows")
  stag=$(seedtag_of "$seeds")
  ivdir=""; [ "$kind" = "n40ext" ] && ivdir=",IV_DIR_VAL=${IV_DIR:-$CAMP/iv}"
  extra=",DVAL=$D,KVAL=$K"; [ -n "$WS" ] && extra="$extra,WSVAL=$WS"
  if [ "$arch" = "large" ]; then tlimit=04:00:00; else tlimit=02:00:00; fi
  [ "$D" = "4" ] && tlimit=08:00:00
  for e in $epslist; do
    jname="ldm-Xs-${cell}-e${e}-${stag}"
    set +e
    njob=$(timeout 60 $SSH $HOST "squeue -u \$USER -h -n $jname -o %i | wc -l"); rc=$?
    set -e
    [ "$rc" = "0" ] || { echo "REFUSING: could not probe the queue (rc $rc)."; exit 6; }
    [ "$njob" = "0" ] || { echo "REFUSING: $jname is already queued/running."; exit 6; }
    jid=$(timeout 120 $SSH $HOST "cd $REPO && mkdir -p $SCORE/$cell/$stag dose_out && \
      sbatch --parsable --time=$tlimit --job-name=$jname \
        --export=ALL,$GATE_PANEL,SEEDS_VAL='$seeds',KIND_VAL=$kind,SCALE_VAL=$scaleval,NROWS_VAL=20,EPS='$e',STEPS=$steps,NETS_DIR=$TRAIN/$cell/nets,OUT_DIR=$SCORE/$cell/$stag$extra$ivdir \
        cluster/mech_score_ext.sbatch" | tr -d '\r' | grep -E '^[0-9]+' | head -1) \
      || { echo "REFUSING: sbatch failed for score $cell eps $e"; exit 5; }
    echo "  submitted ${jid}  score ${cell} eps=${e} seeds='${seeds}' kind=${kind} --time=${tlimit} -> $SCORE/$cell/$stag"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$jid" "score" "$cell" "$e" "$seeds" "$tlimit" "$kind" >> "$LEDGER"
  done
}

# ---------------------------------------------------------------- rows (revised 09-03 after the critique + packing benchmark)
S6="3 4 5 6 7 8"; S3="3 4 5"
E4() {  # d=4 (24 orderings), LR pair so the selection rule can be applied
  for lr in 1e-3 3e-4; do
    fire base $lr 20 100000  03:00:00 "$S6" 8 4 - 0.5 0.75 1.0
    fire base $lr 20 500000  12:00:00 "$S6" 8 4 - 0.5 0.75 1.0
    fire base $lr 20 2000000 44:00:00 "$S6" 8 4 - 0.5 0.75 1.0
  done
  fire large 3e-4 20 500000 20:00:00 "$S3" 8 4 - 0.75 1.0
}
E7() { for lr in 3e-4 1e-4; do fire large $lr 20 2000000 48:00:00 "$S3" 8 3 - 0.5 1.0; done; }
E6() { fire base 1e-3 20 5000000 60:00:00 "$S3" 8 3 - 0.75 1.0; fire base 3e-4 20 5000000 60:00:00 "$S3" 8 3 - 0.75; }
E5() { fire base 1e-3 40 500000 09:00:00 "$S6" 8 3 - 0.5 1.0; }
E2() {  # eps 0.35 only (0.1 / 0.25 non-evaluable under SE <= 0.15)
  fire base  1e-3 20 100000  02:00:00 "$S6" 8 3 - 0.35
  fire base  1e-3 20 500000  07:00:00 "$S6" 8 3 - 0.35
  fire base  1e-3 20 2000000 26:00:00 "$S6" 8 3 - 0.35
  fire large 3e-4 20 500000  12:00:00 "$S3" 8 3 - 0.35
}
E3() {  # K=2 only: the one K value that moves the order component's share
  fire base 1e-3 20 100000 02:00:00 "$S6" 2 3 - 0.5 0.75 1.0
  fire base 1e-3 20 500000 07:00:00 "$S6" 2 3 - 0.5 0.75 1.0
}
WS() {  # world-seed replication: three fresh atom libraries at K=8, d=3
  for ws in 11 12 13; do
    fire base 1e-3 20 100000  02:00:00 "$S6" 8 3 $ws 0.5 0.75 1.0
    fire base 1e-3 20 500000  07:00:00 "$S6" 8 3 $ws 0.5 0.75 1.0
    fire base 1e-3 20 2000000 26:00:00 "$S3" 8 3 $ws 0.75
  done
}
E8() {  # second tier, fired only when the queue drains
  fire base  1e-3 20 5000000 60:00:00 "$S3" 8 3 - 0.5
  fire large 3e-4 20 2000000 60:00:00 "$S3" 8 4 - 0.75
  for ws in 11 12 13; do fire base 1e-3 20 2000000 26:00:00 "$S3" 8 3 $ws 0.5 1.0; done
  fire base 1e-3 20 2000000 26:00:00 "$S6" 2 3 - 0.5 0.75 1.0
}

case "$MODE" in
  E4|E7|E6|E5|E2|E3|WS|E8) "$MODE" ;;
  score) shift; score "$@" ;;
  *) echo "usage: $0 {check|E4|E7|E6|E5|E2|E3|WS|E8|score CELL \"EPS\" \"SEEDS\" [KIND]}"; exit 64 ;;
esac
echo "[X] $MODE submitted; ledger: $LEDGER — journal the job IDs in WORKLOG.md now."
timeout 60 $SSH $HOST "squeue -u \$USER -h -o '%i %j %t %L %R' | head -80" \
  | grep -vE 'WARNING|post-quantum|store now|openssh|Loading' || true
