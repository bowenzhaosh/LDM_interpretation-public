#!/usr/bin/env bash
# Amendment G — score the confirmatory fleet. Run from bo-computer AFTER the lock,
# once the training endpoints it names exist on the mirror.
#
#   bash cluster/score_confirm_g.sh select    # SELECTION panel, the LR-grid arms only
#   bash cluster/score_confirm_g.sh gate      # GATE panel, every gate-bearing cell
#   bash cluster/score_confirm_g.sh report    # GATE panel, the wave-2 reported cells
#   bash cluster/score_confirm_g.sh readout   # G4: oracles + projections on the gate panel
#   bash cluster/score_confirm_g.sh n40       # build the paired n40 panel (cheap, login node)
#
# G.1 requires the selection table to exist before a gate reads a selected cell,
# so `select` is submitted and completed FIRST. `gate` may be submitted in
# parallel: mech_gates.py reads only the selected LR's gate-panel cell either
# way, and every cell asserts its own panel identity, so an unselected cell that
# happens to be scored is reported and never adjudicated.
#
# Every path is REPO-ROOT relative: mech_score.sbatch cd's to $REPO and passes
# NETS_DIR / OUT_DIR / IV_DIR_VAL through unchanged, so a value must carry the
# campaigns/mech_20260827/ prefix itself.
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
SSH="ssh -o ControlPath=none -o ConnectTimeout=20 -o BatchMode=yes"
HOST=washu1
PY=/engrfs/project/class/zhao.b/conda_envs/tidpo/bin/python
CAMP=campaigns/mech_20260827
CONF=$CAMP/confirm
MODE="${1:-}"

# The registered panels (G.2). CONFIRM_VAL=1 forbids a silent fallback to either.
GATE_PANEL="PANEL_SEED_VAL=770000101,SPLIT_SEED_VAL=880000101,NPH_VAL=1000"
SEL_PANEL="PANEL_SEED_VAL=770000201,SPLIT_SEED_VAL=880000201,NPH_VAL=500"

echo "[G] probing /engrfs"
timeout 40 $SSH $HOST "timeout 20 stat -c %y $REPO/cluster/mech_score.sbatch" >/dev/null \
  || { echo "REFUSING: /engrfs stalled"; exit 2; }
timeout 60 $SSH $HOST "test -f $REPO/AMENDMENT_G.sha256" \
  || { echo "REFUSING: AMENDMENT_G.sha256 absent on the mirror — not locked"; exit 3; }

lrtag_of() {
  case "$1" in
    3e-3) echo 0.003 ;; 1e-3) echo 0.001 ;; 3e-4) echo 0.0003 ;; 1e-4) echo 0.0001 ;;
    *) echo "unregistered learning rate $1" >&2; exit 65 ;;
  esac
}

# score PANEL_KIND ARCH LR NROWS DOSE EPSLIST
# PANEL_KIND is "gate" or "select"; it fixes both the panel seeds and the output root.
# Wall time is set per row, not by mech_score.sbatch's 2 h header: a `large` cell
# runs a 2.1M-parameter model on CPU over 1000 contexts x 3 seeds x |eps|, and a
# timeout mid-job leaves the finished eps files, which a re-run under
# MECH_CONFIRM=1 then refuses -- an unrecoverable state the pinned command has no
# way out of.
score() {
  local panelkind="$1" arch="$2" lr="$3" nrows="$4" dose="$5" epslist="$6"
  local lrtag rowtag scaleval cell panel outroot kind ivdir jid tlimit
  lrtag=$(lrtag_of "$lr")
  if [ "$nrows" = "20" ]; then rowtag="$arch"; else rowtag="${arch}_n${nrows}"; fi
  # MECH_SCALE is the net prefix mech_train.py wrote: no lr tag at the registered 1e-3.
  if [ "$lr" = "1e-3" ]; then scaleval="$rowtag"; else scaleval="${rowtag}_lr${lrtag}"; fi
  cell="${rowtag}_lr${lrtag}_d${dose}"
  if [ "$panelkind" = "select" ]; then panel="$SEL_PANEL"; outroot="$CAMP/predgain_select/$cell"
  else                                 panel="$GATE_PANEL"; outroot="$CAMP/predgain_confirm/$cell"; fi
  # the n_rows-40 arm is scored as the n20 panel's paired EXTENSION (G3)
  if [ "$nrows" = "40" ]; then
    kind="n40ext"; ivdir=",IV_DIR_VAL=$CONF/iv"
    outroot="$CAMP/predgain_confirm_n40/$cell"
  else
    kind="orig"; ivdir=""
  fi
  if [ "$arch" = "large" ]; then tlimit=12:00:00; else tlimit=06:00:00; fi
  jid=$(timeout 120 $SSH $HOST "cd $REPO && mkdir -p $outroot dose_out && \
    sbatch --parsable --time=$tlimit --job-name=ldm-Gs-${panelkind}-${cell} \
      --export=ALL,CONFIRM_VAL=1,$panel,LADDER_VAL=0,SEEDS_VAL='3 4 5',KIND_VAL=$kind,SCALE_VAL=$scaleval,NROWS_VAL=20,EPS='$epslist',STEPS=$dose,NETS_DIR=$CONF/$cell/nets,OUT_DIR=$outroot$ivdir \
      cluster/mech_score.sbatch" | tr -d '\r' | grep -E '^[0-9]+' | head -1) \
    || { echo "REFUSING: sbatch failed for $panelkind $cell"; exit 5; }
  echo "  submitted ${jid}  ${panelkind}  ${cell}  eps='${epslist}'  scale=${scaleval}  --time=${tlimit} -> ${outroot}"
}

case "$MODE" in
select)
  # Every LR-grid arm on the SELECTION panel. No gate reads these artifacts;
  # they decide LR*(arch, dose, eps) by the registered argmin rule (G.1).
  for lr in 3e-3 1e-3 3e-4 1e-4; do
    score select base  "$lr" 20 500000  "0.75 1.0"
    score select large "$lr" 20 500000  "0.75 1.0"
    score select base  "$lr" 20 2000000 "0.75"
  done
  ;;
gate)
  # registered base arm (G1 G2 G3 G4) and the dose law
  score gate base 1e-3 20 100000  "0.5 0.75 1.0"
  score gate base 1e-3 20 500000  "0.5 0.75 1.0"
  score gate base 1e-3 20 2000000 "0.5 0.75 1.0"
  score gate base 1e-3 20 10000   "0.5 0.75 1.0"
  score gate base 1e-3 20 25000   "0.5 0.75 1.0"
  # G3's paired n40 extension
  score gate base 1e-3 40 500000  "0.75"
  # the LR-grid arms on the GATE panel (G5a G5b G6 read only the selected one)
  for lr in 3e-3 3e-4 1e-4; do
    score gate base  "$lr" 20 500000  "0.75 1.0"
    score gate base  "$lr" 20 2000000 "0.75"
  done
  for lr in 3e-3 1e-3 3e-4 1e-4; do
    score gate large "$lr" 20 500000  "0.75 1.0"
  done
  ;;
report)
  # The wave-2 cells (G.2 rows 9-11). NO gate reads any of these; they exist so
  # the reported ladder and the eps-0.5 grid have scored artifacts on the same
  # gate panel, and the evaluator will refuse them anywhere a gate is computed.
  for lr in 3e-3 3e-4 1e-4; do score gate base  "$lr" 20 500000 "0.5"; done
  for lr in 3e-3 1e-3 3e-4 1e-4; do score gate large "$lr" 20 500000 "0.5"; done
  for lr in 3e-3 1e-3 3e-4 1e-4; do score gate small "$lr" 20 500000 "0.5 0.75 1.0"; done
  score gate large 3e-4 20 2000000 "0.75"
  score gate large 1e-4 20 2000000 "0.75"
  ;;
readout)
  # G4: the Amendment F projection readout, on the registered gate panel and the
  # registered 500k fleet. CPU-heavy over 1000 contexts -> a batch job, never the
  # login node.
  jid=$(timeout 120 $SSH $HOST "cd $REPO && mkdir -p $CONF/phase1 dose_out && \
    sbatch --parsable --job-name=ldm-G-readout cluster/mech_readout_g.sbatch" \
    | tr -d '\r' | grep -E '^[0-9]+' | head -1) \
    || { echo "REFUSING: sbatch failed for the G4 readout"; exit 5; }
  echo "  submitted ${jid}  G4 oracles + projections -> $CONF/phase1"
  ;;
n40)
  # G3's paired panel: the n20 contexts extended by 20 rows from their own latent.
  # Cheap (numpy RNG replay, no exact posterior), so the login node is fine.
  timeout 600 $SSH $HOST "cd $REPO && MECH_CONFIRM=1 MECH_PANEL_SEED=770000101 \
    MECH_SPLIT_SEED=880000101 MECH_N_PER_HALF=1000 MECH_PHASE1_OUT=$CONF/phase1 \
    MECH_IV_DIR=$CONF/iv PYTHONPATH=$REPO/src $PY scripts/mech_interventions.py build_n40 --eps 0.75"
  ;;
*)
  echo "usage: $0 {select|gate|report|readout|n40}"; exit 64 ;;
esac

echo "[G] $MODE submitted. Journal the job IDs in WORKLOG.md now."
