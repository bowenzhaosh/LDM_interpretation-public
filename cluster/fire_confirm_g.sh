#!/usr/bin/env bash
# Amendment G — fire the confirmatory fleet. Run from bo-computer AFTER the lock.
#
#   bash cluster/fire_confirm_g.sh wave1     # the 33 gate-bearing + reported jobs
#   bash cluster/fire_confirm_g.sh wave2     # the reported-only arms
#   bash cluster/fire_confirm_g.sh check     # verify the lock and the mirror, submit nothing
#
# Fail-closed before any submission:
#   1. /engrfs and Slurm probed SEPARATELY (both have hung this week).
#   2. the LOCAL AMENDMENT_G.md matches its own sidecar  (catches a post-lock edit);
#   3. the MIRROR's AMENDMENT_G.md is byte-identical to the local, signed one;
#   4. every file AMENDMENT_G.lock.json digests is rsynced and then VERIFIED
#      byte-identical on the mirror. The fleet runs the mirror's copy while the
#      lock digests bo-computer's; without step 4 the confirmatory runs execute
#      code that is not what was sealed. Code ships by rsync, never git pull
#      (.git is 297 MB and the checkout saturates the login node).
#
# Cell convention (G.2): one OUTROOT per (arch, lr, dose),
#   campaigns/mech_20260827/confirm/{arch}[_n{rows}]_lr{lr}_d{dose}/
# so no two arms can write each other's base_s{seed}.pt. Checkpoints _ck0 and
# _ck{steps} only: every compared state is the fully decayed endpoint of its own
# run, never a mid-cosine checkpoint of a longer one.
set -euo pipefail

REPO=/engrfs/project/class/zhao.b/LDM_interpretation
SSH="ssh -o ControlPath=none -o ConnectTimeout=20 -o BatchMode=yes"
HOST=washu1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONF=campaigns/mech_20260827/confirm
MODE="${1:-check}"

# ---------------------------------------------------------------- preflight

echo "[G] probing /engrfs"
timeout 40 $SSH $HOST "timeout 20 stat -c %y $REPO/cluster/mech_train.sbatch" >/dev/null \
  || { echo "REFUSING: /engrfs stalled"; exit 2; }
echo "[G] probing Slurm"
timeout 60 $SSH $HOST "timeout 30 squeue -u \$USER -h | wc -l" >/dev/null \
  || { echo "REFUSING: Slurm not answering"; exit 2; }

echo "[G] checking the lock"
[ -f "$ROOT/AMENDMENT_G.sha256" ] && [ -f "$ROOT/AMENDMENT_G.lock.json" ] \
  || { echo "REFUSING: no local lock — run scripts/lock_amendment_g.py first"; exit 3; }
LREC=$(head -1 "$ROOT/AMENDMENT_G.sha256" | awk '{print $3}')
LACT=$(sha256sum "$ROOT/AMENDMENT_G.md" | awk '{print $1}')
[ -n "$LREC" ] && [ "$LREC" = "$LACT" ] \
  || { echo "REFUSING: local AMENDMENT_G.md was edited after the lock ($LREC vs $LACT)"; exit 3; }

echo "[G] staging the locked code to the mirror"
FILES=$(python3 -c 'import json,sys; print("\n".join(sorted(json.load(open(sys.argv[1]))["digests"])))' \
        "$ROOT/AMENDMENT_G.lock.json")
# shellcheck disable=SC2086
timeout 300 rsync -R -e "$SSH" --checksum $(printf '%s\n' "$FILES" | sed "s|^|$ROOT/./|") \
  "$HOST:$REPO/" >/dev/null \
  || { echo "REFUSING: rsync of the locked code failed"; exit 4; }
timeout 120 rsync -e "$SSH" "$ROOT/AMENDMENT_G.md" "$ROOT/AMENDMENT_G.sha256" \
  "$ROOT/AMENDMENT_G.lock.json" "$HOST:$REPO/" >/dev/null \
  || { echo "REFUSING: rsync of the lock failed"; exit 4; }

echo "[G] verifying the mirror against the lock"
MACT=$(timeout 60 $SSH $HOST "sha256sum $REPO/AMENDMENT_G.md | awk '{print \$1}'")
[ "$MACT" = "$LACT" ] \
  || { echo "REFUSING: mirror AMENDMENT_G.md != the signed local one ($MACT vs $LACT)"; exit 3; }
python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))["digests"]
sys.stdout.write("".join(f"{h}  {p}\n" for p,h in sorted(d.items())))' "$ROOT/AMENDMENT_G.lock.json" \
  | timeout 300 $SSH $HOST "cd $REPO && sha256sum -c --quiet -" \
  || { echo "REFUSING: the mirror's code does not match the digests G.0 registers"; exit 4; }
echo "[G] lock verified $LACT; mirror code matches G.0"

if [ "$MODE" = "check" ]; then echo "[G] check only — nothing submitted"; exit 0; fi

# ---------------------------------------------------------------- submission

# lr -> the tag mech_train.py:57 writes ("{:g}" of the float), fixed for the
# registered grid so the directory name never depends on a shell float format.
lrtag_of() {
  case "$1" in
    3e-3) echo 0.003 ;; 1e-3) echo 0.001 ;; 3e-4) echo 0.0003 ;; 1e-4) echo 0.0001 ;;
    *) echo "unregistered learning rate $1" >&2; exit 65 ;;
  esac
}

# fire ARCH LR NROWS STEPS TIME EPS...
fire() {
  local arch="$1" lr="$2" nrows="$3" steps="$4" tlimit="$5"; shift 5
  local lrtag rowtag outroot e jid etag rc njob
  lrtag=$(lrtag_of "$lr")
  if [ "$nrows" = "20" ]; then rowtag="$arch"; else rowtag="${arch}_n${nrows}"; fi
  outroot="$CONF/${rowtag}_lr${lrtag}_d${steps}"
  for e in "$@"; do
    # Never retrain over an endpoint that exists: a second run of train_pfn saves
    # unconditionally and would replace checkpoints a scored artifact already
    # digested. Refuse and let a human decide what to remove.
    case "$e" in 0.5) etag=0p5 ;; 0.75) etag=0p75 ;; 1.0) etag=1p0 ;;
      *) echo "unregistered eps $e" >&2; exit 66 ;; esac
    # Fail CLOSED: only a clean "absent" (rc 1) proceeds. ssh 255, timeout 124 or a
    # remote shell error must not read as "nothing there" and submit over a trained
    # cell -- /engrfs has stalled twice this week.
    set +e
    timeout 60 $SSH $HOST "test -d $REPO/$outroot/nets/eps$etag"; rc=$?
    set -e
    case "$rc" in
      0) echo "REFUSING: $outroot/nets/eps$etag exists on the mirror — that cell is trained."; exit 6 ;;
      1) : ;;
      *) echo "REFUSING: could not probe $outroot/nets/eps$etag (rc $rc) — not submitting blind."; exit 6 ;;
    esac
    # A row re-submitted while its first jobs are still PENDING would pass the
    # directory probe twice (train_pfn only creates the netdir once it RUNS), so
    # refuse if a job of this exact name is already queued or running.
    set +e
    njob=$(timeout 60 $SSH $HOST "squeue -u \$USER -h -n ldm-G-${rowtag}-lr${lrtag}-d${steps}-e${e} -o %i | wc -l"); rc=$?
    set -e
    [ "$rc" = "0" ] || { echo "REFUSING: could not probe the queue (rc $rc)."; exit 6; }
    [ "$njob" = "0" ] || { echo "REFUSING: ldm-G-${rowtag}-lr${lrtag}-d${steps}-e${e} is already queued/running."; exit 6; }
    jid=$(timeout 120 $SSH $HOST "cd $REPO && mkdir -p $outroot dose_out && \
      sbatch --parsable --time=$tlimit --job-name=ldm-G-${rowtag}-lr${lrtag}-d${steps}-e${e} \
        --export=ALL,EPSVAL=$e,SCALEVAL=$arch,NROWS=$nrows,STEPSVAL=$steps,LRVAL=$lr,SEEDSVAL='3 4 5',CKPTS='0',OUTROOT=$REPO/$outroot \
        cluster/mech_train.sbatch" | tr -d '\r' | grep -E '^[0-9]+' | head -1) \
      || { echo "REFUSING: sbatch failed for ${rowtag} lr=$lrtag d=$steps eps=$e"; exit 5; }
    echo "  submitted ${jid}  ${rowtag} lr=${lrtag} d=${steps} eps=${e}  --time=${tlimit}  -> ${outroot}"
  done
}

wave1() {
  echo "[G] wave 1 — long rows first so the 6-GPU QOS cap is never idle"
  # row 6: base 2M, all eps            (G6 grid arm at .75; dose law at .5/1.0)
  fire base 1e-3   20 2000000 24:00:00 0.5 0.75 1.0
  # row 7: base 2M, the other LRs at eps .75  (G6's base selection)
  fire base 3e-3   20 2000000 24:00:00 0.75
  fire base 3e-4   20 2000000 24:00:00 0.75
  fire base 1e-4   20 2000000 24:00:00 0.75
  # row 5: large 500k, full grid, eps .75 and 1.0   (G5a / G5b / G6)
  fire large 3e-3  20 500000  12:00:00 0.75 1.0
  fire large 1e-3  20 500000  12:00:00 0.75 1.0
  fire large 3e-4  20 500000  12:00:00 0.75 1.0
  fire large 1e-4  20 500000  12:00:00 0.75 1.0
  # row 2: base 500k, registered recipe, all eps    (G1 G2 G3 G4 + grid arm)
  fire base 1e-3   20 500000  08:00:00 0.5 0.75 1.0
  # row 4: base 500k, the other LRs at eps .75 / 1.0  (G5a / G5b base selection)
  fire base 3e-3   20 500000  08:00:00 0.75 1.0
  fire base 3e-4   20 500000  08:00:00 0.75 1.0
  fire base 1e-4   20 500000  08:00:00 0.75 1.0
  # row 3: base n_rows 40, eps .75                  (G3)
  fire base 1e-3   40 500000  08:00:00 0.75
  # row 1: base 100k, all eps                       (G1 G2)
  fire base 1e-3   20 100000  04:00:00 0.5 0.75 1.0
  # row 8: base 10k / 25k, all eps                  (reported: the early/late figure)
  fire base 1e-3   20 10000   02:00:00 0.5 0.75 1.0
  fire base 1e-3   20 25000   02:00:00 0.5 0.75 1.0
}

wave2() {
  echo "[G] wave 2 — reported arms only, no gate depends on them"
  # row 9: base and large grid at eps 0.5. 1e-3 is DELIBERATELY absent from the
  # base loop: wave 1 row 2 already trained base/1e-3/500k at eps 0.5, and that
  # cell carries G1 (both doses), G2 (eps 0.5) and the G4 digest cross-check.
  # Re-firing it would overwrite registered endpoints with exit 0.
  for lr in 3e-3 3e-4 1e-4; do fire base  "$lr" 20 500000  08:00:00 0.5; done
  for lr in 3e-3 1e-3 3e-4 1e-4; do fire large "$lr" 20 500000  12:00:00 0.5; done
  # row 10: the small rung of the ladder, full grid, all eps
  for lr in 3e-3 1e-3 3e-4 1e-4; do fire small "$lr" 20 500000  06:00:00 0.5 0.75 1.0; done
  # row 11: large 2M at the two low LRs, eps .75 (the asymptote)
  fire large 3e-4  20 2000000 48:00:00 0.75
  fire large 1e-4  20 2000000 48:00:00 0.75
}

case "$MODE" in
  wave1) wave1 ;;
  wave2) wave2 ;;
  *) echo "usage: $0 {check|wave1|wave2}"; exit 64 ;;
esac

echo "[G] $MODE submitted. Journal the job IDs in WORKLOG.md now."
timeout 60 $SSH $HOST "squeue -u \$USER -h -o '%i %j %t %L %R' | head -50" \
  | grep -vE 'WARNING|post-quantum|store now|openssh|Loading' || true
