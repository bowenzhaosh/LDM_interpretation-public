#!/usr/bin/env bash
# Offline test for the EXT drivers. No cluster, no GPU, nothing submitted:
# ssh / rsync / sbatch / squeue are PATH-shadowed by stubs and every remote path is
# rewritten into a throwaway "mirror" under $TMPDIR.
#
#   bash cluster/tests/test_ext_drivers.sh
#
# Covers the 2026-09-03 FIX list (WORKLOG 01:55):
#   T1 syntax of all four drivers
#   T2 fire_ext.sh helpers (cell/prefix/seed tags, lane round-robin)
#   T3 preflight verifies ALL 25 G.0 digests and refuses on any one of them
#   T4 score(): one sbatch PER EPS, seed-tagged OUT_DIR and job name, dup-queue probe
#   T5 fire(): OMPVAL threaded through to mech_train_ext.sbatch
#   T6 the registered-campaign guard in both sbatch files (absolute, relative, bare
#      directory, '..' traversal)
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_REMOTE=/engrfs/project/class/zhao.b/LDM_interpretation
FIRE=${FIRE_EXT:-$ROOT/cluster/fire_ext.sh}          # FIRE_EXT: test a candidate before swapping it in
TD=$(mktemp -d "${TMPDIR:-/tmp}/extdrv.XXXXXX")
trap 'rm -rf "$TD"' EXIT
MIRROR=$TD/mirror; BIN=$TD/bin; mkdir -p "$MIRROR" "$BIN"
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }
is()   { [ "$2" = "$3" ] && ok "$1" || bad "$1" "expected [$3] got [$2]"; }
has()  { case "$2" in *"$3"*) ok "$1" ;; *) bad "$1" "missing [$3] in: ${2:0:400}" ;; esac; }
hasnt(){ case "$2" in *"$3"*) bad "$1" "unexpected [$3]" ;; *) ok "$1" ;; esac; }

# ------------------------------------------------------------------ T1 syntax
echo "T1 syntax"
for f in "$FIRE" "$ROOT/cluster/mech_train_ext.sbatch" "$ROOT/cluster/mech_score_ext.sbatch"; do
  out=$(bash -n "$f" 2>&1); is "bash -n ${f#$ROOT/}" "$?" "0"; [ -z "$out" ] || echo "     $out"
done
"$ROOT/.venv/bin/python" -m py_compile "$ROOT/scripts/mech_predgain_ext.py" 2>&1
is "py_compile scripts/mech_predgain_ext.py" "$?" "0"

# ------------------------------------------------------------------ T2 helpers
echo "T2 fire_ext.sh helpers"
sed -n '/^# -*  *helpers/,/^# fire ARCH/p' "$FIRE" > "$TD/helpers.sh"
grep -q 'seedtag_of' "$TD/helpers.sh" && ok "helper block extracted" || bad "helper block extracted" "sed range missed"
# shellcheck disable=SC1090
LANES=(shard6 gpu shard4 gpu gpu); LANE_I=0; source "$TD/helpers.sh"
is "cell_of base 1e-3 20 500000 8 3 -"        "$(cell_of base 1e-3 20 500000 8 3 -)"      "base_lr0.001_d500000"
is "cell_of base 1e-3 40 500000 8 3 - (n40)"  "$(cell_of base 1e-3 40 500000 8 3 -)"      "base_n40_lr0.001_d500000"
is "cell_of base 1e-3 20 100000 2 4 11 (K,d,ws)" "$(cell_of base 1e-3 20 100000 2 4 11)"  "base_lr0.001_d100000_K2_dim4_ws11"
is "prefix_of base 1e-3 20"                   "$(prefix_of base 1e-3 20)"                 "base"
is "prefix_of large 3e-4 40"                  "$(prefix_of large 3e-4 40)"                "large_n40_lr0.0003"
is "seedtag_of '3 4 5'"                       "$(seedtag_of '3 4 5')"                     "s3-5"
is "seedtag_of '6 7 8 9 10 11'"               "$(seedtag_of '6 7 8 9 10 11')"             "s6-11"
is "seedtag_of '3 4 5 6 7 8'"                 "$(seedtag_of '3 4 5 6 7 8')"               "s3-8"
# two DIFFERENT seed sets of one cell must never share a tag (B1)
a=$(seedtag_of '3 5 7'); b=$(seedtag_of '3 4 5 6 7')
[ "$a" != "$b" ] && ok "non-contiguous seed sets get distinct tags ($a vs $b)" \
                 || bad "non-contiguous seed sets get distinct tags" "both [$a]"
is "nseeds_of '3 4 5 6 7 8'"                  "$(nseeds_of '3 4 5 6 7 8')"                "6"
lanes=""; for _ in 1 2 3 4 5 6; do lane_next; lanes="$lanes$LANE "; done
is "lane_next round-robins (keeps its counter)" "$lanes" "shard6 gpu shard4 gpu gpu shard6 "
is "lane_opts shard6 6" "$(lane_opts shard6 6)" "--gres=shard:a6000:6 --cpus-per-task=12 --mem=16G --exclude=a60-2208"
is "lane_opts gpu 3"    "$(lane_opts gpu 3)"    "--gres=gpu:a6000:1 --cpus-per-task=6 --mem=8G "

# ------------------------------------------------------------------ stubs
cat > "$BIN/ssh" <<'EOS'
#!/usr/bin/env bash
args=(); while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) args+=("$1"); shift ;; esac; done
host=${args[0]}; cmd="${args[*]:1}"
cmd=${cmd//$REPO_REMOTE/$MIRROR}
printf 'SSH %s :: %s\n' "$host" "$cmd" >> "$STUB_LOG"
cd "$MIRROR" || exit 3
bash -c "$cmd"
EOS
cat > "$BIN/rsync" <<'EOS'
#!/usr/bin/env bash
srcs=(); while [ $# -gt 0 ]; do case "$1" in -e) shift 2 ;; -*) shift ;; *) srcs+=("$1"); shift ;; esac; done
unset 'srcs[${#srcs[@]}-1]'                    # drop HOST:DEST
for s in "${srcs[@]}"; do rel=${s#*/./}; mkdir -p "$MIRROR/$(dirname "$rel")"; cp "$s" "$MIRROR/$rel"; done
printf 'RSYNC %s\n' "${srcs[*]}" >> "$STUB_LOG"
EOS
cat > "$BIN/sbatch" <<'EOS'
#!/usr/bin/env bash
n=$(( $(cat "$STUB_JID" 2>/dev/null || echo 900000) + 1 )); echo "$n" > "$STUB_JID"
printf 'SBATCH %s\n' "$*" >> "$STUB_LOG"; echo "$n"
EOS
printf '#!/usr/bin/env bash\nexit 0\n' > "$BIN/squeue"        # no job ever queued
chmod +x "$BIN"/*
export STUB_LOG=$TD/stub.log STUB_JID=$TD/jid REPO_REMOTE MIRROR
export PATH="$BIN:$PATH"
# nothing may reach the cluster: ssh MUST resolve to the stub or the run aborts
[ "$(command -v ssh)" = "$BIN/ssh" ] || { echo "ABORT: ssh does not resolve to the stub"; exit 9; }
[ "$(command -v sbatch)" = "$BIN/sbatch" ] || { echo "ABORT: sbatch does not resolve to the stub"; exit 9; }
# a faithful mirror of the 25 G.0-locked files (real content) + a stat target
mkdir -p "$MIRROR/cluster" "$MIRROR/scripts" "$MIRROR/src/pfn_dag_verify"
"$ROOT/.venv/bin/python" - "$ROOT" "$MIRROR" <<'EOP'
import json, shutil, sys
from pathlib import Path
root, mirror = Path(sys.argv[1]), Path(sys.argv[2])
for p in json.loads((root / "AMENDMENT_G.lock.json").read_text())["digests"]:
    (mirror / p).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root / p, mirror / p)
EOP

RC=0; OUT=""
run_fire() {  # MODE ARGS... -> $OUT (stdout+stderr) and $RC. Never a $(...) — a
  : > "$STUB_LOG"; rm -f "$STUB_JID"          # subshell would lose both, as lane_next once did.
  ( cd "$TD" && EXT_LEDGER=$TD/ledger.tsv bash "$FIRE" "$@" ) > "$TD/out.txt" 2>&1
  RC=$?; OUT=$(cat "$TD/out.txt")
}

# ------------------------------------------------------------------ T3 preflight
echo "T3 preflight: all 25 G.0 digests"
run_fire check; is "check exits 0 against a clean mirror" "$RC" "0"
has "check reports the digest count" "$OUT" "25"
hasnt "check submitted nothing" "$(cat "$STUB_LOG")" "SBATCH"
n=$(grep -c 'sha256sum -c' "$STUB_LOG"); [ "$n" -ge 1 ] && ok "preflight ran sha256sum -c on the mirror" || bad "preflight ran sha256sum -c" "n=$n"
# corrupt one of the 16 files the OLD 'keep' filter never checked (feasibility M1)
cp "$MIRROR/src/pfn_dag_verify/corrected_sem.py" "$TD/sem.bak"
echo "# drift" >> "$MIRROR/src/pfn_dag_verify/corrected_sem.py"
run_fire check; is "check REFUSES on drift in corrected_sem.py" "$RC" "4"
has "refusal names the locked code" "$OUT" "REFUSING: the mirror's locked code does not match G.0"
cp "$TD/sem.bak" "$MIRROR/src/pfn_dag_verify/corrected_sem.py"
run_fire check; is "check exits 0 again once restored" "$RC" "0"

# ------------------------------------------------------------------ T4 score
echo "T4 score(): one job per eps, seed-tagged output"
run_fire score base_lr0.001_d500000 "0.5 0.75 1.0" "3 4 5 6 7 8"
is "score rc" "$RC" "0"
n=$(grep -c '^SBATCH' "$STUB_LOG"); is "one sbatch PER EPS (3 eps)" "$n" "3"
sb=$(grep '^SBATCH' "$STUB_LOG")
has "OUT_DIR carries the seed tag"  "$sb" "OUT_DIR=campaigns/mech_ext_20260902/predgain/base_lr0.001_d500000/s3-8"
has "job name carries eps and seed tag" "$sb" "--job-name=ldm-Xs-base_lr0.001_d500000-e0.5-s3-8"
has "each eps is its own job (eps 1.0)"  "$sb" "EPS=1.0,"
hasnt "no multi-eps job"                 "$sb" "EPS=0.5 0.75 1.0"
has "gate panel pinned"             "$sb" "PANEL_SEED_VAL=770000101,SPLIT_SEED_VAL=880000101,NPH_VAL=1000"
is "ledger got 3 rows" "$(grep -c '' "$TD/ledger.tsv")" "3"
# a second seed triple of the same cell must not collide
: > "$TD/ledger.tsv"
run_fire score base_lr0.001_d500000 "1.0" "9 10 11"
has "a different seed triple writes elsewhere" "$(grep '^SBATCH' "$STUB_LOG")" "/base_lr0.001_d500000/s9-11"
# the world axes reach the scorer
run_fire score base_lr0.001_d100000_K2_dim4_ws11 "0.75" "3 4 5"
sb=$(grep '^SBATCH' "$STUB_LOG")
has "K forwarded"  "$sb" "KVAL=2"; has "d forwarded" "$sb" "DVAL=4"; has "world seed forwarded" "$sb" "WSVAL=11"
# duplicate-queue probe: make squeue report one job
printf '#!/usr/bin/env bash\necho 999999\n' > "$BIN/squeue"; chmod +x "$BIN/squeue"
run_fire score base_lr0.001_d500000 "0.5" "3 4 5"
is "score REFUSES when the job name is already queued" "$RC" "6"
hasnt "nothing submitted on the refusal" "$(cat "$STUB_LOG")" "SBATCH"
printf '#!/usr/bin/env bash\nexit 0\n' > "$BIN/squeue"; chmod +x "$BIN/squeue"

# ------------------------------------------------------------------ T5 fire
echo "T5 fire(): OMPVAL threaded through"
run_fire E3
is "E3 rc" "$RC" "0"
sb=$(grep '^SBATCH' "$STUB_LOG")
has "OMPVAL=1 exported to the trainer" "$sb" "OMPVAL=1"
has "trainer is mech_train_ext.sbatch"  "$sb" "cluster/mech_train_ext.sbatch"
has "OUTROOT is under the ext campaign" "$sb" "OUTROOT=$MIRROR/campaigns/mech_ext_20260902/train/"
hasnt "OUTROOT is never the registered campaign" "$sb" "campaigns/mech_20260827"
n=$(grep -c '^SBATCH' "$STUB_LOG"); is "E3 = 2 cells x 3 eps" "$n" "6"

# ------------------------------------------------------------------ T6 guards
echo "T6 registered-campaign guard (both sbatch files)"
guard() {  # FILE VARNAME PATH EXPECT(refuse|accept)
  local f=$1 var=$2 p=$3 want=$4 rc out got
  out=$(cd "$TD" && env -u OUTROOT -u MECH_OUT "$var=$p" EPSVAL=1.0 STEPS=1 EPS=1.0 \
        NETS_DIR=n OUT_DIR="${p#/}" PANEL_SEED_VAL=1 SPLIT_SEED_VAL=1 NPH_VAL=1 \
        bash "$ROOT/cluster/$f" 2>&1); rc=$?
  case "$out" in *REFUSING*) got=refuse ;; *) got=accept ;; esac
  if [ "$got" = "$want" ] && { [ "$want" = accept ] || [ "$rc" = 7 ]; }
    then ok "$f $var $p -> $want"; else bad "$f $var $p -> $want" "got $got rc=$rc: ${out:0:200}"; fi
}
for p in "$REPO_REMOTE/campaigns/mech_20260827/x" \
         "$REPO_REMOTE/campaigns/mech_20260827" \
         "$REPO_REMOTE/campaigns/mech_ext_20260902/../mech_20260827/x" \
         "campaigns/mech_20260827/x"; do
  guard mech_train_ext.sbatch OUTROOT "$p" refuse
done
guard mech_train_ext.sbatch OUTROOT "$REPO_REMOTE/campaigns/mech_ext_20260902/train/c" accept
# mech_score_ext builds MECH_OUT from OUT_DIR, so drive it through OUT_DIR
score_guard() { local d=$1 want=$2 out rc got
  out=$(cd "$TD" && env STEPS=1 EPS=1.0 NETS_DIR=n OUT_DIR="$d" PANEL_SEED_VAL=1 SPLIT_SEED_VAL=1 NPH_VAL=1 \
        bash "$ROOT/cluster/mech_score_ext.sbatch" 2>&1); rc=$?
  case "$out" in *REFUSING*) got=refuse ;; *) got=accept ;; esac
  if [ "$got" = "$want" ] && { [ "$want" = accept ] || [ "$rc" = 7 ]; }
    then ok "mech_score_ext OUT_DIR $d -> $want"; else bad "mech_score_ext OUT_DIR $d -> $want" "got $got rc=$rc: ${out:0:200}"; fi
}
score_guard campaigns/mech_20260827/predgain_confirm/c refuse
score_guard campaigns/mech_20260827/phase1 refuse
score_guard campaigns/mech_ext_20260902/../mech_20260827/x refuse

echo
printf '%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" = 0 ]
