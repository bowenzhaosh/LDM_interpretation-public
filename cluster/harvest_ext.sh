#!/usr/bin/env bash
# EXT harvester: every INTERVAL seconds, score every (cell, seeds) group whose training
# jobs are all COMPLETED and which has no score row yet (scripts/ext_ready.py decides
# from the ledger + sacct). Runs on bo-computer; stops on .claude/harvest.stop or after
# STOP_DATE. One line per action in .claude/harvest.log.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
INTERVAL="${INTERVAL:-1800}"
STOP_DATE="${STOP_DATE:-2026-09-23}"
LOG="$ROOT/.claude/harvest.log"
cd "$ROOT"
while true; do
  [ -f .claude/harvest.stop ] && { echo "$(date -u +%FT%TZ) stop file present — exiting" >> "$LOG"; exit 0; }
  [[ "$(date -u +%F)" > "$STOP_DATE" ]] && { echo "$(date -u +%FT%TZ) past $STOP_DATE — exiting" >> "$LOG"; exit 0; }
  ids=$(awk -F'\t' '$3!="score"{print $2}' .claude/ext_jobs.tsv | paste -sd,)
  ready=$(ssh -o BatchMode=yes -o ConnectTimeout=30 washu "sacct -j $ids -X -n -o JobID,State -P" 2>/dev/null \
          | .venv/bin/python scripts/ext_ready.py 2>>"$LOG")
  if [ -n "$ready" ]; then
    while IFS=$'\t' read -r cell eps seeds kind; do
      [ -z "$cell" ] && continue
      echo "$(date -u +%FT%TZ) scoring $cell eps='$eps' seeds='$seeds' kind=$kind" >> "$LOG"
      bash cluster/fire_ext.sh score "$cell" "$eps" "$seeds" "$kind" 2>&1 \
        | grep -E 'submitted|REFUS' >> "$LOG"
    done <<< "$ready"
  else
    echo "$(date -u +%FT%TZ) nothing ready" >> "$LOG"
  fi
  sleep "$INTERVAL"
done
