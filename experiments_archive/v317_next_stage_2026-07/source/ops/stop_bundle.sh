#!/bin/bash
# stop_bundle.sh — gracefully stop a running training bundle by agent PID.
# Kills the whole descendant tree (ranks own their pgids, so group-kill on
# the agent misses them). TERM first, KILL after 30s. Exact pids only.
set -u
ROOT_PID="${1:?usage: stop_bundle.sh <AGENT_PID>}"
PIDS=$(ps -eo pid,ppid --no-headers | awk -v r="$ROOT_PID" '
  BEGIN { keep[r] = 1 }
  { ppid[$1] = $2 }
  END {
    changed = 1
    while (changed) {
      changed = 0
      for (pid in ppid) {
        if (ppid[pid] in keep && !(pid in keep)) { keep[pid] = 1; changed = 1 }
      }
    }
    for (pid in keep) print pid
  }' | tr '\n' ' ')
if [ -z "$PIDS" ]; then
  echo "no processes under $ROOT_PID"
  exit 0
fi
echo "TERM: $PIDS"
kill -TERM $PIDS 2>/dev/null
sleep 30
LEFT=$(ps -eo pid,ppid --no-headers | awk -v r="$ROOT_PID" '
  BEGIN { keep[r] = 1 }
  { ppid[$1] = $2 }
  END {
    changed = 1
    while (changed) {
      changed = 0
      for (pid in ppid) {
        if (ppid[pid] in keep && !(pid in keep)) { keep[pid] = 1; changed = 1 }
      }
    }
    for (pid in keep) print pid
  }' | tr '\n' ' ')
[ -n "$LEFT" ] && { echo "KILL: $LEFT"; kill -KILL $LEFT 2>/dev/null; }
echo "stopped $ROOT_PID"
