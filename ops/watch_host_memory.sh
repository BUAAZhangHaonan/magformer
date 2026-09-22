#!/bin/bash
# watch_host_memory.sh — host memory watchdog for one 4-rank g2 bundle (GPUs 4-7).
#
# Usage: watch_host_memory.sh <AGENT_PID> <run_name>
#
# torchrun rank processes each form their OWN process group (verified live:
# ranks get pgid == own pid), so killing -<agent_pgid> would only hit the
# agent. This watchdog kills the whole descendant TREE rooted at AGENT_PID
# (transitive children via /proc PPid), TERM first then KILL after 30s.
#
# - Polls every POLL_S seconds: host used% (MemTotal/MemAvailable), RSS sum
#   of the tree, GPU 4-7 memory. Appends JSONL to g2_runs/watch_<name>.jsonl
# - RED LINE: user rule = never exceed 90% of 251GB. We act at used% >= 88.0
#   sustained STRIKE_LIMIT polls (~60s) so the hard line is never crossed.
# - Never pkill -f (operator self-kill x3 last campaign); exact pids only.
set -u

ROOT_PID="${1:?usage: watch_host_memory.sh <AGENT_PID> <run_name>}"
RUN_NAME="${2:?usage: watch_host_memory.sh <AGENT_PID> <run_name>}"
POLL_S=20
KILL_PCT=88.0
STRIKE_LIMIT=3
LOG_DIR=/home/hdd3/zhanghaonan/magformer/output/aps_20260913/g2_runs
LOG="$LOG_DIR/watch_${RUN_NAME}.jsonl"
mkdir -p "$LOG_DIR"

tree_pids() {
  # transitive descendants of ROOT_PID (including it), live at snapshot time
  ps -eo pid,ppid --no-headers | awk -v r="$ROOT_PID" '
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
    }'
}

tree_rss_gb() {
  ps -eo pid,rss --no-headers | awk -v pids="$1" 'BEGIN{split(pids,a," ");for(i in a)want[a[i]]=1} $1 in want {s+=$2} END{printf "%.2f", s/1048576}'
}

strikes=0
while true; do
  PIDS="$(tree_pids | tr '\n' ' ')"
  if [ -z "$PIDS" ]; then
    echo "{\"ts\":$(date +%s),\"event\":\"tree_gone\",\"root\":$ROOT_PID}" >> "$LOG"
    break
  fi
  read -r mem_total mem_avail <<< "$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print t, a}' /proc/meminfo)"
  used_gb=$(awk -v t="$mem_total" -v a="$mem_avail" 'BEGIN{printf "%.2f", (t-a)/1048576}')
  avail_gb=$(awk -v a="$mem_avail" 'BEGIN{printf "%.2f", a/1048576}')
  used_pct=$(awk -v t="$mem_total" -v a="$mem_avail" 'BEGIN{printf "%.2f", (1-a/t)*100}')
  run_rss_gb=$(tree_rss_gb "$PIDS")
  gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr -d ' ,' | paste -sd, -)
  echo "{\"ts\":$(date +%s),\"used_gb\":$used_gb,\"avail_gb\":$avail_gb,\"used_pct\":$used_pct,\"run_rss_gb\":$run_rss_gb,\"nproc\":$(echo $PIDS | wc -w),\"gpu_mem_mib\":[$gpu_mem]}" >> "$LOG"

  over=$(awk -v u="$used_pct" -v k="$KILL_PCT" 'BEGIN{print (u>=k)?1:0}')
  if [ "$over" = "1" ]; then
    strikes=$((strikes+1))
    echo "{\"ts\":$(date +%s),\"event\":\"strike\",\"strikes\":$strikes,\"used_pct\":$used_pct}" >> "$LOG"
    if [ "$strikes" -ge "$STRIKE_LIMIT" ]; then
      echo "{\"ts\":$(date +%s),\"event\":\"KILL\",\"used_pct\":$used_pct,\"pids\":\"$PIDS\"}" >> "$LOG"
      kill -TERM $PIDS 2>/dev/null
      sleep 30
      LEFT="$(tree_pids | tr '\n' ' ')"
      [ -n "$LEFT" ] && kill -KILL $LEFT 2>/dev/null
      echo "{\"ts\":$(date +%s),\"event\":\"killed\",\"root\":$ROOT_PID}" >> "$LOG"
      break
    fi
  else
    strikes=0
  fi
  sleep "$POLL_S"
done
