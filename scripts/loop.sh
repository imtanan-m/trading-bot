#!/usr/bin/env bash
# Runs the bot every INTERVAL seconds (aligned to the clock, +60s after each quarter hour)
# for LOOP_MINUTES, committing state + dashboard data after every cycle.
# The GitHub copy is always the source of truth: each cycle starts from it, and if a push races with
# another change, this cycle's update is dropped (the next cycle catches up on the missed candles).
set -u
INTERVAL=${INTERVAL:-900}
end=$(( $(date +%s) + ${LOOP_MINUTES:-330} * 60 ))
git config user.name "trading-bot"
git config user.email "trading-bot@users.noreply.github.com"
sync_main() { git rebase --abort >/dev/null 2>&1; git fetch -q origin main && git reset -q --hard origin/main; }
while :; do
  sync_main || echo "sync failed, continuing with local copy"
  python run_cycle.py || echo "cycle failed, will retry next round"
  git add state docs/data
  if ! git diff --cached --quiet; then
    git commit -qm "bot update $(date -u +'%Y-%m-%d %H:%M')"
    pushed=0
    for i in 1 2 3; do
      if git pull --rebase -q && git push -q; then pushed=1; break; fi
      git rebase --abort >/dev/null 2>&1; sleep 5
    done
    [ "$pushed" = 1 ] || { echo "push failed, dropping this update"; sync_main; }
  fi
  now=$(date +%s)
  next=$(( (now / INTERVAL + 1) * INTERVAL + 60 ))
  [ "$next" -ge "$end" ] && break
  sleep $(( next - now ))
done
echo "loop finished"
