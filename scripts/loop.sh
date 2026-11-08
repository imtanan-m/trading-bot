#!/usr/bin/env bash
# Runs the bot every INTERVAL seconds (aligned to the clock, +60s after each quarter hour)
# for LOOP_MINUTES, committing state + dashboard data after every cycle.
set -u
INTERVAL=${INTERVAL:-900}
end=$(( $(date +%s) + ${LOOP_MINUTES:-330} * 60 ))
git config user.name "trading-bot"
git config user.email "trading-bot@users.noreply.github.com"
while :; do
  git pull --rebase -q || echo "pull failed, continuing"
  python run_cycle.py || echo "cycle failed, will retry next round"
  git add state docs/data
  if ! git diff --cached --quiet; then
    git commit -qm "bot update $(date -u +'%Y-%m-%d %H:%M')"
    for i in 1 2 3; do git pull --rebase -q && git push -q && break; sleep 5; done
  fi
  now=$(date +%s)
  next=$(( (now / INTERVAL + 1) * INTERVAL + 60 ))
  [ "$next" -ge "$end" ] && break
  sleep $(( next - now ))
done
echo "loop finished"
