"""Regression: a coin that joins a running bot later must NOT have its old history traded as if live
(this bug once gave Opportunity Hunter a fake +32%)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_backtests as mb  # noqa: E402
from engine.portfolio import Portfolio, run_series  # noqa: E402
from engine.strategies import build  # noqa: E402

d = {s: mb.load(s, "1d") for s in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]}
s = build("rotation", dict(k=3))
p = Portfolio(s, dict(max_open_positions=3, max_drawdown_kill_pct=100, daily_loss_limit_pct=100), 1000)
t0 = pd.Timestamp("2025-01-01", tz="UTC")
run_series(p, {"BTC/USDT": s.prepare(d["BTC/USDT"][d["BTC/USDT"].index <= t0])}, start=t0)
run_series(p, {k: s.prepare(v[v.index <= t0 + pd.Timedelta(days=1)]) for k, v in d.items()})
assert not [e for e in p.st["equity"] if pd.Timestamp(e[0]) < t0], "old history was replayed"
assert not [t for t in p.st["trades"] if pd.Timestamp(t["entry_time"]) < t0], "old trades created"
print("ok - joining coins start from the bot's current time (no history replay)")
