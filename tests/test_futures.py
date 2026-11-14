"""Hand-checked futures math: short P/L, leverage margin, liquidation, take-profit, funding."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.portfolio import Portfolio, Row  # noqa: E402


class S:  # minimal futures strategy driven by a script
    futures, cross, sizing, alloc = True, False, "risk", 0.5

    def __init__(self, side, stop, tp):
        self.side, self.stop, self.tp = side, stop, tp

    def entry_signal(self, r):
        return self.side if r["go"] else 0

    def initial_stop(self, e, r, side):
        return self.stop

    def take_profit(self, e, r, side):
        return self.tp

    def trail_stop(self, p, r):
        return None

    def exit_signal(self, r, p):
        return False


def bar(t, o, h, l, c, go=False):
    return pd.Timestamp(t, tz="UTC"), Row(open=o, high=h, low=l, close=c, go=go)


def run(strat, bars, lev=5):
    p = Portfolio(strat, dict(leverage=lev, risk_per_trade_pct=1, max_position_pct=100, max_open_positions=1,
                              min_cash_reserve_pct=0, daily_loss_limit_pct=100, max_drawdown_kill_pct=100,
                              funding_pct_8h=0), 1000, fee_pct=0.0, slippage_pct=0.0)
    for t, b in bars:
        p._fill_pending("X", t, b["open"])
        p.process_bar("X", t, b)
        p.mark(t)
    return p


n = 0
def ok(c, m):
    global n
    assert c, "FAIL " + m
    n += 1
    print("  ok -", m)


# SHORT at 100, stop 105, tp 90 -> risk 1% of 1000 = $10 -> qty 2 (=$200 notional, $40 margin at 5x)
p = run(S(-1, 105, 90), [bar("2024-01-01 00:00", 100, 100, 100, 100, True), bar("2024-01-01 01:00", 100, 101, 95, 96),
                         bar("2024-01-01 02:00", 96, 97, 89, 91)])
t = p.st["trades"][0]
ok(t["side"] == "short" and t["reason"] == "take-profit" and t["exit"] == 90, "short hits take-profit at 90")
ok(abs(t["pnl"] - 20) < 1e-6 and abs(p.equity() - 1020) < 1e-6, f"short profit = 2 x (100-90) = $20 (got {t['pnl']})")
ok(abs(t["ret_pct"] - 50) < 1e-6, "return on $40 margin = +50% (5x leverage)")

# SHORT stopped out at 105 -> -$10
p = run(S(-1, 105, 90), [bar("2024-01-01 00:00", 100, 100, 100, 100, True), bar("2024-01-01 01:00", 100, 106, 99, 104)])
ok(abs(p.st["trades"][0]["pnl"] + 10) < 1e-6, "short stop-loss loses exactly the 1% risk ($10)")

# open position value while running: short 2 @100, price 98 -> margin 40 + 4
p = run(S(-1, 105, 80), [bar("2024-01-01 00:00", 100, 100, 100, 100, True), bar("2024-01-01 01:00", 100, 100, 98, 98)])
ok(abs(p.equity() - 1004) < 1e-6, "unrealized short P/L counted in equity")

# LONG with stop beyond liquidation: 10x, entry 100, liq = 100*(1-0.1+0.005) = 90.5, stop 80
p = run(S(1, 80, 130), [bar("2024-01-01 00:00", 100, 100, 100, 100, True), bar("2024-01-01 01:00", 100, 100, 90, 91)], lev=10)
t = p.st["trades"][0]
ok(t["reason"] == "LIQUIDATED" and abs(t["exit"] - 90.5) < 1e-9, "10x long liquidated at 90.5 before far stop")
ok(t["pnl"] < 0 and p.equity() > 0, f"liquidation loses (most of) margin only: P/L ${t['pnl']:.2f}")

# gap through the stop fills at the worse open
p = run(S(1, 95, 130), [bar("2024-01-01 00:00", 100, 100, 100, 100, True), bar("2024-01-01 01:00", 92, 93, 90, 91)])
ok(p.st["trades"][0]["exit"] == 92, "gap below stop fills at the open (92), not at 95")

# funding: 0.01%/8h on notional, held 24h
p = Portfolio(S(1, 90, 200), dict(leverage=5, risk_per_trade_pct=1, max_position_pct=100, max_open_positions=1,
              min_cash_reserve_pct=0, daily_loss_limit_pct=100, max_drawdown_kill_pct=100, funding_pct_8h=0.01),
              1000, 0.0, 0.0)
for t_, b in [bar("2024-01-01 00:00", 100, 100, 100, 100, True), bar("2024-01-01 01:00", 100, 100, 100, 100)]:
    p._fill_pending("X", t_, b["open"]); p.process_bar("X", t_, b)
p._close("X", 100, pd.Timestamp("2024-01-02 01:00", tz="UTC"), "test")
qty = 1.0  # risk $10 / (100-90)
ok(abs(p.st["trades"][0]["pnl"] + qty * 100 * 0.0001 * 3) < 1e-6, "funding charged: 3 x 0.01% of notional for 24h")
print(f"\nALL {n} FUTURES CHECKS PASSED")
