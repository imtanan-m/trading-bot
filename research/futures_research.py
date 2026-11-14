"""
Futures scalping research (runs on GitHub Actions). Spot candles are used as a stand-in for perpetual
futures prices (Binance futures data is blocked from US servers; for BTC/ETH/SOL etc. the two track
each other within a few hundredths of a percent). Futures fees: taker 0.05% + 0.02% slippage per side,
funding 0.01% per 8h on open positions, 5x leverage, isolated margin with liquidation.
Train = before 2025-06-01, Test = after (never used for choosing settings).
"""
import itertools
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import market  # noqa: E402
from engine.portfolio import Portfolio, run_series  # noqa: E402
from engine.report import equity_series  # noqa: E402
from engine.strategies import build  # noqa: E402

COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT"]
TFS = {"5m": "2025-01-01", "15m": "2024-01-01", "1h": "2023-01-01"}
SPLIT = pd.Timestamp("2025-06-01", tz="UTC")
DATA = os.path.join(ROOT, "data")
BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24}


def load(sym, tf, start):
    path = os.path.join(DATA, f"{sym.replace('/', '_')}_{tf}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.astype(float)
        if len(df) and df.index[0] <= pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=2) \
                and df.index[-1] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1):
            return df
    df = market.history(sym, tf, start)
    os.makedirs(DATA, exist_ok=True)
    df.reset_index().to_csv(path, index=False)
    return df


def metrics(st, a, b):
    eq = equity_series(st)
    eq = eq[(eq.index >= a) & (eq.index < b)]
    if len(eq) < 2:
        return None
    days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400, 1)
    tr = [t for t in st["trades"] if a <= pd.Timestamp(t["exit_time"]) < b]
    wins = [t for t in tr if t["pnl"] > 0]
    gl = -sum(t["pnl"] for t in tr if t["pnl"] <= 0)
    return {"return": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
            "cagr": round(((eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1) * 100, 1),
            "max_dd": round(float((eq / eq.cummax() - 1).min() * 100), 1),
            "trades": len(tr), "per_day": round(len(tr) / days, 2),
            "win_rate": round(len(wins) / len(tr) * 100, 1) if tr else None,
            "pf": round(sum(t["pnl"] for t in wins) / gl, 2) if gl > 0 else None,
            "long_pnl": round(sum(t["pnl"] for t in tr if t.get("side") == "long"), 1),
            "short_pnl": round(sum(t["pnl"] for t in tr if t.get("side") == "short"), 1),
            "liquidations": sum(1 for t in tr if t["reason"] == "LIQUIDATED")}


def main():
    t0 = time.time()
    now = pd.Timestamp.now(tz="UTC")
    data = {tf: {c: load(c, tf, st) for c in COINS} for tf, st in TFS.items()}
    print(f"data loaded {time.time()-t0:.0f}s", flush=True)
    risk = dict(leverage=5, risk_per_trade_pct=0.5, max_position_pct=100, max_open_positions=3,
                min_cash_reserve_pct=1, daily_loss_limit_pct=5, max_drawdown_kill_pct=100, funding_pct_8h=0.01)
    V = []
    for tf in TFS:
        for mode, votes, (tp, sl) in itertools.product(("both", "trend", "range"), (4, 5),
                                                        ((2.5, 1.5), (1.5, 1.0), (4.0, 2.0))):
            if mode == "range" and votes == 5:
                continue  # votes don't matter in pure range mode
            par = dict(mode=mode, min_votes=votes, tp_atr=tp, sl_atr=sl, range_tp_atr=tp, range_sl_atr=sl,
                       max_bars={"5m": 24, "15m": 16, "1h": 24}[tf])
            V.append((f"{tf} {mode} votes{votes} TP{tp}/SL{sl}", tf, par, 0.05, 0.02))
    out = {"generated": now.isoformat(), "coins": COINS, "results": []}
    for name, tf, par, fee, slip in V:
        t1 = time.time()
        s = build("scalp", par)
        pr = {c: s.prepare(df) for c, df in data[tf].items()}
        p = Portfolio(s, risk, 1000.0, fee, slip)
        run_series(p, pr, start=pd.Timestamp(TFS[tf], tz="UTC") + pd.Timedelta(days=20))
        start = pd.Timestamp(TFS[tf], tz="UTC")
        row = {"name": name, "tf": tf, "params": par, "fee": fee, "slip": slip,
               "full": metrics(p.st, start, now), "train": metrics(p.st, start, SPLIT),
               "test": metrics(p.st, SPLIT, now), "last_90d": metrics(p.st, now - pd.Timedelta(days=90), now)}
        out["results"].append(row)
        f, tr, te = row["full"], row["train"], row["test"]
        print(f"{name:<36} {time.time()-t1:4.0f}s ret {f['return']:>7}% DD {f['max_dd']:>6}% "
              f"{f['per_day']:>5}/day win {f['win_rate']}% PF {f['pf']} L {f['long_pnl']} S {f['short_pnl']} "
              f"| train CAGR {tr['cagr']}% test CAGR {te['cagr']}% | 90d {row['last_90d']['return']}%", flush=True)
    # cheaper execution check (limit/maker orders) for the best 3 by train CAGR
    best = sorted(out["results"], key=lambda r: -(r["train"]["cagr"] or -999))[:3]
    for r in best:
        s = build("scalp", r["params"])
        pr = {c: s.prepare(df) for c, df in data[r["tf"]].items()}
        p = Portfolio(s, risk, 1000.0, 0.02, 0.01)
        run_series(p, pr, start=pd.Timestamp(TFS[r["tf"]], tz="UTC") + pd.Timedelta(days=20))
        m = {"name": r["name"] + " [maker fees 0.02%]", "tf": r["tf"], "params": r["params"], "fee": 0.02, "slip": 0.01,
             "full": metrics(p.st, pd.Timestamp(TFS[r["tf"]], tz="UTC"), now),
             "train": metrics(p.st, pd.Timestamp(TFS[r["tf"]], tz="UTC"), SPLIT), "test": metrics(p.st, SPLIT, now),
             "last_90d": metrics(p.st, now - pd.Timedelta(days=90), now)}
        out["results"].append(m)
        print(f"{m['name']:<52} ret {m['full']['return']}% DD {m['full']['max_dd']}% train {m['train']['cagr']}% "
              f"test {m['test']['cagr']}%", flush=True)
    with open(os.path.join(ROOT, "research", "futures_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
