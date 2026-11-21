"""
Upgrade research (GitHub Actions): which new bots would actually have made money?
  A) Momentum LONG/SHORT (futures, daily)      on BTC/ETH/SOL and on the top-10 coins
  B) Futures trend on 4h candles (fewer trades than 1h -> fewer fees)
  C) Pump Rider (buy Pump-Radar signals with target / stop / time limit) on ALL liquid coins
  D) Dip Catcher with a BTC market filter (don't buy dips while BTC is in a downtrend)
Train = before 2025-06-01, Test = after.  Results -> research/upgrade_results.json
"""
import itertools
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
from engine import market  # noqa: E402
from engine.portfolio import Portfolio, btc_regime, run_series, volume_universe  # noqa: E402
from engine.report import equity_series  # noqa: E402
from engine.strategies import build  # noqa: E402
import pump_research as PR  # noqa: E402

START = pd.Timestamp("2023-06-01", tz="UTC")
SPLIT = pd.Timestamp("2025-06-01", tz="UTC")
TOP10 = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT", "SUI/USDT"]


def load4h(sym):
    path = os.path.join(ROOT, "data", f"{sym.replace('/', '_')}_4h_up.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if df.index[-1] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1):
            return df.astype(float)
    df = market.history(sym, "4h", "2022-11-01")
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
            "trades": len(tr), "per_week": round(len(tr) / days * 7, 2),
            "win_rate": round(len(wins) / len(tr) * 100, 1) if tr else None,
            "pf": round(sum(t["pnl"] for t in wins) / gl, 2) if gl > 0 else None,
            "long_pnl": round(sum(t["pnl"] for t in tr if t.get("side", "long") == "long"), 1),
            "short_pnl": round(sum(t["pnl"] for t in tr if t.get("side") == "short"), 1)}


def main():
    t0 = time.time()
    now = pd.Timestamp.now(tz="UTC")
    coins = PR.all_usdt_coins()
    daily = {}
    for c in coins:
        try:
            df = PR.load(c)
            if len(df) >= 120:
                daily[c] = df
        except Exception:  # noqa: BLE001
            pass
    btc = daily["BTC/USDT"]
    reg = btc_regime(btc, 50)
    h4 = {c: load4h(c) for c in TOP10}
    print(f"data: {len(daily)} daily coins, 4h {len(h4)}; {time.time()-t0:.0f}s", flush=True)
    periods = {"full": (START, now), "train": (START, SPLIT), "test": (SPLIT, now), "last_90d": (now - pd.Timedelta(days=90), now)}
    big = dict(max_drawdown_kill_pct=100, daily_loss_limit_pct=100, min_cash_reserve_pct=1)
    V = []  # (name, kind, params, data, risk, fee, slip, regime, eligible, setup)
    for coins_name, cs in (("BTC/ETH/SOL", TOP10[:3]), ("top10", TOP10)):
        for sma, lb in ((50, 30), (20, 14)):
            for sh in (True, False):
                for lev in (1, 2):
                    V.append((f"A momentum {'L/S' if sh else 'long'} {coins_name} sma{sma} lb{lb} {lev}x", "momls",
                              dict(sma=sma, lookback=lb, n_coins=len(cs), shorts=sh), {c: daily[c] for c in cs},
                              dict(big, leverage=lev, max_open_positions=len(cs), funding_pct_8h=0.01), 0.05, 0.02, None, None))
    for votes, (tp, sl), mb in itertools.product((4, 5), ((4.0, 2.0), (6.0, 3.0)), (12, 30)):
        V.append((f"B futures 4h trend votes{votes} TP{tp}/SL{sl} max{mb}", "scalp",
                  dict(mode="trend", min_votes=votes, tp_atr=tp, sl_atr=sl, max_bars=mb), h4,
                  dict(leverage=5, funding_pct_8h=0.01, risk_per_trade_pct=0.5, max_position_pct=100, max_open_positions=5,
                       min_cash_reserve_pct=1, daily_loss_limit_pct=6, max_drawdown_kill_pct=100), 0.05, 0.02, None, None))
    liquid = {c: df for c, df in daily.items()}
    for sig, (tp, sl), mb, rg in itertools.product(("quiet", "spike", "either"), ((0.2, 0.1), (0.3, 0.12), (0.5, 0.15)),
                                                  (10, 20), (False, True)):
        V.append((f"C pump-rider {sig} TP{int(tp*100)} SL{int(sl*100)} {mb}d{' +BTCfilter' if rg else ''}", "breakout",
                  dict(signal=sig, take_profit_pct=tp, stop_pct=sl, max_bars=mb, k=5), liquid,
                  dict(big, max_open_positions=5), 0.1, 0.1, reg if rg else None, None))
    pit = volume_universe(daily, top=30)
    for rg in (False, True):
        for tp in (0.08, 0.12):
            V.append((f"D dip rsi<25 TP{int(tp*100)} SL8{' +BTCfilter' if rg else ''}", "dip",
                      dict(rsi_buy=25, take_profit_pct=tp, stop_pct=0.08, max_bars=10, k=3), daily,
                      dict(big, max_open_positions=3), 0.1, 0.05, reg if rg else None, pit))
    V.append(("BENCH hold BTC/ETH/SOL", "hold", dict(n_coins=3), {c: daily[c] for c in TOP10[:3]}, dict(big, max_open_positions=3),
              0.1, 0.05, None, None))
    out = {"generated": now.isoformat(), "results": []}
    for name, kind, par, data, risk, fee, slip, rg, el in V:
        t1 = time.time()
        s = build(kind, par)
        if kind == "breakout":
            s.btc = btc
        pr = {c: s.prepare(df) for c, df in data.items()}
        p = Portfolio(s, risk, 1000.0, fee, slip)
        run_series(p, pr, start=START, regime=rg, eligible=el)
        row = {"name": name, "kind": kind, "params": par, "risk": risk, "fee": fee, "slip": slip,
               **{k: metrics(p.st, a, b) for k, (a, b) in periods.items()}}
        out["results"].append(row)
        f, tr, te, l9 = row["full"], row["train"], row["test"], row["last_90d"]
        print(f"{name:<52} {time.time()-t1:4.0f}s ret {f['return']:>7}% DD {f['max_dd']:>6}% {f['per_week']:>5}/wk "
              f"win {f['win_rate']} PF {f['pf']} L {f['long_pnl']} S {f['short_pnl']} | train {tr['cagr']}% "
              f"test {te['cagr']}% | 90d {l9['return'] if l9 else None}%", flush=True)
    with open(os.path.join(ROOT, "research", "upgrade_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
