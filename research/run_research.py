"""
Research: which multi-coin strategy actually works? Runs on GitHub Actions (Binance is reachable there).
  1. universe = top N USDT spot coins by 24h volume (no stablecoins / leveraged / wrapped tokens)
  2. download daily + 4h history
  3. backtest many variants with the same engine the live bots use
  4. write research/results.json + research/results.md
NOTE: today's top coins are used for the whole history -> survivorship bias (coins that died are
missing), so real results would likely be somewhat worse than shown.
"""
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import market  # noqa: E402
from engine.portfolio import Portfolio, btc_regime, run_series  # noqa: E402
from engine.report import equity_series, monthly  # noqa: E402
from engine.strategies import build  # noqa: E402

N = int(os.environ.get("UNIVERSE_N", "40"))
START = pd.Timestamp(os.environ.get("START", "2022-06-01"), tz="UTC")
SPLIT = pd.Timestamp("2024-06-01", tz="UTC")
DATA = os.path.join(ROOT, "data")
STABLE = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "TRY", "BRL", "AEUR", "USDE", "PYUSD", "USD1",
          "BFUSD", "RLUSD", "EURI", "XUSD", "UST", "USTC", "GBP", "AUD", "JPY", "PAXG", "XAUT", "WBTC", "WBETH",
          "BETH", "STETH", "WSTETH", "USDS", "U", "BNSOL"}


def universe():
    info = market._get("/api/v3/exchangeInfo", {"permissions": "SPOT"})
    ok = {s["symbol"]: s["baseAsset"] for s in info["symbols"]
          if s["status"] == "TRADING" and s["quoteAsset"] == "USDT" and s.get("isSpotTradingAllowed", True)}
    tick = market._get("/api/v3/ticker/24hr", {})
    rows = []
    for t in tick:
        sym = t["symbol"]
        if sym not in ok:
            continue
        base = ok[sym]
        if base in STABLE or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        rows.append((float(t["quoteVolume"]), f"{base}/USDT"))
    rows.sort(reverse=True)
    syms = [s for _, s in rows[:N]]
    if "BTC/USDT" not in syms:
        syms.append("BTC/USDT")
    return syms


def load(sym, tf, start):
    path = os.path.join(DATA, f"{sym.replace('/', '_')}_{tf}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.astype(float)
        if len(df) and df.index[-1] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2):
            return df
    df = market.history(sym, tf, start)
    os.makedirs(DATA, exist_ok=True)
    df.reset_index().to_csv(path, index=False)
    return df


def metrics(st, capital, a, b):
    eq = equity_series(st)
    eq = eq[(eq.index >= a) & (eq.index < b)]
    if len(eq) < 2:
        return None
    base = eq.iloc[0]
    days = (eq.index[-1] - eq.index[0]).days or 1
    tr = [t for t in st["trades"] if a <= pd.Timestamp(t["exit_time"]) < b]
    wins = [t for t in tr if t["pnl"] > 0]
    gl = -sum(t["pnl"] for t in tr if t["pnl"] <= 0)
    m = eq.resample("ME").last().pct_change().dropna()
    hold = [(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 86400 for t in tr]
    return {"return": round((eq.iloc[-1] / base - 1) * 100, 1),
            "cagr": round(((eq.iloc[-1] / base) ** (365.25 / days) - 1) * 100, 1),
            "max_dd": round(float((eq / eq.cummax() - 1).min() * 100), 1),
            "trades": len(tr), "per_week": round(len(tr) / (days / 7), 2),
            "win_rate": round(len(wins) / len(tr) * 100, 1) if tr else None,
            "pf": round(sum(t["pnl"] for t in wins) / gl, 2) if gl > 0 else None,
            "avg_hold_days": round(sum(hold) / len(hold), 1) if hold else None,
            "pos_months": round(float((m > 0).mean() * 100)) if len(m) else None}


def main():
    t0 = time.time()
    syms = universe()
    print("universe:", syms, flush=True)
    d1 = {s: load(s, "1d", "2021-06-01") for s in syms}
    h4 = {s: load(s, "4h", "2021-11-01") for s in syms}
    full_hist = [s for s in syms if d1[s].index[0] <= START - pd.Timedelta(days=60)]
    print(f"data loaded in {time.time()-t0:.0f}s; {len(full_hist)}/{len(syms)} coins have full history", flush=True)
    reg = btc_regime(d1["BTC/USDT"], 50)
    now = pd.Timestamp.now(tz="UTC")
    periods = {"full": (START, now), "A_2022_24": (START, SPLIT), "B_2024_now": (SPLIT, now),
               "last_90d": (now - pd.Timedelta(days=90), now)}
    big = dict(max_drawdown_kill_pct=100, daily_loss_limit_pct=100, min_cash_reserve_pct=1)
    V = []
    for lb in (14, 28):
        for sma in (20, 50):
            for k in (3, 5):
                for rg in (False, True):
                    V.append((f"rotation lb{lb} sma{sma} k{k}{' +BTCfilter' if rg else ''}", "rotation",
                              dict(lookback=lb, sma=sma, k=k), "1d", dict(big, max_open_positions=k), rg))
    for tp, sl in ((0.15, 0.10), (0.25, 0.12)):
        for rg in (False, True):
            V.append((f"rotation lb28 sma20 k3 TP{int(tp*100)} SL{int(sl*100)}{' +BTCfilter' if rg else ''}", "rotation",
                      dict(lookback=28, sma=20, k=3, take_profit_pct=tp, stop_pct=sl), "1d", dict(big, max_open_positions=3), rg))
    for rb in (25, 30):
        for tp in (0.05, 0.10):
            for ts in (0, 100):
                V.append((f"dip rsi<{rb} TP{int(tp*100)} SL8 trend{ts or 'off'}", "dip",
                          dict(rsi_buy=rb, take_profit_pct=tp, stop_pct=0.08, max_bars=10, trend_sma=ts, k=3), "1d",
                          dict(big, max_open_positions=3), False))
    for tpa in (0, 4):
        for rg in (False, True):
            V.append((f"trend4h hunter TPatr{tpa or 'off'}{' +BTCfilter' if rg else ''}", "trend",
                      dict(rank=True, take_profit_atr=tpa, trail_stop_atr=3.5), "4h",
                      dict(risk_per_trade_pct=0.75, max_position_pct=25, max_open_positions=4,
                           min_cash_reserve_pct=2, daily_loss_limit_pct=3, max_drawdown_kill_pct=100), rg))
    V.append(("BENCH hold BTC/ETH/SOL", "hold", dict(n_coins=3), "1d", dict(big, max_open_positions=3), False))
    V.append(("BENCH current momentum_daily (3 coins)", "momentum", dict(sma=50, lookback=30, n_coins=3), "1d",
              dict(big, max_open_positions=3), False))
    out = {"generated": now.isoformat(), "universe": syms, "full_history": full_hist, "results": []}
    for name, kind, par, tf, risk, rg in V:
        t1 = time.time()
        s = build(kind, par)
        src = d1 if tf == "1d" else h4
        coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT"] if name.startswith("BENCH") else syms
        pr = {c: s.prepare(src[c]) for c in coins if c in src}
        p = Portfolio(s, risk, 1000.0, 0.1, 0.05)
        run_series(p, pr, start=START, regime=reg if rg else None)
        row = {"name": name, "kind": kind, "tf": tf, "params": par, "btc_filter": rg,
               "periods": {k: metrics(p.st, 1000.0, a, b) for k, (a, b) in periods.items()},
               "monthly": monthly(equity_series(p.st), 1000.0),
               "top_coins": pd.Series([t["pnl"] for t in p.st["trades"]],
                                      index=[t["symbol"] for t in p.st["trades"]]).groupby(level=0).sum()
                          .sort_values(ascending=False).round(1).head(8).to_dict() if p.st["trades"] else {}}
        out["results"].append(row)
        f = row["periods"]["full"]
        print(f"{name:<48} {time.time()-t1:5.0f}s  ret {f['return']:>8}%  CAGR {f['cagr']:>6}%  DD {f['max_dd']:>6}%  "
              f"tr {f['trades']:>4} ({f['per_week']}/wk)  win {f['win_rate']}%  PF {f['pf']}  "
              f"A {row['periods']['A_2022_24']['cagr']}%  B {row['periods']['B_2024_now']['cagr']}%  "
              f"90d {row['periods']['last_90d']['return']}%", flush=True)
    os.makedirs(os.path.join(ROOT, "research"), exist_ok=True)
    with open(os.path.join(ROOT, "research", "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
