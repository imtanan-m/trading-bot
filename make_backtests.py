"""
Backtest every bot in bots.yaml with the SAME engine the cloud uses, and write
docs/data/backtest.json for the dashboard.   python make_backtests.py [start]
Uses CSVs in data/ if present (SYMBOL_TF.csv), otherwise downloads from Binance.
"""
import json
import os
import sys

import pandas as pd
import yaml

from engine import market
from engine.portfolio import Portfolio, run_series
from engine.report import downsample, equity_series, monthly, stats
from engine.strategies import build

HERE = os.path.dirname(os.path.abspath(__file__))
START = sys.argv[1] if len(sys.argv) > 1 else "2022-06-01"
WARMUP_FROM = "2021-06-01"


def load(sym, tf):
    def csv(t):
        p = os.path.join(HERE, "data", f"{sym.replace('/', '_')}_{t}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["ts"], index_col="ts")
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df.astype(float).sort_index()
        return None
    df = csv(tf)
    if df is None and tf == "1d":  # build daily from the finest local data we have
        src = csv("4h") if csv("4h") is not None else csv("1h")
        if src is not None:
            df = src.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                         "close": "last", "volume": "sum"}).dropna()
    if df is None:
        df = market.history(sym, tf, WARMUP_FROM)
    return df


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "bots.yaml"), encoding="utf-8"))
    out = {"start": START, "generated": pd.Timestamp.now(tz="UTC").isoformat(), "bots": {}}
    for b in cfg["bots"]:
        s = build(b["kind"], b.get("params"))
        pr = {sym: s.prepare(load(sym, b["timeframe"])) for sym in b["symbols"]}
        p = Portfolio(s, b.get("risk", {}), float(b["capital"]), cfg["fee_pct"], cfg["slippage_pct"])
        run_series(p, pr, start=pd.Timestamp(START, tz="UTC"))
        eq = equity_series(p.st)
        st = stats(p.st, float(b["capital"]))
        st["return_pct"] = round((eq.iloc[-1] / float(b["capital"]) - 1) * 100, 1)
        st["killed"] = p.st["killed"]
        yearly = eq.resample("YE").last()
        prev, yr = float(b["capital"]), {}
        for t, v in yearly.items():
            yr[str(t.year)] = round((v / prev - 1) * 100, 1)
            prev = v
        st["yearly"] = yr
        st["monthly"] = monthly(eq, float(b["capital"]))
        out["bots"][b["id"]] = {"stats": st, "curve": downsample(eq, 250)}
        print(f"{b['name']:<28} return {st['return_pct']:>8}%  CAGR {st.get('cagr')}%  DD {st['max_dd']}%  "
              f"trades {st['trades']}  {yr}")
    os.makedirs(os.path.join(HERE, "docs", "data"), exist_ok=True)
    with open(os.path.join(HERE, "docs", "data", "backtest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
