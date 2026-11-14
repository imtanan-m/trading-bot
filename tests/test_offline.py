"""
Offline end-to-end test (no internet). Replays historical candles as if the cloud
runner was called every 6 hours for 60 days, then checks:
  - state continues correctly between runs (same result as one continuous backtest)
  - summary.json has everything the dashboard needs
  - one broken bot doesn't stop the others

    python tests/test_offline.py        (needs CSVs in data/, see make_backtests.py)
"""
import json
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_backtests  # noqa: E402
import run_cycle  # noqa: E402
from engine import market  # noqa: E402

FULL = {}
NOW = {"t": None}
TF = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def fake_klines(sym, tf, limit=1000):
    key = (sym, tf)
    if key not in FULL:
        FULL[key] = make_backtests.load(sym, tf)
    df = FULL[key]
    df = df[df.index + TF[tf] <= NOW["t"]]  # only candles closed by "now"
    return df.iloc[-limit:]


def fake_prices(symbols):
    return {s: float(fake_klines(s, "1h").close.iloc[-1]) for s in symbols}


class NoNews:
    def __init__(self, *a, **k):
        pass

    def check(self, s):
        return True, ""

    def fear_greed(self):
        return 50


market.klines, market.prices, market.usd_pkr = fake_klines, fake_prices, lambda: 281.5
market.top_coins = lambda n=40: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
run_cycle.NewsGuard = NoNews
ok_count = 0


def ok(c, m):
    global ok_count
    assert c, "FAIL: " + m
    ok_count += 1
    print("  ok -", m)


with tempfile.TemporaryDirectory() as d:
    run_cycle.STATE_DIR, run_cycle.OUT = os.path.join(d, "state"), os.path.join(d, "out")
    start = pd.Timestamp("2024-02-01 00:05", tz="UTC")
    for i in range(240):  # every 6h for 60 days
        NOW["t"] = start + pd.Timedelta(hours=6 * i)
        run_cycle.main()
    summ = json.load(open(os.path.join(run_cycle.OUT, "summary.json")))
    ok(len(summ["bots"]) == 7 and not summ["errors"], "7 bots ran without errors")
    need = {"id", "name", "equity", "return_pct", "positions", "trades", "stats", "curve", "change_24h", "radar"}
    ok(set(summ["coins"]) == {"BTC/USDT", "ETH/USDT", "SOL/USDT"} and summ["usd_pkr"], "market snapshot + PKR rate")
    ok(all(r["conditions"] for b in summ["bots"] if b["id"] != "hold" for r in b["radar"].values()), "signal radar filled")
    ok(all("score" in r for b in summ["bots"] if b["scanner"] for r in b["radar"].values()), "scanner bots rank coins")
    ok(all(need <= set(b) for b in summ["bots"]), "summary has all dashboard fields")
    byid = {b["id"]: b for b in summ["bots"]}
    ok(byid["hold"]["positions"] and len(byid["hold"]["positions"]) == 3, "buy & hold bought all 3 coins")
    ok(sum(b["stats"]["trades"] for b in summ["bots"]) > 5, "bots made trades: " +
       ", ".join(f"{b['id']}={b['stats']['trades']}" for b in summ["bots"]))
    for b in summ["bots"]:
        ok(b["equity"] > 0 and b["cash"] >= -1e-6, f"{b['id']}: equity ${b['equity']:.2f}, cash ok")

    # continuity: incremental runs == one continuous run over the same candles
    st = json.load(open(os.path.join(run_cycle.STATE_DIR, "trend_1h.json")))
    from engine.portfolio import Portfolio, run_series
    from engine.strategies import build
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "bots.yaml")))
    b = next(x for x in cfg["bots"] if x["id"] == "trend_1h")
    s = build(b["kind"], b["params"])
    first = pd.Timestamp(st["equity"][0][0])
    pr = {sym: s.prepare(fake_klines(sym, "1h", 5000)) for sym in b["symbols"]}
    p = Portfolio(s, b["risk"], b["capital"], cfg["fee_pct"], cfg["slippage_pct"])
    run_series(p, pr, start=first)
    ok(abs(p.equity() - st["equity"][-1][1]) < 0.5 and len(p.st["trades"]) == len(st["trades"]),
       f"incremental == continuous (${p.equity():.2f} vs ${st['equity'][-1][1]:.2f}, "
       f"{len(st['trades'])} trades)")

    # same check for a multi-coin scanner bot (BTC filter + point-in-time universe)
    from engine.portfolio import btc_regime, volume_universe
    st = json.load(open(os.path.join(run_cycle.STATE_DIR, "hunter_daily.json")))
    b = next(x for x in cfg["bots"] if x["id"] == "hunter_daily")
    s = build(b["kind"], b["params"])
    syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    daily = {sym: fake_klines(sym, "1d", 5000) for sym in syms}
    p = Portfolio(s, b["risk"], b["capital"], cfg["fee_pct"], cfg["slippage_pct"])
    run_series(p, {k: s.prepare(v) for k, v in daily.items()}, start=pd.Timestamp(st["equity"][0][0]),
               regime=btc_regime(daily["BTC/USDT"]), eligible=volume_universe(daily, top=30))
    ok(abs(p.equity() - st["equity"][-1][1]) < 0.5 and len(p.st["trades"]) == len(st["trades"]),
       f"scanner incremental == continuous (${p.equity():.2f} vs ${st['equity'][-1][1]:.2f})")

    # a broken bot must not stop the others
    orig = run_cycle.run_bot

    def broken(bb, *a):
        if bb["id"] == "trend_4h":
            raise RuntimeError("simulated failure")
        return orig(bb, *a)
    run_cycle.run_bot = broken
    NOW["t"] += pd.Timedelta(hours=6)
    run_cycle.main()
    summ = json.load(open(os.path.join(run_cycle.OUT, "summary.json")))
    ok(len(summ["bots"]) == 6 and len(summ["errors"]) == 1, "one failing bot is isolated and reported")

print(f"\nALL {ok_count} CHECKS PASSED")
