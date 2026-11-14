"""
One cloud cycle (GitHub Actions runs this every ~15 minutes):
  1. download latest closed candles for every bot
  2. continue each bot's paper portfolio from its saved state (state/<id>.json)
  3. write docs/data/summary.json for the dashboard app

Run locally:  python run_cycle.py
"""
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

import pandas as pd
import yaml

import copy

from engine import market, notify
from engine.news import NewsGuard
from engine.portfolio import Portfolio, btc_regime, run_series, volume_universe
from engine.report import downsample, equity_series, stats
from engine.strategies import build

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
OUT = os.path.join(HERE, "docs", "data")
NEWS_CFG = {
    "enabled": True, "fear_greed_block_above": 90, "lookback_hours": 12,
    "rss_feeds": ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss",
                  "https://decrypt.co/feed"],
    "danger_keywords": ["hack", "hacked", "exploit", "stolen", "drained", "delist", "halt", "suspend",
                        "insolvent", "bankrupt", "lawsuit", "sues", "ban", "banned", "emergency"],
}
log = logging.getLogger("cycle")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def get_klines(cache, sym, tf, limit=1000):
    key = (sym, tf)
    if key not in cache:
        cache[key] = market.klines(sym, tf, limit)
    return cache[key]


def coin_pool(n):
    """Top-n coins by volume, refreshed once a day (saved in state/universe.json)."""
    path = os.path.join(STATE_DIR, "universe.json")
    u = load_json(path, {})
    fresh = u.get("date") == datetime.now(timezone.utc).strftime("%Y-%m-%d") and len(u.get("coins", [])) >= n
    if not fresh:
        try:
            u = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "coins": market.top_coins(n)}
            save_json(path, u)
        except Exception as e:  # noqa: BLE001
            log.warning(f"universe refresh failed, using last list: {e}")
    return u.get("coins", [])[:n]


def bot_symbols(b, state):
    if "universe" not in b:
        return list(b["symbols"])
    syms = coin_pool(int(b["universe"].get("pool", 80)))
    held = list((state or {}).get("positions", {})) + list((state or {}).get("pend_entry", {}))
    return list(dict.fromkeys(["BTC/USDT"] + syms + held))


def run_bot(b, cfg, news, cache):
    strat = build(b["kind"], b.get("params"))
    path = os.path.join(STATE_DIR, f"{b['id']}.json")
    state = load_json(path)
    p = Portfolio(strat, b.get("risk", {}), float(b["capital"]), cfg.get("fee_pct", 0.1),
                  cfg.get("slippage_pct", 0.05), state)
    prepared, raw = {}, {}
    for sym in bot_symbols(b, state):
        try:
            raw[sym] = get_klines(cache, sym, b["timeframe"])
        except Exception as e:  # noqa: BLE001 - a single delisted/odd coin must not stop the bot
            log.warning(f"{b['id']}: {sym} skipped ({e})")
            continue
        if len(raw[sym]) >= 30:
            prepared[sym] = strat.prepare(raw[sym])
    regime = eligible = None
    if b.get("btc_filter"):
        regime = btc_regime(get_klines(cache, "BTC/USDT", "1d"), int(b.get("btc_filter_sma", 50)))
    if "universe" in b:
        daily = {s: get_klines(cache, s, "1d") for s in prepared}
        eligible = volume_universe(daily, top=int(b["universe"].get("top", 30)))
        p.eligible_now = {s: eligible(pd.Timestamp.now(tz="UTC"), s) for s in prepared}
    p.regime_now = regime(pd.Timestamp.now(tz="UTC")) if regime else None
    b["symbols"] = list(prepared)
    start = None
    if state is None:  # brand-new bot: start trading from the latest closed candle, not from history
        start = max(df.index[-1] for df in prepared.values())
        log.info(f"{b['id']}: new bot, starting at {start}")
    flt = news.check if b.get("news_filter") else None
    n_before = len(p.st["trades"])
    before = copy.deepcopy({"positions": p.st["positions"], "trades": p.st["trades"], "killed": p.st["killed"]})
    run_series(p, prepared, start=start, entry_filter_last=flt, regime=regime, eligible=eligible)
    p.st["events"] = p.st.get("events", [])[-50:]
    save_json(path, p.st)
    p.alerts = [] if state is None else notify.diff_messages(b["name"], before, p.st)
    p.radar = {}
    for sym, df in prepared.items():
        r = df.iloc[-1]
        conds = strat.explain(r)
        p.radar[sym] = {"candle": r.name.isoformat(), "close": round(float(r.close), 4), "conditions": conds,
                        "met": sum(c["ok"] for c in conds), "total": len(conds)}
        if getattr(strat, "cross", False):
            sc = strat.score(r)
            p.radar[sym]["score"] = round(float(sc), 3) if sc == sc else None
            p.radar[sym]["eligible"] = getattr(p, "eligible_now", {}).get(sym, True)
    log.info(f"{b['id']}: equity {p.equity():.2f}, open {list(p.st['positions'])}, "
             f"new trades {len(p.st['trades']) - n_before}")
    return p


def bot_summary(b, p, live):
    st, cap = p.st, float(b["capital"])
    positions = []
    live_eq = st["cash"]
    for sym, pos in st["positions"].items():
        px = live.get(sym, st["last_close"].get(sym, pos["entry"]))
        live_eq += pos["qty"] * px
        stop_dist = round((pos["stop"] / px - 1) * 100, 2) if pos["stop"] else None
        positions.append({"stop_dist": stop_dist, "symbol": sym, "qty": round(pos["qty"], 6), "entry": round(pos["entry"], 4),
                          "price": round(px, 4), "stop": round(pos["stop"], 4) if pos["stop"] else None,
                          "value": round(pos["qty"] * px, 2), "pnl_pct": round((px / pos["entry"] - 1) * 100, 2),
                          "since": pos["time"]})
    eq = equity_series(st)
    day_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    week_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)

    def change_since(ts):
        e = eq[eq.index <= ts]
        base = float(e.iloc[-1]) if len(e) else cap
        return round((live_eq / base - 1) * 100, 2)

    return {
        "id": b["id"], "name": b["name"], "tag": b.get("tag", ""), "color": b.get("color", "#888"),
        "about": " ".join(b.get("about", "").split()), "timeframe": b["timeframe"], "symbols": b["symbols"],
        "capital": cap, "equity": round(live_eq, 2), "cash": round(st["cash"], 2),
        "return_pct": round((live_eq / cap - 1) * 100, 2),
        "change_24h": change_since(day_ago), "change_7d": change_since(week_ago),
        "started": st.get("created"), "killed": st.get("killed"),
        "positions": positions, "pending": list(st["pend_entry"]),
        "trades": st["trades"][-200:][::-1], "events": st.get("events", [])[-10:][::-1],
        "stats": stats(st, cap), "curve": downsample(eq, 400),
        "radar": getattr(p, "radar", {}), "params": b.get("params", {}), "risk": b.get("risk", {}),
        "kind": b["kind"], "news_filter": bool(b.get("news_filter")),
        "scanner": "universe" in b, "btc_filter": bool(b.get("btc_filter")),
        "market_ok": getattr(p, "regime_now", None),
        "universe": b.get("universe"),
    }


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "bots.yaml"), encoding="utf-8"))
    news = NewsGuard(NEWS_CFG, log)
    cache, bots_out, errors = {}, [], []
    core = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    for b in cfg["bots"]:
        try:
            p = run_bot(b, cfg, news, cache)
            bots_out.append((b, p))
        except Exception as e:  # noqa: BLE001 - one broken bot must not stop the others
            log.error(f"{b['id']} failed: {e}\n{traceback.format_exc()}")
            errors.append(f"{b['name']}: {e}")
    held = sorted({s for _, p in bots_out for s in p.st["positions"]} - set(core))
    all_syms = core + held
    live = market.prices(all_syms) if bots_out else {}
    coins = {}
    for sym in all_syms:
        try:
            h = cache.get((sym, "1h"))
            if h is None:
                h = cache[(sym, "1h")] = market.klines(sym, "1h", 200)
            last = live.get(sym, float(h.close.iloc[-1]))
            wk = h.close.iloc[-168:]
            coins[sym] = {"price": round(last, 4),
                          "change_24h": round((last / float(h.close.iloc[-25]) - 1) * 100, 2) if len(h) > 25 else None,
                          "change_7d": round((last / float(wk.iloc[0]) - 1) * 100, 2),
                          "high_7d": round(float(h.high.iloc[-168:].max()), 4),
                          "low_7d": round(float(h.low.iloc[-168:].min()), 4),
                          "spark": [round(float(v), 4) for v in wk.iloc[::2]]}
        except Exception as e:  # noqa: BLE001
            log.warning(f"market snapshot {sym}: {e}")
    for b, p in bots_out:
        for m in getattr(p, "alerts", []):
            notify.send(m, log)
    try:
        fng = news.fear_greed()
    except Exception:  # noqa: BLE001
        fng = None
    summary = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "prices": {s: round(v, 4) for s, v in live.items()},
        "fear_greed": fng,
        "coins": coins,
        "usd_pkr": market.usd_pkr(),
        "telegram": notify.enabled(),
        "interval_min": 15,
        "errors": errors,
        "bots": sorted([bot_summary(b, p, live) for b, p in bots_out], key=lambda x: -x["return_pct"]),
    }
    save_json(os.path.join(OUT, "summary.json"), summary)
    log.info(f"summary written, {len(bots_out)} bots, {len(errors)} errors")
    if not bots_out:
        sys.exit(1)


if __name__ == "__main__":
    main()
