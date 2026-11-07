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

from engine import market
from engine.news import NewsGuard
from engine.portfolio import Portfolio, run_series
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


def run_bot(b, cfg, news, cache):
    strat = build(b["kind"], b.get("params"))
    path = os.path.join(STATE_DIR, f"{b['id']}.json")
    state = load_json(path)
    p = Portfolio(strat, b.get("risk", {}), float(b["capital"]), cfg.get("fee_pct", 0.1),
                  cfg.get("slippage_pct", 0.05), state)
    prepared = {}
    for sym in b["symbols"]:
        key = (sym, b["timeframe"])
        if key not in cache:
            cache[key] = market.klines(sym, b["timeframe"], 1000)
        prepared[sym] = strat.prepare(cache[key])
    start = None
    if state is None:  # brand-new bot: start trading from the latest closed candle, not from history
        start = min(df.index[-1] for df in prepared.values())
        log.info(f"{b['id']}: new bot, starting at {start}")
    flt = news.check if b.get("news_filter") else None
    n_before = len(p.st["trades"])
    run_series(p, prepared, start=start, entry_filter_last=flt)
    p.st["events"] = p.st.get("events", [])[-50:]
    save_json(path, p.st)
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
        positions.append({"symbol": sym, "qty": round(pos["qty"], 6), "entry": round(pos["entry"], 4),
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
        "trades": st["trades"][-30:][::-1], "events": st.get("events", [])[-10:][::-1],
        "stats": stats(st, cap), "curve": downsample(eq, 300),
    }


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "bots.yaml"), encoding="utf-8"))
    news = NewsGuard(NEWS_CFG, log)
    cache, bots_out, errors = {}, [], []
    all_syms = sorted({s for b in cfg["bots"] for s in b["symbols"]})
    for b in cfg["bots"]:
        try:
            p = run_bot(b, cfg, news, cache)
            bots_out.append((b, p))
        except Exception as e:  # noqa: BLE001 - one broken bot must not stop the others
            log.error(f"{b['id']} failed: {e}\n{traceback.format_exc()}")
            errors.append(f"{b['name']}: {e}")
    live = market.prices(all_syms) if bots_out else {}
    try:
        fng = news.fear_greed()
    except Exception:  # noqa: BLE001
        fng = None
    summary = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "prices": {s: round(v, 4) for s, v in live.items()},
        "fear_greed": fng,
        "errors": errors,
        "bots": sorted([bot_summary(b, p, live) for b, p in bots_out], key=lambda x: -x["return_pct"]),
    }
    save_json(os.path.join(OUT, "summary.json"), summary)
    log.info(f"summary written, {len(bots_out)} bots, {len(errors)} errors")
    if not bots_out:
        sys.exit(1)


if __name__ == "__main__":
    main()
