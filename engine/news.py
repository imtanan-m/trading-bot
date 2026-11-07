"""
News & sentiment guard.

It does NOT predict prices from news (by the time a headline is public, big
players have already moved). It only BLOCKS NEW BUYS when:
  - the Crypto Fear & Greed Index shows extreme greed (market overheated), or
  - a recent headline mentions the coin (or Binance) together with a danger
    word like "hack", "exploit", "delist", "lawsuit".
Existing positions are protected by the stop-loss, not by news.

If the news sources can't be reached, the guard allows trading and logs a
warning (otherwise one dead RSS feed would stop the bot forever).
"""
import calendar
import re
import time

import requests

try:
    import feedparser
except ImportError:  # optional
    feedparser = None

COIN_WORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "BNB": ["bnb"],
    "XRP": ["xrp", "ripple"],
}
GLOBAL_WORDS = ["binance", "crypto market", "stablecoin", "tether", "usdt"]
FNG_URL = "https://api.alternative.me/fng/?limit=1"


class NewsGuard:
    def __init__(self, cfg: dict, logger):
        self.enabled = bool(cfg.get("enabled", True))
        self.fng_block = cfg.get("fear_greed_block_above")
        self.lookback_s = float(cfg.get("lookback_hours", 12)) * 3600
        self.feeds = cfg.get("rss_feeds", []) or []
        self.keywords = [k.lower() for k in cfg.get("danger_keywords", [])]
        self.log = logger
        self._fng = (0.0, None)          # (fetched_at, value)
        self._heads = (0.0, [])          # (fetched_at, [(ts, title)])

    def fear_greed(self):
        if time.time() - self._fng[0] < 3600:
            return self._fng[1]
        try:
            r = requests.get(FNG_URL, timeout=10)
            r.raise_for_status()
            val = int(r.json()["data"][0]["value"])
        except Exception as e:  # noqa: BLE001
            self.log.warning(f"Fear&Greed fetch failed ({e}); ignoring this filter for now")
            val = None
        self._fng = (time.time(), val)
        return val

    def headlines(self):
        if time.time() - self._heads[0] < 900:
            return self._heads[1]
        items = []
        if feedparser is None:
            self.log.warning("feedparser not installed; news headlines filter off")
        else:
            for url in self.feeds:
                try:
                    feed = feedparser.parse(url, request_headers={"User-Agent": "trading-bot/1.0"})
                    for e in feed.entries[:60]:
                        t = e.get("published_parsed") or e.get("updated_parsed")
                        ts = calendar.timegm(t) if t else time.time()
                        title = (e.get("title") or "") + " " + (e.get("summary") or "")[:300]
                        items.append((ts, title.lower()))
                except Exception as ex:  # noqa: BLE001
                    self.log.warning(f"RSS failed {url}: {ex}")
        self._heads = (time.time(), items)
        return items

    def check(self, symbol: str):
        """Return (ok_to_buy: bool, reason: str)."""
        if not self.enabled:
            return True, ""
        if self.fng_block is not None:
            v = self.fear_greed()
            if v is not None and v > float(self.fng_block):
                return False, f"Fear&Greed={v} (extreme greed)"
        base = symbol.split("/")[0].upper()
        words = COIN_WORDS.get(base, [base.lower()]) + GLOBAL_WORDS
        cutoff = time.time() - self.lookback_s
        for ts, text in self.headlines():
            if ts < cutoff:
                continue
            if any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words):
                hit = next((k for k in self.keywords
                            if re.search(r"\b" + re.escape(k) + r"(s|ed|ing|ers?)?\b", text)), None)
                if hit:
                    return False, f"news: '{hit}' in: {text[:90]}"
        return True, ""
