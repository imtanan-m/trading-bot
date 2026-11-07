"""Public Binance market data (no API key). data-api.binance.vision works from most
regions incl. cloud servers in the US, where api.binance.com is geo-blocked."""
import time

import pandas as pd
import requests

BASES = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com",
         "https://api-gcp.binance.com"]
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _get(path, params):
    last = None
    for base in BASES:
        for attempt in range(2):
            try:
                r = requests.get(base + path, params=params, timeout=15,
                                 headers={"User-Agent": "paper-trading-bot/1.0"})
                if r.status_code == 200:
                    return r.json()
                last = f"{base} HTTP {r.status_code}: {r.text[:120]}"
                if r.status_code in (403, 451):
                    break  # geo-blocked: try next base
            except Exception as e:  # noqa: BLE001
                last = f"{base}: {e}"
            time.sleep(1 + attempt)
    raise RuntimeError(f"Binance data unavailable ({last})")


def klines(symbol, timeframe, limit=1000):
    """Closed candles only, as a DataFrame indexed by UTC open time."""
    rows = _get("/api/v3/klines", {"symbol": symbol.replace("/", ""), "interval": timeframe, "limit": limit})
    df = pd.DataFrame([r[:6] + [r[6]] for r in rows],
                      columns=["ts", "open", "high", "low", "close", "volume", "close_time"])
    now_ms = int(time.time() * 1000)
    df = df[df.close_time.astype("int64") < now_ms]
    df["ts"] = pd.to_datetime(df.ts.astype("int64"), unit="ms", utc=True)
    df = df.drop(columns="close_time").set_index("ts").astype(float)
    return df


def prices(symbols):
    out = {}
    for s in symbols:
        try:
            out[s] = float(_get("/api/v3/ticker/price", {"symbol": s.replace("/", "")})["price"])
        except Exception:  # noqa: BLE001
            pass
    return out


def history(symbol, timeframe, start="2022-01-01"):
    """All closed candles since `start` (paginated)."""
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    frames = []
    while True:
        rows = _get("/api/v3/klines", {"symbol": symbol.replace("/", ""), "interval": timeframe,
                                       "startTime": since, "limit": 1000})
        if not rows:
            break
        frames.append(rows)
        since = rows[-1][0] + TF_MS[timeframe]
        if len(rows) < 1000:
            break
    rows = [r for f in frames for r in f]
    df = pd.DataFrame([r[:6] + [r[6]] for r in rows],
                      columns=["ts", "open", "high", "low", "close", "volume", "close_time"])
    df = df[df.close_time.astype("int64") < int(time.time() * 1000)]
    df["ts"] = pd.to_datetime(df.ts.astype("int64"), unit="ms", utc=True)
    return df.drop(columns="close_time").set_index("ts").astype(float)


def usd_pkr():
    """USD -> PKR rate (free, no key). None if unavailable."""
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        return round(float(r.json()["rates"]["PKR"]), 2)
    except Exception:  # noqa: BLE001
        return None
