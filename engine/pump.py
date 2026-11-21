"""Pump Radar signals. Every value uses only data up to (and including) that day's close."""
import numpy as np
import pandas as pd

FEATURES = ["vspike", "vtrend", "ret1", "ret7", "ret30", "rs7", "dd90", "range30", "brk60", "green", "lqvol", "age"]

NAMES = {
    "vspike": "Aaj ka volume (30 din ke average ka guna)",
    "vtrend": "7 din ka volume / 30 din ka volume",
    "ret1": "Aaj ka badlav",
    "ret7": "7 din ka badlav",
    "ret30": "30 din ka badlav",
    "rs7": "7 din mein BTC se aage",
    "dd90": "90 din ke high se faasla",
    "range30": "30 din ki range (kam = khamosh base)",
    "brk60": "60 din ka high toota",
    "green": "Lagataar hare din",
    "lqvol": "Liquidity (volume ka size)",
    "age": "Listing ko kitne din",
}


def features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    qv = c * df["volume"]
    qv30 = qv.rolling(30).mean().shift(1)            # average BEFORE today
    f = pd.DataFrame(index=df.index)
    f["vspike"] = qv / qv30
    f["vtrend"] = qv.rolling(7).mean() / qv30
    f["ret1"] = c / c.shift(1) - 1
    f["ret7"] = c / c.shift(7) - 1
    f["ret30"] = c / c.shift(30) - 1
    b7 = (btc["close"] / btc["close"].shift(7) - 1).reindex(df.index)
    f["rs7"] = f["ret7"] - b7
    f["dd90"] = c / h.rolling(90, min_periods=30).max() - 1
    prev = c.shift(1)
    f["range30"] = (prev.rolling(30).max() - prev.rolling(30).min()) / prev.rolling(30).mean()
    f["brk60"] = (c > h.rolling(60, min_periods=30).max().shift(1)).astype(float)
    up = (c > c.shift(1)).astype(int)
    f["green"] = up.groupby((up == 0).cumsum()).cumsum().clip(upper=10)
    f["lqvol"] = np.log10(qv30.clip(lower=1))
    f["age"] = np.log10(pd.Series(np.arange(1, len(df) + 1), index=df.index))
    f["qvol30"] = qv30
    return f.replace([np.inf, -np.inf], np.nan)


# same definitions as research/pump_research.py -> their historical hit rates are shown next to each pick
RULES = [
    ("spike 3x + breakout", "Volume 3x + 60-din breakout", lambda r: r["vspike"] >= 3 and r["brk60"] >= 1),
    ("quiet base then breakout", "Khamosh base ke baad breakout", lambda r: r["range30"] <= 0.35 and r["brk60"] >= 1),
    ("big day +20%", "Aaj +20% se zyada", lambda r: r["ret1"] >= 0.20),
    ("strong vs BTC (7d +30%)", "7 din mein BTC se 30% aage", lambda r: r["rs7"] >= 0.30),
    ("60-day breakout", "60-din ka high toota", lambda r: r["brk60"] >= 1),
    ("volume spike 3x", "Volume aaj 3x", lambda r: r["vspike"] >= 3),
    ("volume 7d/30d >= 2", "Hafte bhar volume 2x", lambda r: r["vtrend"] >= 2),
]


def score(row: dict, model: dict) -> float:
    """Model probability-style score 0..100 (higher = looks more like coins that pumped before)."""
    z = 0.0
    for k, w in zip(model["features"], model["w"]):
        v = (row[k] - model["mean"][k]) / (model["std"][k] or 1)
        z += w * max(-5, min(5, v))
    z += model["b"]
    return 100 / (1 + np.exp(-z))


def reasons(row: dict) -> list:
    r = []
    if row["vspike"] >= 3:
        r.append(f"Volume aaj {row['vspike']:.1f}x")
    elif row["vtrend"] >= 2:
        r.append(f"Volume hafte bhar se {row['vtrend']:.1f}x")
    if row["brk60"] >= 1:
        r.append("60 din ka high toota")
    if row["range30"] <= 0.35:
        r.append("Lambi khamoshi (tight base)")
    if row["rs7"] >= 0.15:
        r.append(f"BTC se {row['rs7']*100:.0f}% aage (7 din)")
    if row["ret1"] >= 0.15:
        r.append(f"Aaj +{row['ret1']*100:.0f}%")
    if row["green"] >= 4:
        r.append(f"{int(row['green'])} din lagataar hare")
    if row["dd90"] <= -0.6:
        r.append("90 din ke high se 60%+ neeche (sasta)")
    return r


# ---------------------------------------------------------------- live radar (called from run_cycle)
def run_radar(market, model, state_dir, out_path, log, top=25, log_top=10, research=None):
    """Once a day: scan all USDT coins, score them, save picks + an honest track record."""
    import json
    import os
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prev = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            prev = json.load(fh)
    if prev.get("date") == today:
        return prev
    info = market._get("/api/v3/exchangeInfo", {"permissions": "SPOT"})
    coins = [f"{s['baseAsset']}/USDT" for s in info["symbols"]
             if s["status"] == "TRADING" and s["quoteAsset"] == "USDT" and s["baseAsset"] not in market.STABLE
             and not market.is_stock_token(s["baseAsset"]) and not s["baseAsset"].endswith(("UP", "DOWN", "BULL", "BEAR"))]
    btc = market.klines("BTC/USDT", "1d", 200)
    rows, closes = [], {}
    for c in coins:
        try:
            df = market.klines(c, "1d", 120)
        except Exception:  # noqa: BLE001
            continue
        if len(df) < 35:
            continue
        closes[c] = df
        f = features(df, btc).iloc[-1]
        if f[FEATURES].isna().any() or f["qvol30"] < 200_000:
            continue
        r = {k: float(f[k]) for k in FEATURES}
        tags = [key for key, _, fn in RULES if fn(r)]
        rows.append({"tags": tags, "coin": c, "score": round(float(score(r, model)), 1), "price": float(df["close"].iloc[-1]),
                     "ret1": round(r["ret1"] * 100, 1), "ret7": round(r["ret7"] * 100, 1), "ret30": round(r["ret30"] * 100, 1),
                     "vspike": round(r["vspike"], 2), "vtrend": round(r["vtrend"], 2), "qvol30": round(float(f["qvol30"])),
                     "reasons": reasons(r)})
    strong = {"spike 3x + breakout", "quiet base then breakout"}
    rows.sort(key=lambda x: (-len(strong & set(x["tags"])), -len(x["tags"]), -x["score"]))
    # ---- track record: every day's top picks, checked against what really happened ----
    lpath = os.path.join(state_dir, "pump_log.json")
    plog = []
    if os.path.exists(lpath):
        with open(lpath, encoding="utf-8") as fh:
            plog = json.load(fh)
    for i, r in enumerate(rows[:log_top]):
        plog.append({"date": today, "rank": i + 1, "coin": r["coin"], "price": r["price"], "score": r["score"]})
    for p in plog:
        df = closes.get(p["coin"])
        if df is None or p["date"] >= today:
            continue
        since = df[df.index > pd.Timestamp(p["date"], tz="UTC")]
        win = since.iloc[:30]
        if len(win):
            p["max_gain"] = round((float(win["high"].max()) / p["price"] - 1) * 100, 1)
            p["now"] = round((float(since["close"].iloc[-1]) / p["price"] - 1) * 100, 1)
            p["days"] = len(since)
    plog = plog[-3000:]
    os.makedirs(state_dir, exist_ok=True)
    with open(lpath, "w", encoding="utf-8") as fh:
        json.dump(plog, fh, separators=(",", ":"))
    ev = [p for p in plog if "max_gain" in p]
    track = {"picks": len(ev), "since": plog[0]["date"] if plog else today}
    if ev:
        track.update({
            "hit20": round(sum(p["max_gain"] >= 20 for p in ev) / len(ev) * 100, 1),
            "hit50": round(sum(p["max_gain"] >= 50 for p in ev) / len(ev) * 100, 1),
            "hit100": round(sum(p["max_gain"] >= 100 for p in ev) / len(ev) * 100, 1),
            "avg_now": round(sum(p["now"] for p in ev) / len(ev), 1),
            "best": sorted(ev, key=lambda p: -p["max_gain"])[:5],
            "recent": sorted(ev, key=lambda p: (p["date"], -p["rank"]), reverse=True)[:30]})
    out = {"date": today, "generated": datetime.now(timezone.utc).isoformat(), "scanned": len(coins),
           "liquid": len(rows), "picks": rows[:top], "model": model.get("test", {}),
           "base_rate": model.get("base_rate", {}), "track": track, "names": NAMES,
           "rules": {key: dict(label=lbl, **next((x for x in (research or {}).get("rules", []) if x["rule"] == key), {}))
                     for key, lbl, _ in RULES}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    log.info(f"pump radar: {len(coins)} coins scanned, {len(rows)} liquid, top {rows[0]['coin'] if rows else '-'}")
    return out
