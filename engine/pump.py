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
