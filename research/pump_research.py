"""
Pump research: can early signals tell us which coin will pump soon?  (runs on GitHub Actions)

Universe: ALL Binance USDT spot coins (no stablecoins / stock tokens / leveraged tokens), daily candles.
For every coin and every day we compute signals using ONLY data up to that day's close, and the label
"did the price touch +50% (or +100%) within the next 30 days?".  A model is trained on data before
2025-06-01 and judged only on data after that (test), against the base rate (picking coins at random).

Output: research/pump_results.json  (+ research/pump_model.json used by the live Pump Radar)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import market  # noqa: E402
from engine.pump import FEATURES, features  # noqa: E402

DATA = os.path.join(ROOT, "data", "daily_all")
START = "2023-01-01"
SPLIT = pd.Timestamp("2025-06-01", tz="UTC")


def all_usdt_coins():
    info = market._get("/api/v3/exchangeInfo", {"permissions": "SPOT"})
    out = []
    for s in info["symbols"]:
        b = s["baseAsset"]
        if s["status"] != "TRADING" or s["quoteAsset"] != "USDT":
            continue
        if b in market.STABLE or market.is_stock_token(b) or b.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        out.append(f"{b}/USDT")
    return out


def load(sym):
    path = os.path.join(DATA, sym.replace("/", "_") + ".csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if len(df) and df.index[-1] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2):
            return df.astype(float)
    df = market.history(sym, "1d", START)
    os.makedirs(DATA, exist_ok=True)
    df.reset_index().to_csv(path, index=False)
    return df


def labels(df):
    hi = df["high"]
    fut_max30 = hi[::-1].rolling(30, min_periods=30).max()[::-1].shift(-1)
    fut_max7 = hi[::-1].rolling(7, min_periods=7).max()[::-1].shift(-1)
    return pd.DataFrame({
        "max30": fut_max30 / df["close"] - 1,
        "max7": fut_max7 / df["close"] - 1,
        "close30": df["close"].shift(-30) / df["close"] - 1,
    }, index=df.index)


def logistic_fit(X, y, l2=1.0, iters=400, lr=0.5):
    """Plain numpy logistic regression (no extra libraries needed live)."""
    w, b = np.zeros(X.shape[1]), 0.0
    pos = y.mean()
    sw = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))  # balance classes
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def main():
    t0 = time.time()
    coins = all_usdt_coins()
    print(f"{len(coins)} coins", flush=True)
    btc = load("BTC/USDT")
    rows = []
    for i, c in enumerate(coins):
        try:
            df = load(c)
        except Exception as e:  # noqa: BLE001
            print("skip", c, e)
            continue
        if len(df) < 120:
            continue
        f = features(df, btc)
        lab = labels(df)
        d = pd.concat([f, lab], axis=1)
        d["coin"] = c
        rows.append(d.dropna(subset=FEATURES))
        if i % 50 == 0:
            print(f"  {i}/{len(coins)} loaded, {time.time()-t0:.0f}s", flush=True)
    D = pd.concat(rows)
    D = D[D.index >= pd.Timestamp("2023-06-01", tz="UTC")]
    D = D[D["qvol30"] >= 200_000]  # ignore coins with < $200k daily volume (can't really trade them)
    full = D.dropna(subset=["max30"])
    print(f"rows {len(full)}; {time.time()-t0:.0f}s", flush=True)

    out = {"generated": pd.Timestamp.now(tz="UTC").isoformat(), "coins": len(coins), "rows": int(len(full))}
    base = {k: round(float((full["max30"] >= v).mean() * 100), 2) for k, v in (("p50", .5), ("p100", 1.0))}
    base["p30_7d"] = round(float((full["max7"] >= .3).mean() * 100), 2)
    base["avg_close30"] = round(float(full["close30"].mean() * 100), 2)
    out["base_rate"] = base
    print("BASE RATE (random coin-day):", base, flush=True)

    # ---- simple rules people use ----
    rules = {
        "volume spike 3x": full["vspike"] >= 3,
        "volume 7d/30d >= 2": full["vtrend"] >= 2,
        "60-day breakout": full["brk60"] >= 1,
        "spike 3x + breakout": (full["vspike"] >= 3) & (full["brk60"] >= 1),
        "quiet base then breakout": (full["range30"] <= 0.35) & (full["brk60"] >= 1),
        "big day +20%": full["ret1"] >= 0.20,
        "strong vs BTC (7d +30%)": full["rs7"] >= 0.30,
    }
    out["rules"] = []
    for name, m in rules.items():
        sub = full[m]
        r = {"rule": name, "signals": int(len(sub)),
             "hit50_30d": round(float((sub["max30"] >= .5).mean() * 100), 2) if len(sub) else None,
             "hit100_30d": round(float((sub["max30"] >= 1).mean() * 100), 2) if len(sub) else None,
             "avg_close30": round(float(sub["close30"].mean() * 100), 2) if len(sub) else None,
             "median_close30": round(float(sub["close30"].median() * 100), 2) if len(sub) else None}
        out["rules"].append(r)
        print(r, flush=True)

    # ---- model: train before SPLIT, test after ----
    tr, te = full[full.index < SPLIT], full[full.index >= SPLIT]
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
    Xtr, Xte = ((tr[FEATURES] - mu) / sd).clip(-5, 5).values, ((te[FEATURES] - mu) / sd).clip(-5, 5).values
    ytr, yte = (tr["max30"] >= .5).astype(int).values, (te["max30"] >= .5).astype(int).values
    w, b = logistic_fit(Xtr, ytr)
    ste = Xte @ w + b
    te = te.assign(score=ste)
    res = {"auc_test": auc(ste, yte), "auc_train": auc(Xtr @ w + b, ytr),
           "test_base_p50": round(float(yte.mean() * 100), 2)}
    # top-10 per day in the test period
    top = te.sort_values("score", ascending=False).groupby(level=0).head(10)
    res["top10_hit50"] = round(float((top["max30"] >= .5).mean() * 100), 2)
    res["top10_hit100"] = round(float((top["max30"] >= 1).mean() * 100), 2)
    res["top10_avg_close30"] = round(float(top["close30"].mean() * 100), 2)
    res["top10_median_close30"] = round(float(top["close30"].median() * 100), 2)
    res["top10_avg_max30"] = round(float(top["max30"].mean() * 100), 2)
    # deciles
    te["dec"] = pd.qcut(te["score"].rank(method="first"), 10, labels=False)
    res["deciles"] = [{"decile": int(k) + 1, "hit50": round(float((g["max30"] >= .5).mean() * 100), 2),
                       "avg_close30": round(float(g["close30"].mean() * 100), 2)} for k, g in te.groupby("dec")]
    # simulate: buy top-3 each week, sell after 30 days or at +50% target / -25% stop
    sims = []
    weeks = sorted(set(te.index.normalize()))[::7]
    for d in weeks:
        day = te[te.index == d].sort_values("score", ascending=False).head(3)
        for _, r in day.iterrows():
            hit = r["max30"] >= .5
            sims.append(0.5 if hit else max(r["close30"], -0.25))
    res["weekly_top3_trades"] = len(sims)
    res["weekly_top3_avg_trade"] = round(float(np.mean(sims) * 100), 2) if sims else None
    res["weekly_top3_win_rate"] = round(float((np.array(sims) > 0).mean() * 100), 2) if sims else None
    res["weights"] = {f: round(float(x), 3) for f, x in zip(FEATURES, w)}
    out["model"] = res
    print("MODEL (test period, never seen in training):", json.dumps(res, indent=1), flush=True)

    # biggest pumps in the test period: did the radar see them coming?
    big = te[te["max30"] >= 2].sort_values("max30", ascending=False).groupby("coin").head(1).head(15)
    pct_rank = te.groupby(level=0)["score"].rank(pct=True)
    out["big_pumps"] = [{"coin": r.coin, "date": str(i.date()), "max30": round(float(r.max30) * 100),
                         "score_pct": round(float(pct_rank.loc[i][te.loc[i, "coin"] == r.coin].iloc[0]) * 100)
                         if isinstance(pct_rank.loc[i], pd.Series) else None}
                        for i, r in big.iterrows()]
    print("BIG PUMPS:", out["big_pumps"], flush=True)

    model = {"features": FEATURES, "mean": mu.round(6).to_dict(), "std": sd.round(6).to_dict(),
             "w": [float(x) for x in w], "b": float(b), "trained_until": str(SPLIT.date()),
             "test": {k: res[k] for k in ("auc_test", "test_base_p50", "top10_hit50", "top10_hit100",
                                          "top10_avg_close30", "top10_median_close30")},
             "base_rate": base}
    with open(os.path.join(ROOT, "research", "pump_model.json"), "w") as fh:
        json.dump(model, fh, indent=1)
    with open(os.path.join(ROOT, "research", "pump_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
