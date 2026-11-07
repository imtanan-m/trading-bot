"""Turn a portfolio state into numbers + a small equity curve for the dashboard."""
import pandas as pd


def equity_series(st):
    if not st["equity"]:
        return pd.Series(dtype=float)
    return pd.Series([e[1] for e in st["equity"]], index=pd.to_datetime([e[0] for e in st["equity"]], utc=True))


def downsample(eq: pd.Series, n=300):
    if len(eq) <= n:
        pts = eq
    else:
        step = max(1, len(eq) // n)
        pts = pd.concat([eq.iloc[::step], eq.iloc[[-1]]])
        pts = pts[~pts.index.duplicated(keep="last")]
    return [[t.isoformat(), round(float(v), 2)] for t, v in pts.items()]


def stats(st, capital):
    eq = equity_series(st)
    trades = st["trades"]
    wins = [t for t in trades if t["pnl"] > 0]
    out = {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "max_dd": round(float((eq / eq.cummax() - 1).min() * 100), 2) if len(eq) else 0.0,
        "realized_pnl": round(sum(t["pnl"] for t in trades), 2),
    }
    if len(eq) > 1:
        days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400, 1e-9)
        out["days"] = round(days, 1)
        if days > 60:
            out["cagr"] = round(((eq.iloc[-1] / capital) ** (365.25 / days) - 1) * 100, 1)
        m = eq.resample("ME").last().pct_change().dropna()
        if len(m):
            out["pos_months"] = round(float((m > 0).mean() * 100))
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    out["profit_factor"] = round(gross_w / gross_l, 2) if gross_l > 0 else None
    return out
