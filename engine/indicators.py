"""Technical indicators (pure pandas, no extra libraries)."""
import numpy as np
import pandas as pd


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder's Average True Range."""
    return true_range(df).ewm(alpha=1.0 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder's Average Directional Index (trend strength, 0-100)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = true_range(df).ewm(alpha=1.0 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / tr_s
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / tr_s
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / n, adjust=False).mean()
