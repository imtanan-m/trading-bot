"""
All strategies (long-only, spot). Each one answers the same questions so the
same Portfolio engine can run any of them, in backtests and in the cloud.

  prepare(df)                -> df with indicator columns + 'warm'
  entry_signal(row)          -> bool
  initial_stop(entry, row)   -> price or None
  trail_stop(pos, row)       -> new stop price or None
  exit_signal(row, pos)      -> bool (pos['bars'] = candles held)
  sizing                     -> "risk" (risk % per trade) or "alloc" (fixed share of equity)
"""
import math

import pandas as pd

from .indicators import adx, atr, ema


def _ok(*vals):
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-12))


class Base:
    sizing = "risk"
    alloc = 0.5
    min_bars = 1

    def __init__(self, **p):
        self.p = p

    def _warm(self, out):
        out["warm"] = False
        if len(out) >= self.min_bars:
            out.iloc[self.min_bars - 1:, out.columns.get_loc("warm")] = True
        return out

    def initial_stop(self, entry, row):
        return None

    def trail_stop(self, pos, row):
        return None

    def exit_signal(self, row, pos):
        return False


class TrendBreakout(Base):
    """Breakout to a new N-candle high inside an established uptrend, ATR trailing stop.
    Research: time-series momentum (Liu & Tsyvinski NBER w24877; Cheah et al. 2022)."""

    def __init__(self, ema_fast=50, ema_slow=200, breakout_lookback=20, adx_period=14, adx_min=20,
                 atr_period=14, initial_stop_atr=2.5, trail_stop_atr=3.5, shock_candle_atr=3.0):
        super().__init__()
        self.ef, self.es, self.lb = int(ema_fast), int(ema_slow), int(breakout_lookback)
        self.adx_n, self.adx_min, self.atr_n = int(adx_period), float(adx_min), int(atr_period)
        self.k0, self.kt, self.shock = float(initial_stop_atr), float(trail_stop_atr), float(shock_candle_atr)
        self.min_bars = max(self.es * 2, self.lb + 5, 60)

    def prepare(self, df):
        o = df.copy()
        o["ema_fast"], o["ema_slow"] = ema(o.close, self.ef), ema(o.close, self.es)
        o["atr"], o["adx"] = atr(o, self.atr_n), adx(o, self.adx_n)
        o["hh"] = o.high.rolling(self.lb).max().shift(1)
        o["shock"] = (o.high - o.low) > self.shock * o["atr"].shift(1) if self.shock > 0 else False
        return self._warm(o)

    def entry_signal(self, r):
        if not r["warm"] or not _ok(r["hh"], r["ema_fast"], r["ema_slow"], r["adx"], r["atr"]):
            return False
        return bool(r.close > r.hh and r.close > r.ema_slow and r.ema_fast > r.ema_slow
                    and r.adx >= self.adx_min and not bool(r["shock"]))

    def initial_stop(self, entry, r):
        return entry - self.k0 * r.atr

    def trail_stop(self, pos, r):
        return max(pos["stop"], pos["hi"] - self.kt * r.atr)

    def exit_signal(self, r, pos):
        return bool(_ok(r.ema_slow) and r.close < r.ema_slow)


class DailyMomentum(Base):
    """Hold a coin while its daily trend is up (close > SMA and N-day return > 0).
    Equal share of equity per coin. Classic time-series momentum."""
    sizing = "alloc"

    def __init__(self, sma=50, lookback=30, stop_pct=0.0, n_coins=3):
        super().__init__()
        self.sma, self.lb, self.stop_pct = int(sma), int(lookback), float(stop_pct)
        self.alloc = 0.97 / int(n_coins)
        self.min_bars = max(self.sma, self.lb) + 2

    def prepare(self, df):
        o = df.copy()
        o["sma"] = o.close.rolling(self.sma).mean()
        o["mom"] = o.close / o.close.shift(self.lb) - 1
        return self._warm(o)

    def entry_signal(self, r):
        return bool(r["warm"] and _ok(r.sma, r.mom) and r.close > r.sma and r.mom > 0)

    def initial_stop(self, entry, r):
        return entry * (1 - self.stop_pct) if self.stop_pct > 0 else None

    def exit_signal(self, r, pos):
        return bool(_ok(r.sma, r.mom) and (r.close < r.sma or r.mom < 0))


class RSI2Scalper(Base):
    """Fast day-trading: buy short dips (RSI-2 < 10) inside an uptrend, sell on the bounce
    or after 24 candles. Many trades -> fees matter a lot (backtest: loses after fees)."""

    def __init__(self, ema_trend=200, rsi_n=2, rsi_buy=10, exit_ema=5, max_bars=24, stop_atr=2.0):
        super().__init__()
        self.et, self.rn, self.rb, self.xe = int(ema_trend), int(rsi_n), float(rsi_buy), int(exit_ema)
        self.max_bars, self.k = int(max_bars), float(stop_atr)
        self.min_bars = self.et * 2

    def prepare(self, df):
        o = df.copy()
        o["ema_t"], o["ema_x"] = ema(o.close, self.et), ema(o.close, self.xe)
        o["rsi"], o["atr"] = rsi(o.close, self.rn), atr(o, 14)
        return self._warm(o)

    def entry_signal(self, r):
        return bool(r["warm"] and _ok(r.ema_t, r.rsi, r.atr) and r.close > r.ema_t and r.rsi < self.rb)

    def initial_stop(self, entry, r):
        return entry - self.k * r.atr

    def exit_signal(self, r, pos):
        return bool(r.close > r.ema_x or pos["bars"] >= self.max_bars)


class BuyHold(Base):
    """Benchmark: buy once, never sell."""
    sizing = "alloc"

    def __init__(self, n_coins=2):
        super().__init__()
        self.alloc = 0.99 / int(n_coins)

    def prepare(self, df):
        return self._warm(df.copy())

    def entry_signal(self, r):
        return True


REGISTRY = {"trend": TrendBreakout, "momentum": DailyMomentum, "rsi2": RSI2Scalper, "hold": BuyHold}


def build(kind, params):
    return REGISTRY[kind](**(params or {}))
