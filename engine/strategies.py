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
    cross = False  # True = compare all coins and buy the best-scoring ones

    def take_profit(self, entry, row):
        return None

    def score(self, row):
        return 0.0

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

    def explain(self, row):
        """List of entry conditions for the dashboard 'signal radar'."""
        return []


def _pct(a, b):
    return round((a / b - 1) * 100, 2) if _ok(a, b) and b else None


def _cond(label, ok, value, hint=""):
    return {"label": label, "ok": bool(ok), "value": value, "hint": hint}


class TrendBreakout(Base):
    """Breakout to a new N-candle high inside an established uptrend, ATR trailing stop.
    Research: time-series momentum (Liu & Tsyvinski NBER w24877; Cheah et al. 2022)."""

    def __init__(self, ema_fast=50, ema_slow=200, breakout_lookback=20, adx_period=14, adx_min=20,
                 atr_period=14, initial_stop_atr=2.5, trail_stop_atr=3.5, shock_candle_atr=3.0,
                 take_profit_atr=0.0, rank=False):
        super().__init__()
        self.tp_atr, self.cross = float(take_profit_atr), bool(rank)
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
        o["strength"] = (o.close / o.close.shift(self.lb) - 1) / (o["atr"] / o.close)
        return self._warm(o)

    def score(self, r):
        return r["strength"] if _ok(r["strength"]) else 0.0

    def take_profit(self, entry, r):
        return entry + self.tp_atr * r["atr"] if self.tp_atr > 0 else None

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

    def explain(self, r):
        if not _ok(r.hh, r.ema_slow, r.adx):
            return []
        return [
            _cond(f"Price EMA-{self.es} se upar", r.close > r.ema_slow, _pct(r.close, r.ema_slow), "bara trend"),
            _cond(f"EMA-{self.ef} > EMA-{self.es}", r.ema_fast > r.ema_slow, _pct(r.ema_fast, r.ema_slow), "trend ki tasdeeq"),
            _cond(f"{self.lb} candles ka high toota", r.close > r.hh, _pct(r.close, r.hh), "breakout"),
            _cond(f"ADX {self.adx_min:g}+ (trend mein jaan)", r.adx >= self.adx_min, round(float(r.adx), 1), ""),
        ]


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

    def explain(self, r):
        if not _ok(r.sma, r.mom):
            return []
        return [
            _cond(f"Price {self.sma}-din average se upar", r.close > r.sma, _pct(r.close, r.sma), ""),
            _cond(f"{self.lb} din ka return positive", r.mom > 0, round(float(r.mom) * 100, 2), ""),
        ]


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

    def explain(self, r):
        if not _ok(r.ema_t, r.rsi):
            return []
        return [
            _cond(f"Price EMA-{self.et} se upar", r.close > r.ema_t, _pct(r.close, r.ema_t), "uptrend"),
            _cond(f"RSI-{self.rn} {self.rb:g} se neeche (dip)", r.rsi < self.rb, round(float(r.rsi), 1), ""),
        ]


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


class Rotation(Base):
    """Opportunity Hunter (daily): scan many coins, rank by risk-adjusted momentum and hold the
    top K that are in an uptrend. Cross-sectional momentum is one of the strongest documented
    crypto factors (Liu, Tsyvinski & Wu 2022, Journal of Finance)."""
    sizing = "alloc"
    cross = True

    def __init__(self, lookback=28, sma=20, k=3, take_profit_pct=0.0, stop_pct=0.0):
        super().__init__()
        self.lb, self.sma, self.k = int(lookback), int(sma), int(k)
        self.tp, self.sl = float(take_profit_pct), float(stop_pct)
        self.alloc = 0.97 / self.k
        self.min_bars = max(self.lb, self.sma) + 2

    def prepare(self, df):
        o = df.copy()
        o["sma"] = o.close.rolling(self.sma).mean()
        o["mom"] = o.close / o.close.shift(self.lb) - 1
        vol = o.close.pct_change().rolling(self.lb).std()
        o["score"] = o["mom"] / (vol * (self.lb ** 0.5))
        return self._warm(o)

    def entry_signal(self, r):
        return bool(r["warm"] and _ok(r["sma"], r["mom"], r["score"]) and r["close"] > r["sma"] and r["mom"] > 0)

    def score(self, r):
        return r["score"]

    def initial_stop(self, entry, r):
        return entry * (1 - self.sl) if self.sl > 0 else None

    def take_profit(self, entry, r):
        return entry * (1 + self.tp) if self.tp > 0 else None

    def exit_signal(self, r, pos):
        return bool(_ok(r["sma"], r["mom"]) and (r["close"] < r["sma"] or r["mom"] < 0))

    def explain(self, r):
        if not _ok(r["sma"], r["mom"]):
            return []
        return [_cond(f"Price {self.sma}-candle average se upar", r["close"] > r["sma"], _pct(r["close"], r["sma"])),
                _cond(f"{self.lb} candles ka return positive", r["mom"] > 0, round(float(r["mom"]) * 100, 2))]


class DipBuyer(Base):
    """Buy sharp oversold dips (RSI low) and sell quickly at a profit target, with a stop and a
    time limit. Picks the most oversold coins first."""
    sizing = "alloc"
    cross = True

    def __init__(self, rsi_n=14, rsi_buy=30, trend_sma=0, take_profit_pct=0.08, stop_pct=0.08, max_bars=10, k=3):
        super().__init__()
        self.rn, self.rb, self.tsma = int(rsi_n), float(rsi_buy), int(trend_sma)
        self.tp, self.sl, self.max_bars, self.k = float(take_profit_pct), float(stop_pct), int(max_bars), int(k)
        self.alloc = 0.97 / self.k
        self.min_bars = max(self.rn * 3, self.tsma) + 2

    def prepare(self, df):
        o = df.copy()
        o["rsi"] = rsi(o.close, self.rn)
        o["tsma"] = o.close.rolling(self.tsma).mean() if self.tsma else 0.0
        return self._warm(o)

    def entry_signal(self, r):
        if not (r["warm"] and _ok(r["rsi"])):
            return False
        trend_ok = (not self.tsma) or (_ok(r["tsma"]) and r["close"] > r["tsma"])
        return bool(r["rsi"] < self.rb and trend_ok)

    def score(self, r):
        return -r["rsi"]

    def initial_stop(self, entry, r):
        return entry * (1 - self.sl)

    def take_profit(self, entry, r):
        return entry * (1 + self.tp)

    def exit_signal(self, r, pos):
        return pos["bars"] >= self.max_bars

    def explain(self, r):
        if not _ok(r["rsi"]):
            return []
        c = [_cond(f"RSI-{self.rn} {self.rb:g} se neeche (oversold)", r["rsi"] < self.rb, round(float(r["rsi"]), 1))]
        if self.tsma:
            c.append(_cond(f"Price {self.tsma}-candle average se upar", _ok(r["tsma"]) and r["close"] > r["tsma"], _pct(r["close"], r["tsma"])))
        return c


REGISTRY = {"trend": TrendBreakout, "momentum": DailyMomentum, "rsi2": RSI2Scalper, "hold": BuyHold,
            "rotation": Rotation, "dip": DipBuyer}


def build(kind, params):
    return REGISTRY[kind](**(params or {}))
