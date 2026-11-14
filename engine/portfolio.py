"""
Paper portfolio that is exactly the same in backtests and in the cloud:
  - decisions on CLOSED candles, orders fill at the NEXT candle's open
  - stop-losses fill intrabar at the stop price (or worse on a gap)
  - fee + slippage on every fill
The whole state is a plain dict, so it can be saved to JSON between cloud runs.
"""
from datetime import datetime, timezone


class Row(dict):
    """Fast candle row: row["close"] and row.close both work."""
    __getattr__ = dict.__getitem__


class Portfolio:
    def __init__(self, strategy, risk: dict, capital: float, fee_pct=0.1, slippage_pct=0.05, state=None):
        self.s = strategy
        self.fee = fee_pct / 100
        self.slip = slippage_pct / 100
        self.risk_pt = float(risk.get("risk_per_trade_pct", 0.75)) / 100
        self.max_pos = float(risk.get("max_position_pct", 45)) / 100
        self.max_open = int(risk.get("max_open_positions", 3))
        self.reserve = float(risk.get("min_cash_reserve_pct", 2)) / 100
        self.daily_lim = float(risk.get("daily_loss_limit_pct", 3.0)) / 100
        self.kill_dd = float(risk.get("max_drawdown_kill_pct", 25.0)) / 100
        self.st = state or {
            "capital": capital, "cash": capital, "positions": {}, "pend_entry": {}, "pend_exit": [],
            "last_ts": {}, "last_close": {}, "peak": capital, "day": "", "day_start": capital,
            "killed": None, "trades": [], "equity": [], "created": _now(),
        }

    # ---------------------------------------------------------------- helpers
    def equity(self):
        st = self.st
        return st["cash"] + sum(p["qty"] * st["last_close"].get(s, p["entry"]) for s, p in st["positions"].items())

    def _size(self, entry, stop):
        eq, cash = self.equity(), self.st["cash"]
        spendable = max(cash - eq * self.reserve, 0) / (entry * (1 + self.fee))
        if self.s.sizing == "alloc":
            return max(0.0, min(eq * self.s.alloc / (entry * (1 + self.fee)), spendable))
        if stop is None or stop >= entry:
            return 0.0
        per_unit = (entry - stop) + entry * self.fee * 2
        return max(0.0, min(eq * self.risk_pt / per_unit, eq * self.max_pos / entry, spendable))

    def _close(self, sym, price, t, reason):
        st = self.st
        p = st["positions"].pop(sym)
        proceeds = p["qty"] * price * (1 - self.fee)
        st["cash"] += proceeds
        st["trades"].append({"symbol": sym, "entry_time": p["time"], "exit_time": _iso(t),
                             "entry": round(p["entry"], 6), "exit": round(price, 6),
                             "pnl": round(proceeds - p["cost"], 4),
                             "ret_pct": round((proceeds / p["cost"] - 1) * 100, 3), "reason": reason})

    # ---------------------------------------------------------------- core
    def process_bar(self, sym, t, bar, entry_filter=None, regime_ok=True):
        """bar = prepared row (open/high/low/close + indicators) of a CLOSED candle."""
        st = self.st
        day = _iso(t)[:10]
        if st["day"] != day:
            st["day"], st["day_start"] = day, self.equity()
        # 1) exit decided at previous close -> next open
        if sym in st["pend_exit"]:
            st["pend_exit"].remove(sym)
            if sym in st["positions"]:
                self._close(sym, bar.open * (1 - self.slip), t, "signal exit")
        # 2) entry decided at previous close -> next open
        if sym in st["pend_entry"]:
            st["pend_entry"].pop(sym)
            if sym not in st["positions"] and not st["killed"]:
                entry = bar.open * (1 + self.slip)
                stop = self.s.initial_stop(entry, self._sig[sym]) if sym in self._sig else None
                qty = self._size(entry, stop)
                cost = qty * entry * (1 + self.fee)
                if qty > 0 and cost >= 5:
                    st["cash"] -= cost
                    tp = self.s.take_profit(entry, self._sig[sym]) if sym in self._sig else None
                    st["positions"][sym] = {"qty": qty, "entry": entry, "stop": stop, "cost": cost,
                                            "hi": entry, "time": _iso(t), "bars": 0, "tp": tp}
        # 3) intrabar stop
        p = st["positions"].get(sym)
        if p and p["stop"] is not None and bar.low <= p["stop"]:
            self._close(sym, min(bar.open, p["stop"]) * (1 - self.slip), t, "stop-loss")
            p = None
        if p and p.get("tp") and bar.high >= p["tp"]:
            self._close(sym, max(bar.open, p["tp"]) * (1 - self.slip), t, "take-profit")
        st["last_close"][sym] = float(bar.close)
        st["last_ts"][sym] = _iso(t)
        # 4) decisions at this close
        p = st["positions"].get(sym)
        if p:
            p["bars"] += 1
            p["hi"] = max(p["hi"], float(bar.close))
            ns = self.s.trail_stop(p, bar)
            if ns is not None:
                p["stop"] = float(ns)
            if self.s.exit_signal(bar, p):
                st["pend_exit"].append(sym)
        elif not st["killed"] and regime_ok and self.s.entry_signal(bar):
            if getattr(self.s, "cross", False):  # multi-coin mode: rank all candidates first
                self._cands.append((float(self.s.score(bar)), sym, bar, entry_filter))
                return
            n_busy = len(st["positions"]) + len(st["pend_entry"])
            day_ok = st["day_start"] <= 0 or self.equity() / st["day_start"] - 1 > -self.daily_lim
            ok, why = (entry_filter(sym) if entry_filter else (True, ""))
            if n_busy < self.max_open and day_ok and ok:
                st["pend_entry"][sym] = _iso(t)
                self._sig[sym] = bar
            elif not ok:
                st.setdefault("events", []).append({"t": _iso(t), "msg": f"{sym}: buy blocked ({why})"})

    def mark(self, t, record=True):
        st = self.st
        eq = self.equity()
        st["peak"] = max(st["peak"], eq)
        if not st["killed"] and st["peak"] > 0 and eq / st["peak"] - 1 <= -self.kill_dd:
            st["killed"] = _iso(t)
            for sym in list(st["positions"]):
                self._close(sym, st["last_close"][sym] * (1 - self.slip), t, "KILL SWITCH")
            st["pend_entry"].clear()
            eq = self.equity()
        if record:
            st["equity"].append([_iso(t), round(eq, 4)])
        return eq

    def pick_candidates(self, t):
        """Multi-coin mode: buy the best-scoring candidates while there are free slots."""
        st = self.st
        cands, self._cands = sorted(self._cands, key=lambda c: -c[0]), []
        day_ok = st["day_start"] <= 0 or self.equity() / st["day_start"] - 1 > -self.daily_lim
        for score, sym, bar, flt in cands:
            if len(st["positions"]) + len(st["pend_entry"]) >= self.max_open or not day_ok:
                break
            ok, why = flt(sym) if flt else (True, "")
            if not ok:
                st.setdefault("events", []).append({"t": _iso(t), "msg": f"{sym}: buy blocked ({why})"})
                continue
            st["pend_entry"][sym] = _iso(t)
            self._sig[sym] = bar

    @property
    def _cands(self):
        if not hasattr(self, "_cand_list"):
            self._cand_list = []
        return self._cand_list

    @_cands.setter
    def _cands(self, v):
        self._cand_list = v

    @property
    def _sig(self):
        # signal rows are needed only between a decision and the next candle; kept in memory,
        # and rebuilt from the data when a cloud run restarts (see runner)
        if not hasattr(self, "_sig_rows"):
            self._sig_rows = {}
        return self._sig_rows


def run_series(portfolio, prepared: dict, start=None, entry_filter_last=None, regime=None):
    """Feed candles (dict sym->prepared df) in time order. Only candles after each symbol's
    last processed timestamp are used, so this works for backtests AND incremental cloud runs."""
    st = portfolio.st
    events = []
    for sym, df in prepared.items():
        last = st["last_ts"].get(sym)
        d = df[df.index > last] if last else df
        if start is not None:
            d = d[d.index >= start]
        # rebuild pending-entry signal row after a restart
        if sym in st["pend_entry"] and sym not in portfolio._sig and last:
            prev = df[df.index <= last]
            if len(prev):
                portfolio._sig[sym] = Row(prev.iloc[-1].to_dict())
        for t, rec in zip(d.index, d.to_dict("records")):
            events.append((t, sym, Row(rec)))
    events.sort(key=lambda e: (e[0], e[1]))
    latest = {s: (df.index[-1] if len(df) else None) for s, df in prepared.items()}
    i = 0
    while i < len(events):
        t = events[i][0]
        while i < len(events) and events[i][0] == t:
            _, sym, row = events[i]
            flt = entry_filter_last if (entry_filter_last and t == latest[sym]) else None
            portfolio.process_bar(sym, t, row, flt, regime(t) if regime else True)
            i += 1
        if portfolio._cands:
            portfolio.pick_candidates(t)
        portfolio.mark(t)
    return portfolio


def _iso(t):
    return t.isoformat() if hasattr(t, "isoformat") else str(t)


def _now():
    return datetime.now(timezone.utc).isoformat()


def btc_regime(btc_daily, sma=50):
    """Market filter: True when BTC's last CLOSED daily candle is above its N-day average.
    Returns f(t) usable as run_series(regime=...). No look-ahead: a daily candle only counts
    after it has closed."""
    import pandas as pd
    ok = (btc_daily["close"] > btc_daily["close"].rolling(sma).mean())
    ok.index = ok.index + pd.Timedelta(days=1)  # known at the candle's close
    idx, vals = ok.index, ok.values

    def f(t):
        i = idx.searchsorted(t, side="right") - 1
        return bool(vals[i]) if i >= 0 else False
    return f
