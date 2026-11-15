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
        # futures (USD-M perpetual, isolated margin): long + short, leverage, liquidation, funding
        self.fut = bool(getattr(strategy, "futures", False))
        self.lev = float(risk.get("leverage", 1)) if self.fut else 1.0
        self.mm = 0.005                                                    # maintenance margin
        self.funding = float(risk.get("funding_pct_8h", 0.01)) / 100 if self.fut else 0.0
        self.st = state or {
            "capital": capital, "cash": capital, "positions": {}, "pend_entry": {}, "pend_exit": [],
            "last_ts": {}, "last_close": {}, "peak": capital, "day": "", "day_start": capital,
            "killed": None, "trades": [], "equity": [], "created": _now(),
        }

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def pos_value(p, px):
        if "margin" not in p:  # spot
            return p["qty"] * px
        return max(0.0, p["margin"] + p["side"] * p["qty"] * (px - p["entry"]))

    def equity(self):
        st = self.st
        return st["cash"] + sum(self.pos_value(p, st["last_close"].get(s, p["entry"])) for s, p in st["positions"].items())

    def _size(self, entry, stop):
        eq, cash, L = self.equity(), self.st["cash"], self.lev
        free = max(cash - eq * self.reserve, 0)
        spendable = free / (entry * (1 / L + self.fee))          # margin + entry fee must fit in cash
        if self.s.sizing == "alloc":
            return max(0.0, min(eq * self.s.alloc * L / entry, spendable))
        if stop is None or stop == entry:
            return 0.0
        per_unit = abs(entry - stop) + entry * self.fee * 2
        return max(0.0, min(eq * self.risk_pt / per_unit, eq * self.max_pos * L / entry, spendable))

    def _liq_price(self, p):
        if "margin" not in p or self.lev <= 1:
            return None
        return p["entry"] * (1 - p["side"] * (1 / self.lev - self.mm))

    def _close(self, sym, price, t, reason):
        st = self.st
        p = st["positions"].pop(sym)
        if "margin" in p:  # futures
            hours = max((_ts(t) - _ts(p["time"])).total_seconds() / 3600, 0)
            gross = p["side"] * p["qty"] * (price - p["entry"])
            costs = p["qty"] * price * self.fee + p["qty"] * p["entry"] * self.funding * hours / 8
            proceeds = max(0.0, p["margin"] + gross - costs)
        else:
            proceeds = p["qty"] * price * (1 - self.fee)
        st["cash"] += proceeds
        tr = {"symbol": sym, "entry_time": p["time"], "exit_time": _iso(t),
              "entry": round(p["entry"], 6), "exit": round(price, 6),
              "pnl": round(proceeds - p["cost"], 4),
              "ret_pct": round((proceeds / p["cost"] - 1) * 100, 3), "reason": reason}
        if "margin" in p:
            tr["side"] = "long" if p["side"] > 0 else "short"
            tr["lev"] = self.lev
            tr["move_pct"] = round(p["side"] * (price / p["entry"] - 1) * 100, 3)
        st["trades"].append(tr)

    # ---------------------------------------------------------------- core
    def process_bar(self, sym, t, bar, entry_filter=None, regime_ok=True):
        """bar = prepared row (open/high/low/close + indicators) of a CLOSED candle."""
        st = self.st
        day = _iso(t)[:10]
        if st["day"] != day:
            st["day"], st["day_start"] = day, self.equity()
        self._fill_pending(sym, t, bar.open)
        # 3) intrabar stop
        p = st["positions"].get(sym)
        if p:
            d = p.get("side", 1)
            adverse = bar.low if d > 0 else bar.high          # worst price inside the candle
            favour = bar.high if d > 0 else bar.low
            hit = lambda lvl: lvl is not None and (adverse - lvl) * d <= 0          # noqa: E731
            worse = lambda a, b: min(a, b) if d > 0 else max(a, b)                   # noqa: E731
            liq = self._liq_price(p)
            stop = p["stop"]
            if liq is not None and hit(liq) and (stop is None or (liq - stop) * d > 0):
                self._close(sym, liq, t, "LIQUIDATED")
                p = None
            elif hit(stop):
                self._close(sym, worse(bar.open, stop) * (1 - d * self.slip), t, "stop-loss")
                p = None
            if p and p.get("tp") and (favour - p["tp"]) * d >= 0:
                better = max(bar.open, p["tp"]) if d > 0 else min(bar.open, p["tp"])
                self._close(sym, better * (1 - d * self.slip), t, "take-profit")
        st["last_close"][sym] = float(bar.close)
        st["last_ts"][sym] = _iso(t)
        # 4) decisions at this close
        p = st["positions"].get(sym)
        if p:
            p["bars"] += 1
            p["hi"] = max(p["hi"], float(bar.close))
            p["lo"] = min(p.get("lo", p["entry"]), float(bar.close))
            ns = self.s.trail_stop(p, bar)
            if ns is not None:
                p["stop"] = float(ns)
            if self.s.exit_signal(bar, p):
                st["pend_exit"].append(sym)
        elif not st["killed"] and regime_ok and (sig := self.s.entry_signal(bar)):
            st.setdefault("pend_side", {})[sym] = -1 if (self.fut and sig is not True and sig < 0) else 1
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

    def _fill_pending(self, sym, t, open_price):
        """Orders decided at the previous close fill at this candle's open."""
        st = self.st
        if sym in st["pend_exit"]:
            st["pend_exit"].remove(sym)
            if sym in st["positions"]:
                self._close(sym, open_price * (1 - self.slip), t, "signal exit")
        if sym in st["pend_entry"]:
            st["pend_entry"].pop(sym)
            side = st.get("pend_side", {}).pop(sym, 1)
            if sym not in st["positions"] and not st["killed"]:
                entry = open_price * (1 + side * self.slip)
                sig = self._sig.get(sym)
                if self.fut:
                    stop = self.s.initial_stop(entry, sig, side) if sig is not None else None
                    tp = self.s.take_profit(entry, sig, side) if sig is not None else None
                else:
                    stop = self.s.initial_stop(entry, sig) if sig is not None else None
                    tp = self.s.take_profit(entry, sig) if sig is not None else None
                qty = self._size(entry, stop)
                if self.fut:
                    margin = qty * entry / self.lev
                    cost = margin + qty * entry * self.fee
                else:
                    cost = qty * entry * (1 + self.fee)
                if qty > 0 and cost >= 5:
                    st["cash"] -= cost
                    pos = {"qty": qty, "entry": entry, "stop": stop, "cost": cost,
                           "hi": entry, "time": _iso(t), "bars": 0, "tp": tp}
                    if self.fut:
                        pos.update(margin=margin, side=side, lo=entry, lev=self.lev,
                                   r=abs(entry - stop) if stop else None)
                    st["positions"][sym] = pos

    def fill_at_open(self, sym, t, open_price):
        """Live only: as soon as the next candle has OPENED, fill pending orders at its open price
        (the exact price a backtest would use), instead of waiting for that candle to close."""
        if sym in self.st["pend_entry"] or sym in self.st["pend_exit"]:
            self._fill_pending(sym, t, open_price)
            self.st["last_close"].setdefault(sym, open_price)

    def mark(self, t, record=True):
        st = self.st
        eq = self.equity()
        st["peak"] = max(st["peak"], eq)
        if not st["killed"] and st["peak"] > 0 and eq / st["peak"] - 1 <= -self.kill_dd:
            st["killed"] = _iso(t)
            for sym in list(st["positions"]):
                self._close(sym, st["last_close"][sym] * (1 - st["positions"][sym].get("side", 1) * self.slip), t, "KILL SWITCH")
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


def run_series(portfolio, prepared: dict, start=None, entry_filter_last=None, regime=None, eligible=None):
    """Feed candles (dict sym->prepared df) in time order. Only candles after each symbol's
    last processed timestamp are used, so this works for backtests AND incremental cloud runs."""
    st = portfolio.st
    events = []
    # A coin that joins a running bot later (e.g. it entered the top-100 list today) starts from the bot's
    # current time, never from its own old history -- otherwise old candles would be traded as if live.
    bot_now = max(st["last_ts"].values()) if st["last_ts"] else None
    for sym, df in prepared.items():
        last = st["last_ts"].get(sym) or bot_now
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
        j = i
        while j < len(events) and events[j][0] == t:  # all fills at the open happen before any close
            portfolio._fill_pending(events[j][1], t, events[j][2]["open"])
            j += 1
        while i < len(events) and events[i][0] == t:
            _, sym, row = events[i]
            flt = entry_filter_last if (entry_filter_last and t == latest[sym]) else None
            ok = (regime(t) if regime else True) and (eligible(t, sym) if eligible else True)
            portfolio.process_bar(sym, t, row, flt, ok)
            i += 1
        if portfolio._cands:
            portfolio.pick_candidates(t)
        portfolio.mark(t)
    return portfolio


def _iso(t):
    return t.isoformat() if hasattr(t, "isoformat") else str(t)


def _ts(x):
    import pandas as pd
    return pd.Timestamp(x)


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


def volume_universe(daily: dict, top=30, window=30, min_days=120):
    """Point-in-time universe (no look-ahead): on each day a coin may be BOUGHT only if it was in the
    top-N by 30-day dollar volume using data known that day, and has >= min_days of history."""
    import pandas as pd
    dv = pd.DataFrame({s: (df["close"] * df["volume"]).rolling(window).mean() for s, df in daily.items()})
    age = pd.DataFrame({s: pd.Series(range(1, len(df) + 1), index=df.index) for s, df in daily.items()}).reindex(dv.index)
    dv = dv.where(age >= min_days)
    rank = dv.rank(axis=1, ascending=False)
    ok = rank <= top
    ok.index = ok.index + pd.Timedelta(days=1)  # known after the daily candle closes
    idx = ok.index
    cols = {c: ok[c].values for c in ok.columns}

    def f(t, sym):
        i = idx.searchsorted(t, side="right") - 1
        return bool(cols[sym][i]) if (i >= 0 and sym in cols) else False
    return f
