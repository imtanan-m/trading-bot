"""Optional Telegram alerts. Turned on by two GitHub secrets:
TELEGRAM_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID. Without them this does nothing."""
import os

import requests


def enabled():
    return bool(os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send(text, log=None):
    if not enabled():
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/sendMessage",
                          json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text,
                                "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=10)
        return r.ok
    except Exception as e:  # noqa: BLE001
        if log:
            log.warning(f"telegram failed: {e}")
        return False


def _px(v):
    return f"{v:,.2f}" if v >= 1 else f"{v:.5f}"


def diff_messages(bot_name, before, after):
    """Human messages for what changed in a bot between two states."""
    msgs = []
    for sym, p in after["positions"].items():
        if sym not in before.get("positions", {}):
            msgs.append(f"🟢 <b>{bot_name}</b> ne {sym} khareeda @ ${_px(p['entry'])}"
                        + (f" (stop ${_px(p['stop'])})" if p.get("stop") else ""))
    for t in after["trades"][len(before.get("trades", [])):]:
        icon = "✅" if t["pnl"] > 0 else "🔴"
        msgs.append(f"{icon} <b>{bot_name}</b> ne {t['symbol']} becha: {t['ret_pct']:+.2f}% "
                    f"(${t['pnl']:+.2f}, {t['reason']})")
    if after.get("killed") and not before.get("killed"):
        msgs.append(f"⛔ <b>{bot_name}</b>: kill switch, bot band ho gaya")
    return msgs
