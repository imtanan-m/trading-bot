# Trading Bots (paper trading, cloud)

5 trading bots GitHub ke cloud pe chalte hain, har ~15 minute mein. Aap ka PC band ho tab bhi chalte rehte hain.
Sab **paper mode** mein hain: nakli paisa ($1000 per bot), lekin Binance ki asli live prices.
Progress dashboard app mein dikhti hai (GitHub Pages). Phone pe "Add to Home Screen" karke ise app ki tarah use karein.

| Bot | Kya karta hai | Backtest (Jun 2022 se ab tak) |
|---|---|---|
| Momentum Daily | Daily trend upar ho to coin rakhta hai (BTC/ETH/SOL) | +42%/saal, giraav -36% |
| Trend Rider 4h | Breakout + stop-loss, kam risk | +8.6%/saal, giraav -15% |
| Trend Rider 1h | Wahi, ghante ki candles pe | +3.8%/saal, giraav -20% |
| Fast Scalper | Day trading (dikhane ke liye ke fees kaise kha jaati hain) | -7.7%/saal, kill switch |
| Buy & Hold | Muqable ke liye: khareed ke rakhna | +16.8%/saal, giraav -65% |

**Backtest future ki guarantee nahi hai.** Ye financial advice nahi hai.

## Kaise kaam karta hai
- `.github/workflows/bot.yml`: GitHub Actions har 15 min `run_cycle.py` chalata hai.
- `run_cycle.py`: Binance se nayi candles lata hai (data-api.binance.vision, koi API key nahi), har bot ko uski saved
  state (`state/*.json`) se aage chalata hai, aur `docs/data/summary.json` likhta hai.
- `docs/index.html`: dashboard app.
- `bots.yaml`: saare bots ki settings. Naya bot ya tabdeeli: Claude ko bolein.
- `make_backtests.py`: saare bots ka backtest, `docs/data/backtest.json` mein.
- `tests/test_offline.py`: offline test (60 din ka simulated cloud run).

## Manual run
GitHub pe repo → **Actions** → "Trading bots (paper)" → **Run workflow**.

## Real paisa?
Abhi nahi. Pehle 2-3 mahine paper results dekhein. Note: GitHub ke servers US mein hain, jahan Binance ke trading
servers block hain. Demo/real trading ke liye alag (non-US) server chahiye hoga.
