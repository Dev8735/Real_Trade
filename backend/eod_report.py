"""
End-of-Day Report
-------------------
Runs ~3:45 PM daily. Covers three things:
1. Performance check: how did today's morning signals actually do?
2. Missed movers: other stocks with big volume/price action today that
   weren't in the morning signal list (both up and down moves).
3. Status read (not a prediction): is each notable stock's momentum
   still building or fading as of today's close?
"""

from volume_screener import scan_all_stocks
from market_context import get_nifty_change, get_daily_change_pct, classify_relative_strength
from dotenv import load_dotenv
from pathlib import Path
import os, json, urllib.request, urllib.parse
import pandas as pd
import yfinance as yf
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SIGNALS_LOG_PATH = PROJECT_ROOT / "data" / "signals_log.csv"

MISSED_MOVER_TOP_N = 10
MOMENTUM_STILL_BUILDING_RATIO = 5.0  # if today's volume ratio is still this high, momentum may be continuing


def send_telegram_message(text):
    params = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    try:
        res = urllib.request.urlopen(url)
        return json.loads(res.read())
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return None


def get_current_price(ticker):
    try:
        data = yf.download(f"{ticker}.NS", period="1d", interval="1d", progress=False)
        if data.empty:
            return None
        close_col = data["Close"]
        return float(close_col.iloc[-1]) if close_col.ndim == 1 else float(close_col.iloc[-1, 0])
    except Exception:
        return None


def check_todays_signal_performance():
    """Look up today's logged signals and check how they've performed intraday."""
    if not SIGNALS_LOG_PATH.exists():
        return []

    df = pd.read_csv(SIGNALS_LOG_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    todays_signals = df[df["signal_date"] == today]

    results = []
    for _, row in todays_signals.iterrows():
        current_price = get_current_price(row["ticker"])
        if current_price is None or pd.isna(row["price_at_signal"]) or row["price_at_signal"] == 0:
            continue

        change_pct = round(((current_price - row["price_at_signal"]) / row["price_at_signal"]) * 100, 2)

        results.append({
            "ticker": row["ticker"],
            "direction": row.get("direction", "N/A"),
            "entry_price": row["price_at_signal"],
            "current_price": current_price,
            "change_pct": change_pct,
            "composite_score": row.get("composite_score", "N/A")
        })

    return results


def find_missed_movers(already_signaled_tickers, nifty_change):
    """Find today's other notable movers that weren't in the morning signal list."""
    top_movers = scan_all_stocks(limit=None, top_n=50, debug=False)

    missed = []
    for stock in top_movers:
        ticker_clean = stock["ticker"].replace(".NS", "")
        if ticker_clean in already_signaled_tickers:
            continue

        stock_change = get_daily_change_pct(f"{ticker_clean}.NS")
        relative_strength = classify_relative_strength(stock_change, nifty_change)

        if relative_strength not in ("outperforming market", "underperforming market"):
            continue

        direction = "Notable UP move" if (stock_change or 0) > 0 else "Notable DOWN move"

        momentum_status = (
            "volume still elevated - momentum may be continuing"
            if stock["ratio"] >= MOMENTUM_STILL_BUILDING_RATIO
            else "volume moderating - move may be settling"
        )

        missed.append({
            "ticker": ticker_clean,
            "direction": direction,
            "volume_ratio": stock["ratio"],
            "change_pct": stock_change,
            "relative_strength": relative_strength,
            "momentum_status": momentum_status
        })

        if len(missed) >= MISSED_MOVER_TOP_N:
            break

    return missed


def format_report(performance_results, missed_movers, nifty_change):
    lines = [f" End-of-Day Report (Nifty: {nifty_change}%)\n"]

    if performance_results:
        lines.append(" Today's Signal Performance:")
        for r in performance_results:
            emoji = "" if r["change_pct"] > 0 else "" if r["change_pct"] < 0 else ""
            lines.append(
                f"{emoji} {r['ticker']} ({r['direction']}): entry ₹{r['entry_price']}  "
                f"now ₹{r['current_price']} ({r['change_pct']:+.2f}%) | score {r['composite_score']}"
            )
        lines.append("")
    else:
        lines.append(" No signals were logged this morning to check.\n")

    if missed_movers:
        lines.append(" Other Notable Movers Today (not in morning signal):")
        for m in missed_movers:
            lines.append(
                f"   {m['ticker']}: {m['direction']}, {m['change_pct']}% "
                f"({m['relative_strength']}), {m['volume_ratio']}x volume"
            )
            lines.append(f"    Status as of today's close: {m['momentum_status']}")
        lines.append("")

    lines.append(
        "ℹ \"Momentum status\" describes today's close only - it is NOT a prediction "
        "of tomorrow's move. No one can reliably predict that. Verify independently before acting."
    )
    return "\n".join(lines)


def run_eod_report():
    print("=== Checking today's signal performance ===")
    performance_results = check_todays_signal_performance()
    print(f"Found {len(performance_results)} signal(s) from today to check.")

    print("\n=== Fetching Nifty for context ===")
    nifty_change = get_nifty_change()
    print(f"Nifty 50 today: {nifty_change}%")

    already_signaled = [r["ticker"] for r in performance_results]

    print("\n=== Scanning for missed movers ===")
    missed_movers = find_missed_movers(already_signaled, nifty_change)
    print(f"Found {len(missed_movers)} notable unsignaled mover(s).")

    print("\n=== Sending report ===")
    report = format_report(performance_results, missed_movers, nifty_change)
    send_telegram_message(report)
    print("Report sent.")


if __name__ == "__main__":
    run_eod_report()
