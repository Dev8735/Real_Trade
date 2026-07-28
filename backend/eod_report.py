"""
End-of-Day Report
-------------------
Runs ~3:45 PM daily. Covers three clearly separated sections:
1. Morning Signal Performance: how did today's morning signals actually do?
2. Top Gainers Today: biggest up-moves today, with evidence, whether or not
   they were in the morning signal.
3. Top Losers Today: biggest down-moves today, with evidence, whether or not
   they were in the morning signal.
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

TOP_GAINERS_LOSERS_N = 8
MOMENTUM_STILL_BUILDING_RATIO = 5.0


def _send_single_message(text):
    params = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    try:
        res = urllib.request.urlopen(url)
        return json.loads(res.read())
    except Exception as e:
        try:
            error_body = e.read().decode() if hasattr(e, "read") else str(e)
        except Exception:
            error_body = str(e)
        print(f"[ERROR] Telegram send failed: {e}")
        print(f"[ERROR] Telegram response body: {error_body}")
        return None


def send_telegram_message(text):
    if len(text) > 3800:
        chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)]
        results = []
        for chunk in chunks:
            results.append(_send_single_message(chunk))
        return results
    return _send_single_message(text)


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
    if not SIGNALS_LOG_PATH.exists():
        return [], set()

    df = pd.read_csv(SIGNALS_LOG_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    todays_signals = df[df["signal_date"] == today]

    results = []
    seen_tickers = set()
    for _, row in todays_signals.iterrows():
        if row["ticker"] in seen_tickers:
            continue
        seen_tickers.add(row["ticker"])

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

    return results, seen_tickers


def get_todays_gainers_and_losers(nifty_change, already_signaled_tickers):
    top_movers = scan_all_stocks(limit=None, top_n=60, debug=False)

    enriched = []
    for stock in top_movers:
        ticker_clean = stock["ticker"].replace(".NS", "")
        stock_change = get_daily_change_pct(f"{ticker_clean}.NS")
        if stock_change is None:
            continue

        relative_strength = classify_relative_strength(stock_change, nifty_change)
        momentum_status = (
            "volume still elevated - move may be continuing"
            if stock["ratio"] >= MOMENTUM_STILL_BUILDING_RATIO
            else "volume moderating - move may be settling"
        )

        enriched.append({
            "ticker": ticker_clean,
            "volume_ratio": stock["ratio"],
            "change_pct": stock_change,
            "relative_strength": relative_strength,
            "momentum_status": momentum_status,
            "already_signaled": ticker_clean in already_signaled_tickers
        })

    gainers = sorted([s for s in enriched if s["change_pct"] > 0], key=lambda x: x["change_pct"], reverse=True)
    losers = sorted([s for s in enriched if s["change_pct"] < 0], key=lambda x: x["change_pct"])

    return gainers[:TOP_GAINERS_LOSERS_N], losers[:TOP_GAINERS_LOSERS_N]


def format_report(performance_results, gainers, losers, nifty_change):
    lines = [f"EOD Report (Nifty: {nifty_change}%)\n"]

    lines.append("=== 1. Morning Signal Performance ===")
    if performance_results:
        for r in performance_results:
            tag = "UP" if r["change_pct"] > 0 else "DOWN" if r["change_pct"] < 0 else "FLAT"
            lines.append(
                f"[{tag}] {r['ticker']} ({r['direction']}): entry {r['entry_price']} -> "
                f"now {r['current_price']} ({r['change_pct']:+.2f}%) | score {r['composite_score']}"
            )
    else:
        lines.append("No signals were logged this morning.")
    lines.append("")

    lines.append("=== 2. Top Gainers Today ===")
    if gainers:
        for g in gainers:
            flag = " [already in morning signal]" if g["already_signaled"] else ""
            lines.append(
                f"  {g['ticker']}: +{g['change_pct']}% | {g['volume_ratio']}x volume | "
                f"{g['relative_strength']}{flag}"
            )
            lines.append(f"    Status: {g['momentum_status']}")
    else:
        lines.append("No notable gainers found.")
    lines.append("")

    lines.append("=== 3. Top Losers Today ===")
    if losers:
        for l in losers:
            flag = " [already in morning signal]" if l["already_signaled"] else ""
            lines.append(
                f"  {l['ticker']}: {l['change_pct']}% | {l['volume_ratio']}x volume | "
                f"{l['relative_strength']}{flag}"
            )
            lines.append(f"    Status: {l['momentum_status']}")
    else:
        lines.append("No notable losers found.")
    lines.append("")

    lines.append(
        "Note: 'Status' describes today's close only - NOT a prediction of tomorrow. "
        "Decision-support only, not financial advice. Verify independently before acting."
    )
    return "\n".join(lines)


def run_eod_report():
    print("=== Checking today's signal performance ===")
    performance_results, already_signaled = check_todays_signal_performance()
    print(f"Found {len(performance_results)} signal(s) from today to check.")

    print("\n=== Fetching Nifty for context ===")
    nifty_change = get_nifty_change()
    print(f"Nifty 50 today: {nifty_change}%")

    print("\n=== Scanning for today's gainers and losers ===")
    gainers, losers = get_todays_gainers_and_losers(nifty_change, already_signaled)
    print(f"Found {len(gainers)} gainer(s), {len(losers)} loser(s).")

    print("\n=== Sending report ===")
    report = format_report(performance_results, gainers, losers, nifty_change)
    print(f"[DEBUG] Report length: {len(report)} characters")
    result = send_telegram_message(report)
    if result:
        print("Report sent successfully.")
    else:
        print("Report send FAILED - see error above.")


if __name__ == "__main__":
    run_eod_report()
