"""
Exit Evaluator
--------------
Checks all OPEN positions daily against exit conditions:
1. Stop-loss hit
2. Target hit (2x risk, i.e. 2x the stop-loss % as profit target)
3. Volume back to normal (original signal has "expired")
4. Max hold time exceeded

Sends a Telegram alert when any condition triggers. Does NOT auto-sell -
this is a nudge to review the position yourself, not an instruction.
"""

import csv
import json
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSITIONS_PATH = PROJECT_ROOT / "data" / "positions.csv"
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

VOLUME_NORMAL_THRESHOLD = 1.5  # if volume ratio drops below this, signal has "expired"


def send_telegram_message(text):
    params = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    res = urllib.request.urlopen(url)
    return json.loads(res.read())


def get_current_data(ticker):
    """Get current price and volume ratio vs recent average."""
    try:
        data = yf.download(f"{ticker}.NS", period="10d", interval="1d", progress=False)
        if data.empty or len(data) < 2:
            return None, None

        close_col = data["Close"]
        volume_col = data["Volume"]
        if close_col.ndim > 1:
            close_col = close_col.iloc[:, 0]
            volume_col = volume_col.iloc[:, 0]

        current_price = float(close_col.iloc[-1])
        today_volume = float(volume_col.iloc[-1])
        avg_volume = float(volume_col.iloc[:-1].mean())
        volume_ratio = today_volume / avg_volume if avg_volume > 0 else 0

        return current_price, volume_ratio
    except Exception as e:
        print(f"  Failed to fetch data for {ticker}: {e}")
        return None, None


def days_held(entry_date_str):
    entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d")
    return (datetime.now() - entry_date).days


def evaluate_positions():
    if not POSITIONS_PATH.exists():
        print("No positions.csv found. Add a position first with positions_manager.py")
        return

    df = pd.read_csv(POSITIONS_PATH)
    open_positions = df[df["status"] == "OPEN"]

    if open_positions.empty:
        print("No open positions to evaluate.")
        return

    alerts = []

    for idx, row in open_positions.iterrows():
        ticker = row["ticker"]
        print(f"Checking {ticker}...")

        current_price, volume_ratio = get_current_data(ticker)
        if current_price is None:
            continue

        # Track highest price since entry (for context, not auto trailing-stop yet)
        highest = max(row["highest_price_since_entry"], current_price)
        df.at[idx, "highest_price_since_entry"] = highest

        triggers = []

        if current_price <= row["stop_loss_price"]:
            triggers.append(f" STOP-LOSS HIT: price {current_price} <= stop {row['stop_loss_price']}")

        if current_price >= row["target_price"]:
            triggers.append(f" TARGET HIT: price {current_price} >= target {row['target_price']}")

        if volume_ratio is not None and volume_ratio < VOLUME_NORMAL_THRESHOLD:
            triggers.append(f" SIGNAL EXPIRED: volume back to normal ({volume_ratio:.2f}x)")

        held = days_held(str(row["entry_date"]))
        if held >= row["max_hold_days"]:
            triggers.append(f" TIME LIMIT: held {held} days (max {row['max_hold_days']})")

        if triggers:
            alerts.append({
                "ticker": ticker,
                "entry_price": row["entry_price"],
                "current_price": current_price,
                "triggers": triggers
            })

    df.to_csv(POSITIONS_PATH, index=False)

    if alerts:
        message = format_exit_alert(alerts)
        send_telegram_message(message)
        print(f"\nSent exit alert for {len(alerts)} position(s).")
    else:
        print("\nNo exit conditions triggered today.")


def format_exit_alert(alerts):
    lines = [" Position Review Needed\n"]
    for a in alerts:
        change_pct = round(((a["current_price"] - a["entry_price"]) / a["entry_price"]) * 100, 2)
        lines.append(f"{a['ticker']}: entry {a['entry_price']} -> now {a['current_price']} ({change_pct:+.2f}%)")
        for t in a["triggers"]:
            lines.append(f"  {t}")
        lines.append("")
    lines.append(" This is a review nudge, not an instruction. Decide based on your own judgment.")
    return "\n".join(lines)


if __name__ == "__main__":
    evaluate_positions()
