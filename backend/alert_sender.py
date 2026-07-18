"""
Alert Sender
------------
Formats watchlist signals and sends them to Telegram.
Only sends an alert if at least one stock has an actionable signal,
unless force_send is True (e.g. for a daily summary regardless of signals).
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import json
import urllib.request
import urllib.parse
from signal_generator import run_watchlist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


def send_telegram_message(text):
    params = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    res = urllib.request.urlopen(url)
    return json.loads(res.read())


def format_signal_message(signals):
    """Build a readable multi-stock summary message."""
    lines = ["📊 Daily Stock Signal Summary\n"]

    actionable_count = 0

    for signal in signals:
        is_actionable = signal["volume"]["unusual_volume"]
        if is_actionable:
            actionable_count += 1

        marker = "🔔" if is_actionable else "•"
        lines.append(
            f"{marker} {signal['company_name']} ({signal['ticker']})\n"
            f"   Volume ratio: {signal['volume']['ratio']}x | Sentiment: {signal['sentiment']['overall_label']}\n"
            f"   {signal['recommendation']}\n"
        )

    lines.append(f"\n{actionable_count} of {len(signals)} stocks flagged today.")
    lines.append("\n⚠️ This is not financial advice - always verify independently.")

    return "\n".join(lines), actionable_count


def run_and_alert(force_send=True):
    """Run the watchlist scan and send a Telegram summary."""
    print("Running watchlist scan...")
    signals = run_watchlist()

    message, actionable_count = format_signal_message(signals)

    if actionable_count > 0 or force_send:
        result = send_telegram_message(message)
        print("Telegram alert sent.")
        return result
    else:
        print("No actionable signals - alert skipped (force_send=False).")
        return None


if __name__ == "__main__":
    run_and_alert(force_send=True)
