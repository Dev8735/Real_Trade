"""
Insider Trading Pattern Detector
---------------------------------
Fetches news related to stocks and sends alerts via Telegram,
only alerting when a genuinely new headline appears.
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import json
import urllib.request
import urllib.parse
import requests

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
last_headline_path = PROJECT_ROOT / "data" / "last_headline.txt"

load_dotenv(dotenv_path=env_path)

# --- Load credentials ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Missing BOT_TOKEN or CHAT_ID - check your .env file")
if not NEWSAPI_KEY:
    raise ValueError("Missing NEWSAPI_KEY - check your .env file")


def send_telegram_message(text):
    """Send a message to your Telegram chat via bot."""
    params = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    res = urllib.request.urlopen(url)
    return json.loads(res.read())


def fetch_news(query="NSE India stock", page_size=3):
    """Fetch recent news articles matching the query."""
    url = "https://newsapi.org/v2/everything"
    resp = requests.get(url, params={
        "q": query,
        "apiKey": NEWSAPI_KEY,
        "pageSize": page_size
    })
    return resp.json()


def get_last_headline():
    """Read the last alerted headline from disk, or None if never saved."""
    if last_headline_path.exists():
        return last_headline_path.read_text(encoding="utf-8").strip()
    return None


def save_last_headline(headline):
    """Save the latest alerted headline to disk (creates data/ if missing)."""
    last_headline_path.parent.mkdir(parents=True, exist_ok=True)
    last_headline_path.write_text(headline, encoding="utf-8")


def main():
    data = fetch_news()
    print("Status:", data.get("status"))
    print("Total results:", data.get("totalResults"))

    articles = data.get("articles")
    if not articles:
        print("No articles found.")
        return

    latest_headline = articles[0]["title"]
    print("Latest fetched headline:", latest_headline)

    last_headline = get_last_headline()
    print("Last alerted headline:", last_headline)

    if latest_headline == last_headline:
        print("No new headline — skipping alert.")
        return

    # Genuinely new headline — alert and remember it
    send_telegram_message(f"New headline: {latest_headline}")
    save_last_headline(latest_headline)
    print("New headline alerted and saved.")


if __name__ == "__main__":
    main()