"""
Full Pipeline
-------------
Stage 1: Scan all NSE stocks for unusual volume (cheap, no API cost)
Stage 2: Run news + sentiment only on the top volume movers (uses NewsAPI quota)
Stage 3: Send combined signals to Telegram

This is a decision-support tool, NOT financial advice.
"""

from volume_screener import scan_all_stocks
from sentiment_analyzer import analyze_sentiment
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
import json
import urllib.request
import urllib.parse
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TOP_N = 30
NEWSAPI_DAILY_LIMIT_BUFFER = 30  # stay well under 100/day


def send_telegram_message(text):
    params = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    res = urllib.request.urlopen(url)
    return json.loads(res.read())


def fetch_news_for_ticker(ticker_symbol, page_size=5):
    """Fetch news using just the raw ticker symbol as the search term."""
    url = "https://newsapi.org/v2/everything"
    try:
        resp = requests.get(url, params={
            "q": ticker_symbol,
            "apiKey": NEWSAPI_KEY,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "language": "en"
        }, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return []
        return [a["title"] for a in data.get("articles", []) if a.get("title")]
    except Exception as e:
        print(f"  News fetch failed for {ticker_symbol}: {e}")
        return []


def get_sentiment_for_headlines(headlines):
    if not headlines:
        return {"label": "neutral", "score": 0}

    total_score = 0
    for h in headlines:
        result = analyze_sentiment(h)
        total_score += result["score"]

    label = "positive" if total_score > 0 else "negative" if total_score < 0 else "neutral"
    return {"label": label, "score": total_score}


def run_full_pipeline():
    print("=== STAGE 1: Scanning all NSE stocks for unusual volume ===")
    top_movers = scan_all_stocks(limit=None, top_n=TOP_N, debug=False)
    print(f"Found top {len(top_movers)} volume movers.\n")

    print("=== STAGE 2: Running news + sentiment on top movers ===")
    final_signals = []

    for i, stock in enumerate(top_movers[:NEWSAPI_DAILY_LIMIT_BUFFER]):
        ticker_clean = stock["ticker"].replace(".NS", "")
        print(f"[{i+1}/{len(top_movers)}] {ticker_clean} (volume ratio {stock['ratio']}x)...")

        headlines = fetch_news_for_ticker(ticker_clean)
        sentiment = get_sentiment_for_headlines(headlines)

        final_signals.append({
            "ticker": ticker_clean,
            "volume_ratio": stock["ratio"],
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "headline_count": len(headlines),
            "sample_headline": headlines[0] if headlines else None
        })

        time.sleep(0.3)  # be gentle on NewsAPI

    print("\n=== STAGE 3: Sending summary to Telegram ===")
    message = format_message(final_signals)
    send_telegram_message(message)
    print("Alert sent.")

    return final_signals


def format_message(signals):
    lines = ["📊 Top NSE Volume Movers + News Sentiment\n"]

    for s in signals:
        emoji = "🟢" if s["sentiment_label"] == "positive" else "🔴" if s["sentiment_label"] == "negative" else "⚪"
        lines.append(
            f"{emoji} {s['ticker']}: {s['volume_ratio']}x volume | "
            f"sentiment: {s['sentiment_label']} ({s['headline_count']} articles)"
        )
        if s["sample_headline"]:
            lines.append(f"    → {s['sample_headline']}")

    lines.append("\n⚠️ Decision-support only, not financial advice. Verify independently.")
    return "\n".join(lines)


if __name__ == "__main__":
    signals = run_full_pipeline()

    print("\n=== FINAL SUMMARY ===")
    for s in signals:
        print(f"{s['ticker']}: vol={s['volume_ratio']}x sentiment={s['sentiment_label']} ({s['headline_count']} articles)")
