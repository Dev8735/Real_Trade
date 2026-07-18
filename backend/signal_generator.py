"""
Signal Generator
----------------
Combines news sentiment + unusual volume detection into one
buy/sell/hold style recommendation for a given stock.

This is a decision-support tool, NOT financial advice.
No signal here guarantees accuracy - always verify independently.
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import requests
import pandas as pd
import yfinance as yf
from sentiment_analyzer import analyze_sentiment
from watchlist import WATCHLIST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
VOLUME_THRESHOLD_MULTIPLIER = 1.5


def fetch_news_for_stock(search_query, company_name, page_size=10):
    """Fetch news and filter out headlines that don't actually mention the company."""
    url = "https://newsapi.org/v2/everything"
    resp = requests.get(url, params={
        "q": search_query,
        "apiKey": NEWSAPI_KEY,
        "pageSize": page_size,
        "sortBy": "publishedAt"
    })
    data = resp.json()
    if data.get("status") != "ok":
        return []

    articles = data.get("articles", [])

    # Relevance filter: keep only headlines that actually mention part of the company name
    name_parts = [p.lower() for p in company_name.split() if len(p) > 2]
    relevant_titles = []
    for article in articles:
        title = article.get("title", "")
        title_lower = title.lower()
        if any(part in title_lower for part in name_parts):
            relevant_titles.append(title)

    return relevant_titles


def get_news_sentiment_summary(search_query, company_name):
    headlines = fetch_news_for_stock(search_query, company_name)

    if not headlines:
        return {"overall_label": "neutral", "overall_score": 0, "headlines_analyzed": []}

    total_score = 0
    analyzed = []

    for headline in headlines:
        result = analyze_sentiment(headline)
        total_score += result["score"]
        analyzed.append({"headline": headline, "label": result["label"], "score": result["score"]})

    if total_score > 0:
        overall_label = "positive"
    elif total_score < 0:
        overall_label = "negative"
    else:
        overall_label = "neutral"

    return {
        "overall_label": overall_label,
        "overall_score": total_score,
        "headlines_analyzed": analyzed
    }


def check_volume_pattern(ticker, period="10d", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty or len(data) < 2:
        return {"unusual_volume": False, "today_volume": None, "avg_volume": None, "ratio": None}

    if isinstance(data.columns, pd.MultiIndex):
        volume_series = data["Volume"].iloc[:, 0]
    else:
        volume_series = data["Volume"]

    today_volume = float(volume_series.iloc[-1])
    prior_days = volume_series.iloc[:-1]
    avg_volume = float(prior_days.mean())

    ratio = today_volume / avg_volume if avg_volume > 0 else 0
    unusual = ratio >= VOLUME_THRESHOLD_MULTIPLIER

    return {
        "unusual_volume": unusual,
        "today_volume": today_volume,
        "avg_volume": avg_volume,
        "ratio": round(ratio, 2)
    }


def generate_signal(ticker, company_name, search_query):
    sentiment = get_news_sentiment_summary(search_query, company_name)
    volume = check_volume_pattern(ticker)

    if volume["unusual_volume"] and sentiment["overall_label"] == "positive":
        recommendation = "WATCH - possible BUY signal (unusual volume + positive news)"
    elif volume["unusual_volume"] and sentiment["overall_label"] == "negative":
        recommendation = "WATCH - possible SELL signal (unusual volume + negative news)"
    elif volume["unusual_volume"]:
        recommendation = "WATCH - unusual volume, but news sentiment is neutral/mixed"
    else:
        recommendation = "NO SIGNAL - volume normal today"

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sentiment": sentiment,
        "volume": volume,
        "recommendation": recommendation
    }


def run_watchlist():
    """Scan every stock in the watchlist and return all signals."""
    results = []
    for stock in WATCHLIST:
        print(f"Scanning {stock['company_name']} ({stock['ticker']})...")
        signal = generate_signal(stock["ticker"], stock["company_name"], stock["search_query"])
        results.append(signal)
    return results


if __name__ == "__main__":
    signals = run_watchlist()

    for signal in signals:
        print(f"\n=== {signal['company_name']} ({signal['ticker']}) ===")
        print(f"Volume ratio: {signal['volume']['ratio']} | Unusual: {signal['volume']['unusual_volume']}")
        print(f"Sentiment: {signal['sentiment']['overall_label']} (score: {signal['sentiment']['overall_score']})")
        print(f">>> {signal['recommendation']}")
