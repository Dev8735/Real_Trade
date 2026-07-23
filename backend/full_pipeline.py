"""
Full Pipeline
-------------
Stage 1: Scan all NSE stocks for unusual volume
Stage 2: Cross-check against block/bulk deals
Stage 3: Compare against Nifty (filter out market-wide noise)
Stage 4: Run news + sentiment on top movers
Stage 5: Score confirmed/unconfirmed signal + confidence
Stage 6: Add risk context (stop-loss, liquidity) to HIGH signals
Stage 7: Send HIGH signals to Telegram
Stage 8: Log everything for accuracy tracking
"""

from volume_screener import scan_all_stocks
from sentiment_analyzer import analyze_sentiment
from block_deal_checker import get_todays_deal_symbols, check_stock_for_deals
from market_context import get_nifty_change, get_daily_change_pct, classify_relative_strength
from risk_manager import get_volatility_stop_loss, check_liquidity_risk
from dotenv import load_dotenv
from pathlib import Path
import os, requests, json, urllib.request, urllib.parse, time, csv, sys
import pandas as pd
import yfinance as yf
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

print(f"[DEBUG] BOT_TOKEN is set: {bool(BOT_TOKEN)} (length: {len(BOT_TOKEN) if BOT_TOKEN else 0})")
print(f"[DEBUG] CHAT_ID is set: {bool(CHAT_ID)} value: {CHAT_ID}")
print(f"[DEBUG] NEWSAPI_KEY is set: {bool(NEWSAPI_KEY)}")

required = ["BOT_TOKEN", "CHAT_ID", "NEWSAPI_KEY"]
missing = [k for k in required if not os.getenv(k)]
if missing:
    print(f"[ERROR] Missing env vars: {', '.join(missing)}")
    sys.exit(1)

TOP_N = 30
NEWSAPI_DAILY_LIMIT_BUFFER = 30
SIGNALS_LOG_PATH = PROJECT_ROOT / "data" / "signals_log.csv"
TICKER_CACHE = PROJECT_ROOT / "data" / "nse_tickers.csv"

CONFIRMED_HIGH_MIN_VOLUME = 5.0
CONFIRMED_HIGH_MIN_ARTICLES = 2
CONFIRMED_HIGH_MIN_SENTIMENT = 2

UNCONFIRMED_HIGH_MIN_VOLUME = 8.0
UNCONFIRMED_HIGH_MIN_MOVE_PCT = 5.0
UNCONFIRMED_MEDIUM_MIN_VOLUME = 3.0


def load_company_names():
    df = pd.read_csv(TICKER_CACHE)
    return dict(zip(df["SYMBOL"], df["NAME OF COMPANY"]))


COMPANY_NAMES = load_company_names()


def send_telegram_message(text):
    params = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    try:
        res = urllib.request.urlopen(url)
        result = json.loads(res.read())
        print(f"[DEBUG] Telegram response: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return None


def fetch_relevant_news(ticker_symbol, page_size=10):
    company_name = COMPANY_NAMES.get(ticker_symbol, ticker_symbol)
    clean_name = company_name.replace("Limited", "").replace("Ltd", "").strip()
    search_query = f'"{clean_name}"'
    url = "https://newsapi.org/v2/everything"
    try:
        resp = requests.get(url, params={
            "q": search_query, "apiKey": NEWSAPI_KEY, "pageSize": page_size,
            "sortBy": "publishedAt", "language": "en"
        }, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return []
    except Exception as e:
        print(f"  News fetch failed for {ticker_symbol}: {e}")
        return []

    name_parts = [p.lower() for p in clean_name.split() if len(p) > 3]
    relevant = []
    for a in data.get("articles", []):
        title = a.get("title", "")
        if title and any(part in title.lower() for part in name_parts):
            relevant.append(title)
    return relevant


def get_sentiment_for_headlines(headlines):
    if not headlines:
        return {"label": "neutral", "score": 0}
    total_score = sum(analyze_sentiment(h)["score"] for h in headlines)
    label = "positive" if total_score > 0 else "negative" if total_score < 0 else "neutral"
    return {"label": label, "score": total_score}


def get_current_price(ticker_symbol):
    try:
        data = yf.download(f"{ticker_symbol}.NS", period="1d", interval="1d", progress=False)
        if data.empty:
            return None
        close_col = data["Close"]
        return float(close_col.iloc[-1]) if close_col.ndim == 1 else float(close_col.iloc[-1, 0])
    except Exception:
        return None


def score_signal(volume_ratio, relevant_article_count, sentiment_score, has_deal_explanation,
                  relative_strength, stock_change_pct):
    if has_deal_explanation:
        return "EXPLAINED", "LOW"

    has_news = relevant_article_count >= 1
    diverging = relative_strength in ("outperforming market", "underperforming market")
    move_size = abs(stock_change_pct) if stock_change_pct is not None else 0

    if has_news:
        if (volume_ratio >= CONFIRMED_HIGH_MIN_VOLUME
                and relevant_article_count >= CONFIRMED_HIGH_MIN_ARTICLES
                and abs(sentiment_score) >= CONFIRMED_HIGH_MIN_SENTIMENT
                and diverging):
            return "CONFIRMED", "HIGH"
        elif volume_ratio >= 3.0:
            return "CONFIRMED", "MEDIUM"
        else:
            return "CONFIRMED", "LOW"
    else:
        if (volume_ratio >= UNCONFIRMED_HIGH_MIN_VOLUME
                and move_size >= UNCONFIRMED_HIGH_MIN_MOVE_PCT
                and diverging):
            return "UNCONFIRMED", "HIGH"
        elif volume_ratio >= UNCONFIRMED_MEDIUM_MIN_VOLUME and diverging:
            return "UNCONFIRMED", "MEDIUM"
        else:
            return "UNCONFIRMED", "LOW"


def run_full_pipeline():
    print("=== STAGE 1: Scanning all NSE stocks for unusual volume ===")
    top_movers = scan_all_stocks(limit=None, top_n=TOP_N, debug=False)
    print(f"Found top {len(top_movers)} volume movers.\n")

    print("=== STAGE 2: Fetching block/bulk deals ===")
    deals_dict = get_todays_deal_symbols()
    print(f"Found deal data for {len(deals_dict)} stocks today.\n")

    print("=== STAGE 3: Fetching Nifty index ===")
    nifty_change = get_nifty_change()
    print(f"Nifty 50 today: {nifty_change}%\n")

    print("=== STAGE 4: Running relevance-filtered news + sentiment ===")
    final_signals = []

    for i, stock in enumerate(top_movers[:NEWSAPI_DAILY_LIMIT_BUFFER]):
        ticker_clean = stock["ticker"].replace(".NS", "")
        print(f"[{i+1}/{len(top_movers)}] {ticker_clean} (volume ratio {stock['ratio']}x)...")

        deal_explanation = check_stock_for_deals(ticker_clean, deals_dict)
        headlines = fetch_relevant_news(ticker_clean)
        sentiment = get_sentiment_for_headlines(headlines)
        price = get_current_price(ticker_clean)
        stock_change = get_daily_change_pct(f"{ticker_clean}.NS")
        relative_strength = classify_relative_strength(stock_change, nifty_change)

        signal_type, confidence = score_signal(
            stock["ratio"], len(headlines), sentiment["score"],
            bool(deal_explanation), relative_strength, stock_change
        )

        risk_info = {"stop_loss_pct": None, "liquidity_risk": None}
        if confidence == "HIGH":
            vol_data = get_volatility_stop_loss(ticker_clean)
            liq_data = check_liquidity_risk(ticker_clean)
            if vol_data:
                risk_info["stop_loss_pct"] = vol_data["suggested_stop_loss_pct"]
            if liq_data:
                risk_info["liquidity_risk"] = liq_data["liquidity_risk"]

        final_signals.append({
            "ticker": ticker_clean,
            "volume_ratio": stock["ratio"],
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "headline_count": len(headlines),
            "sample_headline": headlines[0] if headlines else None,
            "price_at_signal": price,
            "stock_change_pct": stock_change,
            "relative_strength": relative_strength,
            "signal_type": signal_type,
            "confidence": confidence,
            "deal_explanation": deal_explanation,
            "stop_loss_pct": risk_info["stop_loss_pct"],
            "liquidity_risk": risk_info["liquidity_risk"]
        })

        time.sleep(0.3)

    confirmed_high = [s for s in final_signals if s["signal_type"] == "CONFIRMED" and s["confidence"] == "HIGH"]
    unconfirmed_high = [s for s in final_signals if s["signal_type"] == "UNCONFIRMED" and s["confidence"] == "HIGH"]

    print(f"\n=== STAGE 5: {len(confirmed_high)} CONFIRMED-HIGH, {len(unconfirmed_high)} UNCONFIRMED-HIGH today ===")

    if confirmed_high or unconfirmed_high:
        message = format_message(confirmed_high, unconfirmed_high, nifty_change)
        send_telegram_message(message)
    else:
        send_telegram_message(f"No high-confidence signals today (Nifty: {nifty_change}%). Quiet day.")

    print("\n=== STAGE 6: Logging ALL signals for tracking ===")
    log_signals_to_csv(final_signals, nifty_change)

    return final_signals


def format_message(confirmed_high, unconfirmed_high, nifty_change):
    lines = [f" Signal Summary (Nifty: {nifty_change}%)\n"]

    if confirmed_high:
        lines.append(" CONFIRMED signals (news + volume + market divergence):")
        for s in confirmed_high:
            emoji = "" if s["sentiment_label"] == "positive" else ""
            lines.append(
                f"{emoji} {s['ticker']}: {s['volume_ratio']}x vol | {s['stock_change_pct']}% "
                f"({s['relative_strength']}) | {s['sentiment_label']}"
            )
            if s["sample_headline"]:
                lines.append(f"     {s['sample_headline']}")
            if s["stop_loss_pct"]:
                lines.append(f"     Suggested stop-loss: {s['stop_loss_pct']}% | Liquidity: {s['liquidity_risk']}")
        lines.append("")

    if unconfirmed_high:
        lines.append(" UNCONFIRMED signals (big move, NO news - higher risk, verify yourself):")
        for s in unconfirmed_high:
            lines.append(
                f" {s['ticker']}: {s['volume_ratio']}x vol | {s['stock_change_pct']}% "
                f"({s['relative_strength']}) | no news coverage"
            )
            if s["stop_loss_pct"]:
                lines.append(f"     Suggested stop-loss: {s['stop_loss_pct']}% | Liquidity: {s['liquidity_risk']}")
        lines.append("")

    lines.append(" Decision-support only, not financial advice. Always verify independently.")
    return "\n".join(lines)


def log_signals_to_csv(signals, nifty_change):
    SIGNALS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SIGNALS_LOG_PATH.exists()

    with open(SIGNALS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "signal_date", "ticker", "volume_ratio", "sentiment_label", "sentiment_score",
                "headline_count", "sample_headline", "price_at_signal", "stock_change_pct",
                "nifty_change_pct", "relative_strength", "signal_type", "confidence", "deal_explanation",
                "stop_loss_pct", "liquidity_risk",
                "checked_1d", "checked_3d", "checked_7d", "price_1d", "price_3d", "price_7d"
            ])
        today = datetime.now().strftime("%Y-%m-%d")
        for s in signals:
            writer.writerow([
                today, s["ticker"], s["volume_ratio"], s["sentiment_label"], s["sentiment_score"],
                s["headline_count"], s["sample_headline"] or "", s["price_at_signal"], s["stock_change_pct"],
                nifty_change, s["relative_strength"], s["signal_type"], s["confidence"], s["deal_explanation"] or "",
                s["stop_loss_pct"] or "", s["liquidity_risk"] or "",
                "", "", "", "", "", ""
            ])
    print(f"Logged {len(signals)} signals to {SIGNALS_LOG_PATH}")


if __name__ == "__main__":
    signals = run_full_pipeline()
    print("\n=== FINAL SUMMARY ===")
    for s in signals:
        print(f"{s['ticker']}: vol={s['volume_ratio']}x type={s['signal_type']} confidence={s['confidence']} "
              f"stop_loss={s['stop_loss_pct']}% liquidity={s['liquidity_risk']}")
