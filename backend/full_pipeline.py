"""
Full Pipeline
-------------
Stage 1: Scan all NSE stocks for unusual volume (ratio + z-score)
Stage 2: Cross-check against block/bulk deals - confirmation source
Stage 3: Compare against Nifty (filter out market-wide noise)
Stage 4: Run news + sentiment on top movers
Stage 5: Score CONFIRMED / DEAL_CONFIRMED / UNCONFIRMED signals
Stage 6: Add risk context (stop-loss, target, RSI, day-range, candle trend)
Stage 7: Send TWO separate Telegram messages
Stage 8: Log everything for accuracy tracking
"""

from volume_screener import scan_all_stocks
from sentiment_analyzer import analyze_sentiment
from block_deal_checker import get_todays_deal_symbols, check_stock_for_deals
from market_context import get_nifty_change, get_daily_change_pct, classify_relative_strength
from risk_manager import get_volatility_stop_loss, check_liquidity_risk, calculate_rsi, get_day_range_position
from candle_analysis import analyze_candle
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
RUN_LABEL = os.getenv("RUN_LABEL", "Signal Check")

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

MIN_ZSCORE_FOR_HIGH = 3.0
EXTREME_VOLUME_RATIO = 15.0
EXTREME_ZSCORE = 15.0
TARGET_RATIO = 1.5
MIN_SCORE_THRESHOLD = 90  # raised to match CONFIRMED threshold - minimize risk
CONFIRMED_MIN_SCORE_THRESHOLD = 90


def load_company_names():
    df = pd.read_csv(TICKER_CACHE)
    return dict(zip(df["SYMBOL"], df["NAME OF COMPANY"]))


COMPANY_NAMES = load_company_names()


def _send_single_message(text, chat_id):
    params = urllib.parse.urlencode({"chat_id": chat_id, "text": text})
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?{params}"
    try:
        res = urllib.request.urlopen(url)
        result = json.loads(res.read())
        print(f"[DEBUG] Telegram response for {chat_id}: {result}")
        return result
    except Exception as e:
        try:
            error_body = e.read().decode() if hasattr(e, "read") else str(e)
        except Exception:
            error_body = str(e)
        print(f"[ERROR] Telegram send failed for {chat_id}: {e}")
        print(f"[ERROR] Telegram response body: {error_body}")
        return None


def send_telegram_message(text):
    chat_ids = [c.strip() for c in CHAT_ID.split(",") if c.strip()]
    all_results = []
    for chat_id in chat_ids:
        if len(text) > 3800:
            chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)]
            for chunk in chunks:
                all_results.append(_send_single_message(chunk, chat_id))
        else:
            all_results.append(_send_single_message(text, chat_id))
    return all_results


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


def analyze_deal_direction(deal_explanation):
    if not deal_explanation:
        return None
    buy_count = deal_explanation.upper().count("BUY")
    sell_count = deal_explanation.upper().count("SELL")
    if buy_count > sell_count:
        return "bullish"
    elif sell_count > buy_count:
        return "bearish"
    else:
        return "mixed"


def score_signal(volume_ratio, z_score, relevant_article_count, sentiment_score,
                  deal_direction, relative_strength, stock_change_pct):
    has_news = relevant_article_count >= 1
    diverging = relative_strength in ("outperforming market", "underperforming market")
    move_size = abs(stock_change_pct) if stock_change_pct is not None else 0
    strong_zscore = (z_score is not None) and (z_score >= MIN_ZSCORE_FOR_HIGH)

    price_up = (stock_change_pct or 0) > 0
    deal_aligned = (
        (deal_direction == "bullish" and price_up) or
        (deal_direction == "bearish" and not price_up)
    )

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
    elif deal_aligned:
        if diverging and (volume_ratio >= 3.0 or strong_zscore):
            return "DEAL_CONFIRMED", "HIGH"
        else:
            return "DEAL_CONFIRMED", "MEDIUM"
    else:
        if (volume_ratio >= UNCONFIRMED_HIGH_MIN_VOLUME
                and move_size >= UNCONFIRMED_HIGH_MIN_MOVE_PCT
                and diverging):
            return "UNCONFIRMED", "HIGH"
        elif volume_ratio >= UNCONFIRMED_MEDIUM_MIN_VOLUME and diverging:
            return "UNCONFIRMED", "MEDIUM"
        else:
            return "UNCONFIRMED", "LOW"


def determine_direction(stock_change_pct, relative_strength, sentiment_label, signal_type, deal_direction):
    if stock_change_pct is None:
        return "WATCH", "Insufficient price data"

    price_up = stock_change_pct > 0
    outperforming = relative_strength == "outperforming market"
    underperforming = relative_strength == "underperforming market"

    if signal_type == "CONFIRMED":
        sentiment_agrees_up = sentiment_label == "positive"
        sentiment_agrees_down = sentiment_label == "negative"
        if price_up and outperforming and sentiment_agrees_up:
            return "BUY", "Price rising, outperforming Nifty, positive news sentiment - all evidence aligned"
        elif not price_up and underperforming and sentiment_agrees_down:
            return "SELL/AVOID", "Price falling, underperforming Nifty, negative news sentiment - all evidence aligned"
        else:
            return "WATCH", "Evidence is mixed (price, market divergence, and sentiment don't all agree)"
    elif signal_type == "DEAL_CONFIRMED":
        if price_up and outperforming and deal_direction == "bullish":
            return "BUY", "Price rising, outperforming Nifty, and disclosed deals show net buying - real institutional activity confirms this move"
        elif not price_up and underperforming and deal_direction == "bearish":
            return "SELL/AVOID", "Price falling, underperforming Nifty, and disclosed deals show net selling - real institutional activity confirms this move"
        else:
            return "WATCH", "Deal direction and price/market divergence don't fully agree"
    else:
        if price_up and outperforming:
            return "BUY (unconfirmed)", "Price rising with high volume, outperforming Nifty - but no news or deal backing this move"
        elif not price_up and underperforming:
            return "SELL/AVOID (unconfirmed)", "Price falling with high volume, underperforming Nifty - but no news or deal backing this move"
        else:
            return "WATCH", "Volume is unusual but price/market direction is unclear"


def compute_composite_score(volume_ratio, z_score, relevant_article_count, sentiment_score,
                             relative_strength, signal_type):
    """
    Rebalanced so every signal type can theoretically reach 100 -
    CONFIRMED and DEAL_CONFIRMED get credit for their specific evidence
    type, UNCONFIRMED gets more weight on volume/z-score/divergence
    since it has no news or deal evidence to draw from.
    """
    score = 0

    if signal_type == "UNCONFIRMED":
        # No news/deal evidence available - weight volume/z-score/divergence more heavily
        score += min(volume_ratio / 20 * 40, 40)
        if z_score is not None:
            score += min(max(z_score, 0) / 10 * 20, 20)
        if relative_strength in ("outperforming market", "underperforming market"):
            score += 40
    elif signal_type == "DEAL_CONFIRMED":
        score += min(volume_ratio / 20 * 25, 25)
        if z_score is not None:
            score += min(max(z_score, 0) / 10 * 10, 10)
        score += 40  # real disclosed institutional activity - strong evidence
        if relative_strength in ("outperforming market", "underperforming market"):
            score += 25
    else:  # CONFIRMED
        score += min(volume_ratio / 20 * 20, 20)
        if z_score is not None:
            score += min(max(z_score, 0) / 10 * 10, 10)
        score += min(relevant_article_count / 3 * 15, 15)
        score += min(abs(sentiment_score) / 3 * 15, 15)
        score += 20
        if relative_strength in ("outperforming market", "underperforming market"):
            score += 20

    return round(min(score, 100))


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
        print(f"[{i+1}/{len(top_movers)}] {ticker_clean} (volume ratio {stock['ratio']}x, z={stock.get('z_score')})...")

        deal_explanation = check_stock_for_deals(ticker_clean, deals_dict)
        deal_direction = analyze_deal_direction(deal_explanation)
        headlines = fetch_relevant_news(ticker_clean)
        sentiment = get_sentiment_for_headlines(headlines)
        price = get_current_price(ticker_clean)
        stock_change = get_daily_change_pct(f"{ticker_clean}.NS")
        relative_strength = classify_relative_strength(stock_change, nifty_change)

        signal_type, confidence = score_signal(
            stock["ratio"], stock.get("z_score"), len(headlines), sentiment["score"],
            deal_direction, relative_strength, stock_change
        )

        direction, direction_reason = determine_direction(
            stock_change, relative_strength, sentiment["label"], signal_type, deal_direction
        )

        composite_score = compute_composite_score(
            stock["ratio"], stock.get("z_score"), len(headlines), sentiment["score"],
            relative_strength, signal_type
        )

        # Candle analysis - validate direction against actual price action structure
        candle_data = analyze_candle(ticker_clean)
        candle_quality = None
        trend_note = None
        if candle_data:
            trend_note = candle_data.get("trend_note")
            if "BUY" in direction:
                candle_quality = candle_data["buy_quality"]
            elif "SELL" in direction or "AVOID" in direction:
                candle_quality = candle_data["sell_quality"]

        risk_info = {
            "stop_loss_pct": None, "liquidity_risk": None, "stop_loss_price": None,
            "target_price": None, "rsi": None, "rsi_status": None, "day_range_note": None
        }
        if confidence == "HIGH" and price:
            vol_data = get_volatility_stop_loss(ticker_clean)
            liq_data = check_liquidity_risk(ticker_clean)
            rsi_data = calculate_rsi(ticker_clean)
            range_data = get_day_range_position(ticker_clean)

            if vol_data:
                sl_pct = vol_data["suggested_stop_loss_pct"]
                risk_info["stop_loss_pct"] = sl_pct
                risk_info["stop_loss_price"] = round(price * (1 - sl_pct / 100), 2)
                risk_info["target_price"] = round(price * (1 + (sl_pct * TARGET_RATIO) / 100), 2)
            if liq_data:
                risk_info["liquidity_risk"] = liq_data["liquidity_risk"]
            if rsi_data:
                risk_info["rsi"] = rsi_data["rsi"]
                risk_info["rsi_status"] = rsi_data["status"]
            if range_data:
                risk_info["day_range_note"] = range_data["note"]

            # Downgrade: overbought (evidence-based)
            if "BUY" in direction and rsi_data:
                if rsi_data["rsi"] >= 70:
                    confidence = "MEDIUM"
                    direction_reason += " [Downgraded: overbought (RSI >= 70) - elevated pullback risk based on prior accuracy data]"

            # Downgrade: extreme volume/z-score (evidence-based - HIGH tier underperformed MEDIUM in accuracy data)
            is_extreme = (stock["ratio"] >= EXTREME_VOLUME_RATIO) or \
                         (stock.get("z_score") is not None and stock.get("z_score") >= EXTREME_ZSCORE)
            if confidence == "HIGH" and is_extreme:
                confidence = "MEDIUM"
                direction_reason += " [Downgraded: extreme volume/z-score - historical data shows elevated reversal risk]"

            # Downgrade: candle structure contradicts direction
            if candle_quality and ("contradicts" in candle_quality or "WEAK" in candle_quality):
                if confidence == "HIGH":
                    confidence = "MEDIUM"
                    direction_reason += f" [Downgraded: candle warning - {candle_quality}]"

        final_signals.append({
            "ticker": ticker_clean,
            "volume_ratio": stock["ratio"],
            "z_score": stock.get("z_score"),
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "headline_count": len(headlines),
            "sample_headline": headlines[0] if headlines else None,
            "price_at_signal": price,
            "stock_change_pct": stock_change,
            "relative_strength": relative_strength,
            "signal_type": signal_type,
            "confidence": confidence,
            "direction": direction,
            "direction_reason": direction_reason,
            "composite_score": composite_score,
            "deal_explanation": deal_explanation,
            "deal_direction": deal_direction,
            "stop_loss_pct": risk_info["stop_loss_pct"],
            "stop_loss_price": risk_info["stop_loss_price"],
            "target_price": risk_info["target_price"],
            "liquidity_risk": risk_info["liquidity_risk"],
            "rsi": risk_info["rsi"],
            "rsi_status": risk_info["rsi_status"],
            "day_range_note": risk_info["day_range_note"],
            "candle_quality": candle_quality,
            "trend_note": trend_note
        })

        time.sleep(0.3)

    confirmed_all = [s for s in final_signals if s["signal_type"] in ("CONFIRMED", "DEAL_CONFIRMED") and s["composite_score"] >= CONFIRMED_MIN_SCORE_THRESHOLD]
    unconfirmed_high = [s for s in final_signals if s["signal_type"] == "UNCONFIRMED" and s["composite_score"] >= MIN_SCORE_THRESHOLD]

    print(f"\n=== STAGE 5: {len(confirmed_all)} CONFIRMED (>=90), {len(unconfirmed_high)} UNCONFIRMED (>=80) today ===")

    print("\n=== STAGE 6: Sending CONFIRMED signal message ===")
    if confirmed_all:
        send_telegram_message(format_confirmed_message(confirmed_all, nifty_change))
    else:
        send_telegram_message(f"[{RUN_LABEL}] No CONFIRMED signals today (Nifty: {nifty_change}%).")

    print("\n=== STAGE 7: Sending UNCONFIRMED signal message ===")
    if unconfirmed_high:
        send_telegram_message(format_unconfirmed_message(unconfirmed_high, nifty_change))
    else:
        send_telegram_message(f"[{RUN_LABEL}] No UNCONFIRMED signals today (Nifty: {nifty_change}%).")

    print("\n=== STAGE 8: Logging ALL signals for tracking ===")
    log_signals_to_csv(final_signals, nifty_change)

    return final_signals


def format_evidence_block(s):
    lines = []
    dir_emoji = "[BUY]" if "BUY" in s["direction"] else "[SELL]" if "SELL" in s["direction"] else "[WATCH]"

    lines.append(f"{dir_emoji} {s['direction']}: {s['ticker']}  |  Score: {s['composite_score']}/100")
    lines.append(f"Type: {s['signal_type']}")
    lines.append(f"Reason: {s['direction_reason']}")
    lines.append(f"Entry: Rs.{s['price_at_signal']}")

    if s["stop_loss_price"]:
        lines.append(f"Stop-loss: Rs.{s['stop_loss_price']} (-{s['stop_loss_pct']}%)")
    if s["target_price"]:
        target_pct = round(s["stop_loss_pct"] * TARGET_RATIO, 2)
        lines.append(f"Target: Rs.{s['target_price']} (+{target_pct}%)")
    if s["liquidity_risk"]:
        lines.append(f"Liquidity: {s['liquidity_risk']}")
    if s["rsi"]:
        lines.append(f"RSI: {s['rsi']} ({s['rsi_status']})")
    if s["day_range_note"]:
        lines.append(f"Day range: {s['day_range_note']}")
    if s["candle_quality"]:
        lines.append(f"Candle: {s['candle_quality']}")
    if s["trend_note"]:
        lines.append(f"5-day trend: {s['trend_note']}")

    lines.append("Evidence:")
    lines.append(f"  - Volume: {s['volume_ratio']}x normal average (z-score: {s['z_score']})")

    change_display = f"{s['stock_change_pct']:.2f}%" if s['stock_change_pct'] is not None and not pd.isna(s['stock_change_pct']) else "unavailable"
    lines.append(f"  - Price move: {change_display} ({s['relative_strength']})")

    if s["signal_type"] == "CONFIRMED":
        lines.append(f"  - News: {s['headline_count']} relevant article(s), sentiment {s['sentiment_label']}")
        if s["sample_headline"]:
            lines.append(f"  - \"{s['sample_headline']}\"")
    elif s["signal_type"] == "DEAL_CONFIRMED":
        lines.append(f"  - Disclosed deal activity: {s['deal_direction']}")
        if s["deal_explanation"]:
            lines.append(f"  - {s['deal_explanation'][:200]}")
    else:
        lines.append(f"  - News: none found (unconfirmed - higher risk)")

    return "\n".join(lines)


def format_confirmed_message(signals, nifty_change):
    lines = [f"[{RUN_LABEL}] CONFIRMED SIGNALS (Nifty: {nifty_change}%)\n"]
    lines.append("Backed by real news or disclosed deal activity. Score >= 90.\n")
    for s in signals:
        lines.append(format_evidence_block(s))
        lines.append("")
    lines.append("Decision-support only, not financial advice. Verify independently.")
    return "\n".join(lines)


def format_unconfirmed_message(signals, nifty_change):
    lines = [f"[{RUN_LABEL}] UNCONFIRMED SIGNALS (Nifty: {nifty_change}%)\n"]
    lines.append("Volume/price only - no news or deal corroboration. Higher risk, verify yourself.\n")
    for s in signals:
        lines.append(format_evidence_block(s))
        lines.append("")
    lines.append("Decision-support only, not financial advice. Verify independently before acting.")
    return "\n".join(lines)


def log_signals_to_csv(signals, nifty_change):
    SIGNALS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SIGNALS_LOG_PATH.exists()

    with open(SIGNALS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "signal_date", "ticker", "volume_ratio", "z_score", "sentiment_label", "sentiment_score",
                "headline_count", "sample_headline", "price_at_signal", "stock_change_pct",
                "nifty_change_pct", "relative_strength", "signal_type", "confidence", "direction",
                "direction_reason", "composite_score", "deal_explanation", "deal_direction", "stop_loss_pct",
                "stop_loss_price", "target_price", "liquidity_risk", "rsi", "rsi_status", "day_range_note",
                "candle_quality", "trend_note",
                "checked_1d", "checked_3d", "checked_7d", "price_1d", "price_3d", "price_7d"
            ])
        today = datetime.now().strftime("%Y-%m-%d")
        for s in signals:
            writer.writerow([
                today, s["ticker"], s["volume_ratio"], s["z_score"], s["sentiment_label"], s["sentiment_score"],
                s["headline_count"], s["sample_headline"] or "", s["price_at_signal"], s["stock_change_pct"],
                nifty_change, s["relative_strength"], s["signal_type"], s["confidence"], s["direction"],
                s["direction_reason"], s["composite_score"], s["deal_explanation"] or "", s["deal_direction"] or "",
                s["stop_loss_pct"] or "", s["stop_loss_price"] or "", s["target_price"] or "",
                s["liquidity_risk"] or "", s["rsi"] or "", s["rsi_status"] or "", s["day_range_note"] or "",
                s["candle_quality"] or "", s["trend_note"] or "",
                "", "", "", "", "", ""
            ])
    print(f"Logged {len(signals)} signals to {SIGNALS_LOG_PATH}")


if __name__ == "__main__":
    signals = run_full_pipeline()
    print("\n=== FINAL SUMMARY ===")
    for s in signals:
        print(f"{s['ticker']}: {s['direction']} type={s['signal_type']} score={s['composite_score']}")


