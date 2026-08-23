"""
Candle Analysis
----------------
Analyzes today's candle structure AND the recent multi-day trend
(last 5 trading days) to validate whether a BUY/SELL signal is
genuinely supported by how price has actually behaved recently -
not just today in isolation.

This describes observed patterns, not predictions - no chart pattern
guarantees future performance.
"""

import yfinance as yf


def analyze_candle(ticker, lookback_days=5):
    try:
        data = yf.download(f"{ticker}.NS", period=f"{lookback_days + 3}d", interval="1d", progress=False)
        if data.empty or len(data) < 2:
            return None

        opens = data["Open"]
        highs = data["High"]
        lows = data["Low"]
        closes = data["Close"]
        volumes = data["Volume"]

        if opens.ndim > 1:
            opens, highs, lows, closes, volumes = opens.iloc[:, 0], highs.iloc[:, 0], lows.iloc[:, 0], closes.iloc[:, 0], volumes.iloc[:, 0]

        # Today's candle
        open_p, high_p, low_p, close_p = float(opens.iloc[-1]), float(highs.iloc[-1]), float(lows.iloc[-1]), float(closes.iloc[-1])
        full_range = high_p - low_p

        if full_range == 0:
            return None

        is_green = close_p > open_p
        body_size = abs(close_p - open_p)
        body_pct_of_range = round((body_size / full_range) * 100, 1)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p
        upper_shadow_pct = round((upper_shadow / full_range) * 100, 1)
        lower_shadow_pct = round((lower_shadow / full_range) * 100, 1)

        if is_green and upper_shadow_pct < 20 and body_pct_of_range > 50:
            buy_quality = "STRONG - buyers held control into close"
        elif is_green and upper_shadow_pct >= 40:
            buy_quality = "WEAK - large upper shadow, rally faded before close, sellers pushed back"
        elif is_green:
            buy_quality = "MODERATE"
        else:
            buy_quality = "RED CANDLE - closed below open, contradicts a BUY call"

        if not is_green and lower_shadow_pct < 20 and body_pct_of_range > 50:
            sell_quality = "STRONG - sellers held control into close"
        elif not is_green and lower_shadow_pct >= 40:
            sell_quality = "WEAK - large lower shadow, decline was bought back before close"
        elif not is_green:
            sell_quality = "MODERATE"
        else:
            sell_quality = "GREEN CANDLE - closed above open, contradicts a SELL call"

        # Multi-day trend (observed pattern, not a forecast)
        recent_closes = closes.iloc[-lookback_days:]
        recent_opens = opens.iloc[-lookback_days:]
        recent_volumes = volumes.iloc[-lookback_days:]

        green_days = int((recent_closes > recent_opens).sum())
        total_days = len(recent_closes)

        volume_trend = "rising" if recent_volumes.iloc[-1] > recent_volumes.iloc[0] else "falling"
        price_trend_pct = round(((recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]) * 100, 2)

        if green_days >= total_days - 1 and volume_trend == "rising":
            trend_note = f"Building: {green_days}/{total_days} green days, volume rising, {price_trend_pct:+.1f}% over period - consistent upward pattern"
        elif green_days <= 1 and volume_trend == "rising":
            trend_note = f"Building (downward): {total_days - green_days}/{total_days} red days, volume rising, {price_trend_pct:+.1f}% over period - consistent downward pattern"
        elif volume_trend == "falling":
            trend_note = f"Fading: volume declining over last {total_days} days - today's move may be losing steam, not building"
        else:
            trend_note = f"Mixed: {green_days}/{total_days} green days, no clear consistent pattern"

        return {
            "is_green": is_green,
            "body_pct_of_range": body_pct_of_range,
            "upper_shadow_pct": upper_shadow_pct,
            "lower_shadow_pct": lower_shadow_pct,
            "buy_quality": buy_quality,
            "sell_quality": sell_quality,
            "trend_note": trend_note
        }
    except Exception as e:
        print(f"  Failed candle analysis for {ticker}: {e}")
        return None


if __name__ == "__main__":
    test_ticker = "RELIANCE"
    result = analyze_candle(test_ticker)
    print(f"Candle analysis for {test_ticker}:")
    for k, v in result.items():
        print(f"  {k}: {v}")
