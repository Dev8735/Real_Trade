"""
Market Context (Nifty Comparison)
-----------------------------------
Compares each stock's daily price move against the Nifty 50 index,
so we can tell genuine stock-specific moves apart from market-wide moves.
"""

import yfinance as yf

NIFTY_TICKER = "^NSEI"


def get_daily_change_pct(ticker):
    """Get today's % change (close vs previous close) for any ticker."""
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if data.empty or len(data) < 2:
            return None

        close_col = data["Close"]
        if close_col.ndim > 1:
            close_col = close_col.iloc[:, 0]

        today_close = float(close_col.iloc[-1])
        prev_close = float(close_col.iloc[-2])

        if prev_close == 0:
            return None

        return round(((today_close - prev_close) / prev_close) * 100, 2)
    except Exception as e:
        print(f"  Failed to get change for {ticker}: {e}")
        return None


def get_nifty_change():
    """Get today's Nifty 50 index % change."""
    return get_daily_change_pct(NIFTY_TICKER)


def classify_relative_strength(stock_change_pct, nifty_change_pct):
    """
    Compare a stock's move against the index to judge if it's
    genuinely diverging from the market or just following it.
    """
    if stock_change_pct is None or nifty_change_pct is None:
        return "unknown"

    relative_move = stock_change_pct - nifty_change_pct

    if abs(relative_move) < 0.5:
        return "in-line with market"
    elif relative_move >= 0.5:
        return "outperforming market"
    else:
        return "underperforming market"


if __name__ == "__main__":
    nifty_change = get_nifty_change()
    print(f"Nifty 50 today: {nifty_change}%")

    test_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    for t in test_tickers:
        change = get_daily_change_pct(t)
        relative = classify_relative_strength(change, nifty_change)
        print(f"{t}: {change}% ({relative})")
