"""
Risk Manager
------------
Adds risk-management context to signals: volatility-based stop-loss
reference, liquidity risk flag, and a position-sizing calculator.

This is a decision-support framework, NOT financial advice.
It does not tell you what to buy/sell or how much - it gives you
data to make that decision yourself, based on your own risk tolerance.
"""

import yfinance as yf
import pandas as pd


def get_volatility_stop_loss(ticker, lookback_days=14):
    """
    Calculate a stop-loss reference based on Average True Range (ATR) -
    a standard volatility measure. Wider stop for more volatile stocks,
    tighter stop for calmer ones.
    """
    try:
        data = yf.download(f"{ticker}.NS", period=f"{lookback_days + 5}d", interval="1d", progress=False)
        if data.empty or len(data) < lookback_days:
            return None

        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        if high.ndim > 1:
            high = high.iloc[:, 0]
            low = low.iloc[:, 0]
            close = close.iloc[:, 0]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=lookback_days).mean().iloc[-1]
        last_price = float(close.iloc[-1])

        if pd.isna(atr) or last_price == 0:
            return None

        atr_pct = (float(atr) / last_price) * 100

        return {
            "atr_value": round(float(atr), 2),
            "atr_pct_of_price": round(atr_pct, 2),
            "suggested_stop_loss_pct": round(atr_pct * 1.5, 2),  # 1.5x ATR is a common convention
            "last_price": last_price
        }
    except Exception as e:
        print(f"  Failed volatility calc for {ticker}: {e}")
        return None


def check_liquidity_risk(ticker, min_avg_volume=100000, lookback_days=10):
    """
    Flag stocks with low average volume - harder to exit a position
    without moving the price against yourself.
    """
    try:
        data = yf.download(f"{ticker}.NS", period=f"{lookback_days}d", interval="1d", progress=False)
        if data.empty:
            return {"liquidity_risk": "unknown", "avg_volume": None}

        volume = data["Volume"]
        if volume.ndim > 1:
            volume = volume.iloc[:, 0]

        avg_volume = float(volume.mean())

        if avg_volume < min_avg_volume:
            risk = "HIGH - low liquidity, exit may be difficult"
        elif avg_volume < min_avg_volume * 3:
            risk = "MEDIUM - moderate liquidity"
        else:
            risk = "LOW - adequate liquidity"

        return {"liquidity_risk": risk, "avg_volume": round(avg_volume)}
    except Exception as e:
        print(f"  Failed liquidity check for {ticker}: {e}")
        return {"liquidity_risk": "unknown", "avg_volume": None}


def calculate_position_size(capital, risk_per_trade_pct, entry_price, stop_loss_pct):
    """
    Classic risk-based position sizing: risk only a small % of total capital
    per trade, sized so that if the stop-loss is hit, the loss equals that %.

    This is a calculator, not a recommendation - risk_per_trade_pct is
    entirely the user's own choice (commonly 1-2% is used as a conservative
    reference point, but that's a personal risk decision, not a rule).
    """
    if entry_price <= 0 or stop_loss_pct <= 0:
        return None

    risk_amount = capital * (risk_per_trade_pct / 100)
    stop_loss_amount_per_share = entry_price * (stop_loss_pct / 100)

    if stop_loss_amount_per_share == 0:
        return None

    shares = int(risk_amount / stop_loss_amount_per_share)
    position_value = shares * entry_price

    return {
        "shares": shares,
        "position_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "stop_loss_price": round(entry_price * (1 - stop_loss_pct / 100), 2)
    }


if __name__ == "__main__":
    test_ticker = "KROSS"

    print(f"=== Risk analysis for {test_ticker} ===\n")

    vol = get_volatility_stop_loss(test_ticker)
    print("Volatility / Stop-loss reference:")
    print(vol)

    liq = check_liquidity_risk(test_ticker)
    print("\nLiquidity check:")
    print(liq)

    if vol:
        print("\nExample position sizing (illustrative only - use your own numbers):")
        example = calculate_position_size(
            capital=100000,
            risk_per_trade_pct=1,
            entry_price=vol["last_price"],
            stop_loss_pct=vol["suggested_stop_loss_pct"]
        )
        print(example)
