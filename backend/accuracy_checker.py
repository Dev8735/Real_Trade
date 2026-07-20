"""
Accuracy Checker
----------------
Goes through signals_log.csv, finds signals old enough to check (1d/3d/7d),
fetches current price, and records the actual price move.

Run this daily (can be a separate scheduled task) to build up real
performance data over time.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_LOG_PATH = PROJECT_ROOT / "data" / "signals_log.csv"


def get_price_now(ticker):
    """Fetch the latest available closing price for a ticker."""
    try:
        data = yf.download(f"{ticker}.NS", period="1d", interval="1d", progress=False)
        if data.empty:
            return None
        close_col = data["Close"]
        return float(close_col.iloc[-1]) if close_col.ndim == 1 else float(close_col.iloc[-1, 0])
    except Exception:
        return None


def days_since(signal_date_str):
    signal_date = datetime.strptime(signal_date_str, "%Y-%m-%d")
    return (datetime.now() - signal_date).days


def update_accuracy():
    if not SIGNALS_LOG_PATH.exists():
        print("No signals_log.csv found yet.")
        return

    df = pd.read_csv(SIGNALS_LOG_PATH)

    # Ensure checked columns are treated as strings, not floats/NaN
    for col in ["checked_1d", "checked_3d", "checked_7d"]:
        df[col] = df[col].astype("object")

    updated_count = 0

    for idx, row in df.iterrows():
        age = days_since(str(row["signal_date"]))
        ticker = row["ticker"]

        # 1-day check
        if age >= 1 and pd.isna(row.get("checked_1d")) or row.get("checked_1d") == "":
            price = get_price_now(ticker)
            if price is not None:
                df.at[idx, "price_1d"] = price
                df.at[idx, "checked_1d"] = "done"
                updated_count += 1

        # 3-day check
        if age >= 3 and (pd.isna(row.get("checked_3d")) or row.get("checked_3d") == ""):
            price = get_price_now(ticker)
            if price is not None:
                df.at[idx, "price_3d"] = price
                df.at[idx, "checked_3d"] = "done"
                updated_count += 1

        # 7-day check
        if age >= 7 and (pd.isna(row.get("checked_7d")) or row.get("checked_7d") == ""):
            price = get_price_now(ticker)
            if price is not None:
                df.at[idx, "price_7d"] = price
                df.at[idx, "checked_7d"] = "done"
                updated_count += 1

    df.to_csv(SIGNALS_LOG_PATH, index=False)
    print(f"Updated {updated_count} price checkpoints across {len(df)} logged signals.")


def summarize_accuracy():
    """Print a simple win-rate summary by confidence tier, where data exists."""
    if not SIGNALS_LOG_PATH.exists():
        print("No signals_log.csv found yet.")
        return

    df = pd.read_csv(SIGNALS_LOG_PATH)

    for horizon in ["1d", "3d", "7d"]:
        price_col = f"price_{horizon}"
        checked_col = f"checked_{horizon}"

        subset = df[df[checked_col] == "done"].copy()
        if subset.empty:
            print(f"\n--- {horizon} horizon: no data yet ---")
            continue

        subset["move_pct"] = ((subset[price_col] - subset["price_at_signal"]) / subset["price_at_signal"]) * 100

        print(f"\n--- {horizon} horizon ({len(subset)} signals checked) ---")
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            tier = subset[subset["confidence"] == conf]
            if tier.empty:
                continue
            avg_move = tier["move_pct"].mean()
            win_rate = (tier["move_pct"] > 0).mean() * 100 if tier["sentiment_label"].iloc[0] != "negative" else (tier["move_pct"] < 0).mean() * 100
            print(f"  {conf}: {len(tier)} signals | avg move: {avg_move:.2f}% | positive-move rate: {win_rate:.1f}%")


if __name__ == "__main__":
    print("=== Updating accuracy checkpoints ===")
    update_accuracy()

    print("\n=== Accuracy Summary ===")
    summarize_accuracy()
