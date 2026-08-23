"""
Accuracy Checker
----------------
Goes through signals_log.csv, finds signals old enough to check (1d/3d/7d),
and records the price AS OF that specific historical date - not just
"today's current price" reused across all horizons.

Win/loss is judged per-signal based on its own direction:
- BUY signals: win if price went UP
- SELL/AVOID signals: win if price went DOWN
- WATCH signals: excluded from win-rate (no directional call made)
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_LOG_PATH = PROJECT_ROOT / "data" / "signals_log.csv"


def get_price_on_or_after(ticker, target_date):
    """
    Fetch the closing price for the first available trading day on or
    after target_date - a true historical snapshot, not today's live price.
    """
    try:
        start = target_date.strftime("%Y-%m-%d")
        end = (target_date + timedelta(days=5)).strftime("%Y-%m-%d")
        data = yf.download(f"{ticker}.NS", start=start, end=end, interval="1d", progress=False)
        if data.empty:
            return None
        close_col = data["Close"]
        if close_col.ndim > 1:
            close_col = close_col.iloc[:, 0]
        return float(close_col.iloc[0])  # first available trading day on/after target
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

    for col in ["checked_1d", "checked_3d", "checked_7d"]:
        df[col] = df[col].astype("object")

    updated_count = 0

    for idx, row in df.iterrows():
        age = days_since(str(row["signal_date"]))
        ticker = row["ticker"]
        signal_date = datetime.strptime(str(row["signal_date"]), "%Y-%m-%d")

        if age >= 1 and (pd.isna(row.get("checked_1d")) or row.get("checked_1d") == ""):
            price = get_price_on_or_after(ticker, signal_date + timedelta(days=1))
            if price is not None:
                df.at[idx, "price_1d"] = price
                df.at[idx, "checked_1d"] = "done"
                updated_count += 1

        if age >= 3 and (pd.isna(row.get("checked_3d")) or row.get("checked_3d") == ""):
            price = get_price_on_or_after(ticker, signal_date + timedelta(days=3))
            if price is not None:
                df.at[idx, "price_3d"] = price
                df.at[idx, "checked_3d"] = "done"
                updated_count += 1

        if age >= 7 and (pd.isna(row.get("checked_7d")) or row.get("checked_7d") == ""):
            price = get_price_on_or_after(ticker, signal_date + timedelta(days=7))
            if price is not None:
                df.at[idx, "price_7d"] = price
                df.at[idx, "checked_7d"] = "done"
                updated_count += 1

    df.to_csv(SIGNALS_LOG_PATH, index=False)
    print(f"Updated {updated_count} price checkpoints across {len(df)} logged signals.")


def is_win(row, move_pct):
    direction = str(row.get("direction", "")).upper()
    if "BUY" in direction:
        return move_pct > 0
    elif "SELL" in direction or "AVOID" in direction:
        return move_pct < 0
    else:
        return None


def summarize_accuracy():
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
            tier = subset[subset["confidence"] == conf].copy()
            if tier.empty:
                continue

            avg_move = tier["move_pct"].mean()

            if "direction" in tier.columns:
                tier["win"] = tier.apply(lambda r: is_win(r, r["move_pct"]), axis=1)
                directional = tier[tier["win"].notna()]
                if len(directional) > 0:
                    win_rate = directional["win"].mean() * 100
                    print(f"  {conf}: {len(tier)} signals | avg move: {avg_move:.2f}% | "
                          f"directional win rate: {win_rate:.1f}% ({len(directional)} had a clear BUY/SELL call)")
                else:
                    print(f"  {conf}: {len(tier)} signals | avg move: {avg_move:.2f}% | "
                          f"all signals were WATCH (no directional call)")


if __name__ == "__main__":
    print("=== Updating accuracy checkpoints (using true historical prices) ===")
    update_accuracy()

    print("\n=== Accuracy Summary ===")
    summarize_accuracy()
