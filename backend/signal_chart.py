"""
Signal Chart Generator
------------------------
For a given date, plots the price chart of each CONFIRMED (or DEAL_CONFIRMED)
signal from that day, marking entry price, stop-loss, and target levels,
so you can visually see how each signal actually played out.

Usage: python backend/signal_chart.py 2026-08-03
(defaults to today if no date given)
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_LOG_PATH = PROJECT_ROOT / "data" / "signals_log.csv"
CHARTS_OUTPUT_DIR = PROJECT_ROOT / "data" / "charts"


def get_confirmed_signals_for_date(target_date):
    if not SIGNALS_LOG_PATH.exists():
        print("No signals_log.csv found.")
        return pd.DataFrame()

    df = pd.read_csv(SIGNALS_LOG_PATH)
    day_signals = df[
        (df["signal_date"] == target_date) &
        (df["signal_type"].isin(["CONFIRMED", "DEAL_CONFIRMED"]))
    ]
    return day_signals


def plot_signal_chart(row, target_date):
    ticker = row["ticker"]
    entry_price = row["price_at_signal"]
    stop_loss = row.get("stop_loss_price")
    target = row.get("target_price")
    direction = row.get("direction", "")

    signal_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start = (signal_dt - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (signal_dt + timedelta(days=8)).strftime("%Y-%m-%d")

    try:
        data = yf.download(f"{ticker}.NS", start=start, end=end, interval="1d", progress=False)
    except Exception as e:
        print(f"  Failed to fetch chart data for {ticker}: {e}")
        return None

    if data.empty:
        print(f"  No price data available for {ticker}")
        return None

    close = data["Close"]
    if close.ndim > 1:
        close = close.iloc[:, 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, close, label="Close Price", color="steelblue", linewidth=2)

    ax.axhline(y=entry_price, color="black", linestyle="--", linewidth=1, label=f"Entry: {entry_price}")

    if pd.notna(stop_loss) and stop_loss:
        ax.axhline(y=stop_loss, color="red", linestyle="--", linewidth=1, label=f"Stop-loss: {stop_loss}")
    if pd.notna(target) and target:
        ax.axhline(y=target, color="green", linestyle="--", linewidth=1, label=f"Target: {target}")

    ax.axvline(x=signal_dt, color="gray", linestyle=":", linewidth=1, label="Signal Date")

    ax.set_title(f"{ticker} - Signal on {target_date} ({direction})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (Rs.)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    CHARTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_OUTPUT_DIR / f"{ticker}_{target_date}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return output_path


def generate_charts_for_date(target_date):
    signals = get_confirmed_signals_for_date(target_date)

    if signals.empty:
        print(f"No CONFIRMED signals found for {target_date}.")
        return []

    print(f"Found {len(signals)} CONFIRMED signal(s) for {target_date}. Generating charts...\n")

    generated = []
    for _, row in signals.iterrows():
        print(f"Charting {row['ticker']}...")
        path = plot_signal_chart(row, target_date)
        if path:
            generated.append(path)
            print(f"  Saved: {path}")

    return generated


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    print(f"Generating signal charts for {target_date}...\n")
    charts = generate_charts_for_date(target_date)
    print(f"\nDone. {len(charts)} chart(s) generated in data/charts/")
