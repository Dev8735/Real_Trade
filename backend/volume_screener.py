"""
Volume Screener
----------------
Scans ALL NSE-listed stocks for unusual volume (today vs recent average),
and returns the top N movers by volume ratio.

Runs in batches to keep yfinance requests manageable.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TICKER_CACHE = PROJECT_ROOT / "data" / "nse_tickers.csv"

BATCH_SIZE = 50
VOLUME_LOOKBACK_PERIOD = "10d"


def load_tickers(limit=None):
    df = pd.read_csv(TICKER_CACHE)
    symbols = df["SYMBOL"].tolist()
    if limit:
        symbols = symbols[:limit]
    return symbols


def compute_volume_ratio(hist_df, ticker, debug=False):
    """Given a multi-ticker yfinance history dataframe, compute volume ratio for one ticker."""
    try:
        volume_series = hist_df[ticker]["Volume"].dropna()

        if len(volume_series) < 2:
            if debug:
                print(f"  {ticker}: not enough data points ({len(volume_series)})")
            return None

        today_volume = float(volume_series.iloc[-1])
        avg_volume = float(volume_series.iloc[:-1].mean())

        if avg_volume <= 0:
            return None

        ratio = today_volume / avg_volume
        return {
            "ticker": ticker,
            "today_volume": today_volume,
            "avg_volume": avg_volume,
            "ratio": round(ratio, 2)
        }
    except KeyError as e:
        if debug:
            print(f"  {ticker}: KeyError - {e}")
        return None
    except Exception as e:
        if debug:
            print(f"  {ticker}: {type(e).__name__} - {e}")
        return None


def scan_batch(symbols_batch, debug=False):
    yf_tickers = [f"{s}.NS" for s in symbols_batch]
    results = []

    try:
        data = yf.download(
            tickers=yf_tickers,
            period=VOLUME_LOOKBACK_PERIOD,
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception as e:
        print(f"Batch download failed: {e}")
        return results

    if debug:
        print("  Sample columns:", list(data.columns)[:6])

    for yf_ticker in yf_tickers:
        result = compute_volume_ratio(data, yf_ticker, debug=debug)
        if result:
            results.append(result)

    return results


def scan_all_stocks(limit=None, top_n=30, debug=False):
    symbols = load_tickers(limit=limit)
    all_results = []

    total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Scanning {len(symbols)} stocks in {total_batches} batches of {BATCH_SIZE}...")

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} tickers)...")

        batch_results = scan_batch(batch, debug=debug)
        all_results.extend(batch_results)

        time.sleep(1)

    all_results.sort(key=lambda x: x["ratio"], reverse=True)
    top_movers = all_results[:top_n]

    print(f"\nSuccessfully scanned {len(all_results)} of {len(symbols)} stocks.")
    return top_movers


if __name__ == "__main__":
    # debug=True on first batch only, to see what's happening
    top_movers = scan_all_stocks(limit=100, top_n=30, debug=True)

    print(f"\n=== Top {len(top_movers)} volume movers (test batch of 100) ===")
    for stock in top_movers:
        print(f"{stock['ticker']}: ratio={stock['ratio']}x  (today={stock['today_volume']:.0f}, avg={stock['avg_volume']:.0f})")
