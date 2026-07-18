"""
NSE Ticker List Fetcher
------------------------
Downloads the full list of NSE-listed equity symbols and caches it locally,
so we don't hit NSE's servers on every run.
"""

import requests
import pandas as pd
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "nse_tickers.csv"

NSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def fetch_nse_tickers(force_refresh=False):
    """
    Fetch the full NSE equity list. Uses a local cache unless force_refresh=True
    or the cache doesn't exist yet.
    """
    if CACHE_PATH.exists() and not force_refresh:
        print(f"Using cached ticker list: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    print("Fetching fresh NSE ticker list...")
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.headers.update(headers)

    response = session.get(NSE_URL, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved {len(df)} tickers to {CACHE_PATH}")

    return df


if __name__ == "__main__":
    df = fetch_nse_tickers(force_refresh=True)
    print(f"\nTotal tickers: {len(df)}")
    print(df.head(10))
    print("\nColumns:", list(df.columns))
