"""
Block/Bulk Deal Checker
-------------------------
Cross-checks today's flagged stocks against NSE's disclosed block and bulk
deals. If a stock's volume spike is explained by a known large trade,
we flag it as "explained" rather than a mysterious pattern.
"""

import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://www.nseindia.com"
DEAL_API_URL = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"

DATA_KEYS = {
    "bulk_deals": "BULK_DEALS_DATA",
    "block_deals": "BLOCK_DEALS_DATA",
}


def get_nse_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(BASE_URL, timeout=10)
    return session


def fetch_deals(band_type="bulk_deals"):
    session = get_nse_session()
    try:
        resp = session.get(DEAL_API_URL, params={"bandtype": band_type}, timeout=10)
        if resp.status_code != 200:
            print(f"  NSE API returned status {resp.status_code} for {band_type}")
            return []
        data = resp.json()
        key = DATA_KEYS.get(band_type, "data")
        return data.get(key, [])
    except Exception as e:
        print(f"  Failed to fetch {band_type}: {e}")
        return []


def get_todays_deal_symbols():
    all_deals = {}
    for band_type in ["bulk_deals", "block_deals"]:
        deals = fetch_deals(band_type)
        for deal in deals:
            symbol = deal.get("symbol")
            if not symbol:
                continue
            all_deals.setdefault(symbol, []).append({
                "type": band_type,
                "client_name": deal.get("clientName", "Unknown"),
                "quantity": deal.get("qty"),
                "trade_price": deal.get("watp"),
                "buy_sell": deal.get("buySell", "Unknown"),
            })
    return all_deals


def check_stock_for_deals(ticker_symbol, deals_dict):
    matches = deals_dict.get(ticker_symbol, [])
    if not matches:
        return None
    summary_lines = []
    for m in matches:
        summary_lines.append(
            f"{m['type']}: {m['buy_sell']} {m['quantity']} shares @ {m['trade_price']} by {m['client_name']}"
        )
    return "; ".join(summary_lines)


if __name__ == "__main__":
    print("Fetching today's block/bulk deals from NSE...")
    deals = get_todays_deal_symbols()
    print(f"Found deals for {len(deals)} stocks today.\n")

    for symbol, deal_list in list(deals.items())[:10]:
        print(f"{symbol}:")
        for d in deal_list:
            print(f"    {d['type']}: {d['buy_sell']} {d['quantity']} shares @ {d['trade_price']} ({d['client_name']})")
