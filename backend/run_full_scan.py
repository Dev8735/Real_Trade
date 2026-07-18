"""
Run a full NSE volume scan and print the top movers.
"""

from volume_screener import scan_all_stocks

if __name__ == "__main__":
    top_movers = scan_all_stocks(limit=None, top_n=30, debug=False)

    print(f"\n=== Top {len(top_movers)} volume movers (FULL NSE scan) ===")
    for stock in top_movers:
        print(f"{stock['ticker']}: ratio={stock['ratio']}x (today={stock['today_volume']:.0f}, avg={stock['avg_volume']:.0f})")
