"""
Positions Manager
------------------
Tracks positions you have actually entered (not just signals - only
stocks you have genuinely bought). Used by exit_evaluator.py to monitor
for exit conditions.
"""

import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSITIONS_PATH = PROJECT_ROOT / "data" / "positions.csv"

REWARD_RISK_RATIO = 1.5  # target = stop_loss_pct * 1.5 (conservative)
DEFAULT_MAX_HOLD_DAYS = 10


def add_position(ticker, entry_price, stop_loss_pct, max_hold_days=DEFAULT_MAX_HOLD_DAYS):
    """Log a new position you have actually entered."""
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = POSITIONS_PATH.exists()

    target_pct = round(stop_loss_pct * REWARD_RISK_RATIO, 2)
    stop_loss_price = round(entry_price * (1 - stop_loss_pct / 100), 2)
    target_price = round(entry_price * (1 + target_pct / 100), 2)

    with open(POSITIONS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "ticker", "entry_date", "entry_price", "stop_loss_pct", "stop_loss_price",
                "target_pct", "target_price", "max_hold_days", "highest_price_since_entry", "status"
            ])
        writer.writerow([
            ticker, datetime.now().strftime("%Y-%m-%d"), entry_price, stop_loss_pct, stop_loss_price,
            target_pct, target_price, max_hold_days, entry_price, "OPEN"
        ])

    print(f"Position logged: {ticker} @ {entry_price}")
    print(f"  Stop-loss: {stop_loss_price} (-{stop_loss_pct}%)")
    print(f"  Target: {target_price} (+{target_pct}%)")
    print(f"  Max hold: {max_hold_days} days")


if __name__ == "__main__":
    print("=== Add a new position ===")
    ticker = input("Ticker (e.g. KROSS): ").strip().upper()
    entry_price = float(input("Entry price: ").strip())
    stop_loss_pct = float(input("Stop-loss % (e.g. 5.28 from the signal alert): ").strip())
    max_hold = input(f"Max hold days [{DEFAULT_MAX_HOLD_DAYS}]: ").strip()
    max_hold_days = int(max_hold) if max_hold else DEFAULT_MAX_HOLD_DAYS

    add_position(ticker, entry_price, stop_loss_pct, max_hold_days)
