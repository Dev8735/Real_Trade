# backend/backtest.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : For every day the model flagged as "highly suspicious" (Suspicion_Flag == 1),
#           measure the stock's actual price return over the FOLLOWING:
#               1 month  (~22 trading days)
#               3 months (~66 trading days)
#               6 months (~130 trading days)
#               1 year   (~252 trading days)
#
# WHY THIS IS USEFUL:
#   This is the most compelling empirical validation of the project.
#   If the model is genuinely detecting informed trading, then stocks
#   flagged before they move should show significantly higher returns
#   (or drops, for short-side insider activity) than the average stock
#   on an average day. This is called "forward return analysis" in
#   quantitative finance and is the standard way to evaluate a signal
#   without needing labeled insider-trading ground truth.
#
# TWO WAYS TO USE THIS FILE:
#   1. CLI script   — run it directly from terminal, prints a full report
#   2. API endpoint — import it into api.py to expose /backtest and
#                     /backtest/{ticker} endpoints the frontend can call
#
# HOW TO RUN AS A SCRIPT:
#   python backend/backtest.py                → all stocks
#   python backend/backtest.py RELIANCE       → one stock
#   python backend/backtest.py --min-score 70 → only days scoring >= 70
#
# HOW TO ADD TO api.py (add these 3 lines after the existing imports):
#   from backend.backtest import router as backtest_router
#   app.include_router(backtest_router)
#   → endpoints become: GET /backtest  and  GET /backtest/{ticker}
#
# DEPENDENCY:
#   data/processed/{ticker}_features.csv must exist with columns:
#   Date, Close, Suspicion_Score, Suspicion_Flag
#   Run the full pipeline (features.py → isolation_forest.py → scoring.py)
#   before running this file.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import argparse

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

PROCESSED_DIR = "data/processed"

# How many trading days forward to measure returns for each window
WINDOWS = {
    "1_month":  22,
    "3_month":  66,
    "6_month":  130,
    "1_year":   252,
}

# Only include flagged days where the score is AT OR ABOVE this value.
# Set lower to include more events; higher for only the most extreme flags.
DEFAULT_MIN_SCORE = 65

# Required columns the predictions CSV must contain
REQUIRED_COLS = ["Date", "Close", "Suspicion_Score", "Suspicion_Flag"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — load_predictions()
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(ticker: str) -> pd.DataFrame | None:
    """
    Loads the scored predictions CSV for one ticker.

    Parameters
    ----------
    ticker : str
        Clean ticker name, e.g. "RELIANCE".

    Returns
    -------
    pd.DataFrame sorted by Date ascending, or None if the file is missing
    or lacks the required columns.
    """
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=["Date"])

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None

    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — compute_forward_returns()
# ══════════════════════════════════════════════════════════════════════════════

def compute_forward_returns(df: pd.DataFrame, min_score: float = DEFAULT_MIN_SCORE) -> pd.DataFrame:
    """
    For every row where Suspicion_Score >= min_score, measures the stock's
    actual forward return over 1 month, 3 months, 6 months, and 1 year.

    HOW FORWARD RETURN IS COMPUTED:
        Forward_Return_N_days(t) = (Close[t + N] - Close[t]) / Close[t]

        Close[t]     = the closing price on the flagged day itself
        Close[t + N] = the closing price exactly N trading days later

        A positive return means the stock rose after the flag.
        A negative return means the stock fell after the flag.

        If t + N extends beyond the available data (e.g. a flag near the
        end of the dataset has no 1-year future yet), the return for that
        window is left as NaN rather than filled with a fabricated number.

    Parameters
    ----------
    df : pd.DataFrame
        Full scored stock history. Must contain Date, Close, Suspicion_Score,
        Suspicion_Flag columns.
    min_score : float
        Minimum Suspicion_Score to include a row as an "event". Defaults to 65.

    Returns
    -------
    pd.DataFrame with one row per flagged event, containing:
        Date, Suspicion_Score, Close_on_flag_day,
        Return_1M, Return_3M, Return_6M, Return_1Y  (all as percentages)
        Direction_1M  ('UP', 'DOWN', or 'N/A' if data unavailable)
    """
    close_arr = df["Close"].values
    n = len(df)

    events = df[df["Suspicion_Score"] >= min_score].copy()

    if events.empty:
        return pd.DataFrame(columns=[
            "Date", "Suspicion_Score", "Close_on_flag_day",
            "Return_1M_%", "Return_3M_%", "Return_6M_%", "Return_1Y_%",
            "Direction_1M"
        ])

    rows = []
    for pos, (idx, row) in enumerate(events.iterrows()):
        entry = {
            "Date": row["Date"].strftime("%Y-%m-%d"),
            "Suspicion_Score": row["Suspicion_Score"],
            "Close_on_flag_day": row["Close"],
        }

        flag_pos = df.index.get_loc(idx)
        flag_price = close_arr[flag_pos]

        for window_name, window_days in WINDOWS.items():
            future_pos = flag_pos + window_days
            col = f"Return_{window_name.upper().replace('_', '')}_%"

            if future_pos < n and flag_price > 0:
                future_price = close_arr[future_pos]
                ret = round((future_price - flag_price) / flag_price * 100, 2)
                entry[col] = ret
            else:
                entry[col] = None  # not enough future data yet

        # Direction label for the 1-month return
        r1m = entry.get("Return_1MONTH_%")
        if r1m is None:
            entry["Direction_1M"] = "N/A"
        elif r1m > 0:
            entry["Direction_1M"] = "UP"
        else:
            entry["Direction_1M"] = "DOWN"

        rows.append(entry)

    result = pd.DataFrame(rows)
    # Reorder columns cleanly
    col_order = [
        "Date", "Suspicion_Score", "Close_on_flag_day",
        "Return_1MONTH_%", "Return_3MONTH_%", "Return_6MONTH_%", "Return_1YEAR_%",
        "Direction_1M"
    ]
    col_order = [c for c in col_order if c in result.columns]
    result = result[col_order]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — compute_baseline_returns()
# ══════════════════════════════════════════════════════════════════════════════

def compute_baseline_returns(df: pd.DataFrame) -> dict:
    """
    Computes the average forward return across ALL trading days (not just
    flagged ones) as a baseline comparison.

    WHY THIS MATTERS:
        If flagged days return +8% over 3 months on average, that looks
        impressive — unless ALL days return +8% on average (a bull market).
        The baseline tells you whether the flagged days are genuinely
        outperforming the stock's own average behavior.

    Returns
    -------
    dict mapping window name -> average forward return % across all days
    that have enough future data for that window. NaN if no valid rows.
    """
    close_arr = df["Close"].values
    n = len(df)
    baseline = {}

    for window_name, window_days in WINDOWS.items():
        returns = []
        for i in range(n - window_days):
            if close_arr[i] > 0:
                ret = (close_arr[i + window_days] - close_arr[i]) / close_arr[i] * 100
                returns.append(ret)

        baseline[window_name] = round(np.mean(returns), 2) if returns else None

    return baseline


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — run_backtest()
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(ticker: str, min_score: float = DEFAULT_MIN_SCORE) -> dict | None:
    """
    Full backtest for one stock — loads data, computes forward returns for
    all flagged events, computes baseline, and returns a summary dict.

    Parameters
    ----------
    ticker : str
        Clean ticker name, e.g. "RELIANCE".
    min_score : float
        Minimum Suspicion_Score to qualify as a "flagged event".

    Returns
    -------
    dict with keys:
        ticker, total_days, flagged_events (count), min_score_used,
        events (list of dicts — one per flagged day with all forward returns),
        avg_returns (avg forward return per window across all flagged events),
        baseline_returns (avg forward return per window across ALL days),
        signal_lift (avg flagged return minus baseline, per window)
    Returns None if the stock has no data or no flagged events.
    """
    df = load_predictions(ticker)
    if df is None:
        return None

    events_df = compute_forward_returns(df, min_score=min_score)
    baseline  = compute_baseline_returns(df)

    if events_df.empty:
        return {
            "ticker": ticker,
            "total_days": len(df),
            "flagged_events": 0,
            "min_score_used": min_score,
            "message": f"No days with Suspicion_Score >= {min_score} found.",
            "events": [],
            "avg_returns": {},
            "baseline_returns": baseline,
            "signal_lift": {},
        }

    # Average return per window across all flagged events
    avg_returns = {}
    signal_lift = {}
    for window_name in WINDOWS:
        col = f"Return_{window_name.upper().replace('_', '')}_%"
        if col in events_df.columns:
            valid = events_df[col].dropna()
            avg_ret = round(valid.mean(), 2) if not valid.empty else None
            avg_returns[window_name] = avg_ret
            if avg_ret is not None and baseline.get(window_name) is not None:
                signal_lift[window_name] = round(avg_ret - baseline[window_name], 2)
            else:
                signal_lift[window_name] = None

    return {
        "ticker"          : ticker,
        "total_days"      : len(df),
        "flagged_events"  : len(events_df),
        "min_score_used"  : min_score,
        "events"          : events_df.to_dict(orient="records"),
        "avg_returns"     : avg_returns,
        "baseline_returns": baseline,
        "signal_lift"     : signal_lift,
    }


def run_backtest_all(min_score: float = DEFAULT_MIN_SCORE) -> list[dict]:
    """
    Runs backtest for every stock in data/processed/ and returns results
    sorted by the number of flagged events descending.
    """
    if not os.path.exists(PROCESSED_DIR):
        return []

    tickers = sorted([
        f.replace("_features.csv", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_features.csv")
    ])

    results = []
    for ticker in tickers:
        result = run_backtest(ticker, min_score=min_score)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["flagged_events"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FASTAPI ROUTER (imported by api.py)
# ══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/backtest", tags=["Backtest"])


@router.get("/")
def get_backtest_all(min_score: float = DEFAULT_MIN_SCORE):
    """
    GET /backtest?min_score=65

    Returns forward return analysis for ALL stocks, for every day
    the model flagged with Suspicion_Score >= min_score.

    Query params:
        min_score : float (default 65) — minimum score to count as an event
    """
    results = run_backtest_all(min_score=min_score)
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No processed stock data found. Run the full pipeline first."
        )
    return {
        "min_score_used": min_score,
        "stocks_analysed": len(results),
        "results": results,
    }


@router.get("/{ticker}")
def get_backtest_ticker(ticker: str, min_score: float = DEFAULT_MIN_SCORE):
    """
    GET /backtest/RELIANCE?min_score=65

    Returns forward return analysis for ONE stock — every flagged day's
    actual price movement at 1M, 3M, 6M, 1Y after the flag, plus the
    comparison against the stock's own baseline average return.

    Path params:
        ticker    : clean ticker name e.g. RELIANCE
    Query params:
        min_score : float (default 65) — minimum score to count as an event
    """
    result = run_backtest(ticker.upper(), min_score=min_score)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for '{ticker}'. Run the pipeline first."
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI REPORT (terminal output)
# ══════════════════════════════════════════════════════════════════════════════

def print_backtest_report(result: dict):
    """Prints a formatted backtest report for one stock to the terminal."""
    ticker = result["ticker"]

    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT — {ticker}")
    print(f"{'='*60}")
    print(f"  Total trading days  : {result['total_days']}")
    print(f"  Min score threshold : {result['min_score_used']}")
    print(f"  Flagged events found: {result['flagged_events']}")

    if result["flagged_events"] == 0:
        print(f"\n  {result.get('message', 'No flagged events.')}")
        return

    # Per-event table
    print(f"\n  FLAGGED EVENTS — Forward Returns:")
    print(f"  {'Date':<12} {'Score':>6} {'Price':>8} "
          f"{'1M%':>7} {'3M%':>7} {'6M%':>7} {'1Y%':>7} {'Dir':>5}")
    print(f"  {'-'*62}")

    for ev in result["events"]:
        r1m  = f"{ev.get('Return_1MONTH_%',  'N/A'):>7}" if ev.get("Return_1MONTH_%")  is not None else f"{'N/A':>7}"
        r3m  = f"{ev.get('Return_3MONTH_%',  'N/A'):>7}" if ev.get("Return_3MONTH_%")  is not None else f"{'N/A':>7}"
        r6m  = f"{ev.get('Return_6MONTH_%',  'N/A'):>7}" if ev.get("Return_6MONTH_%")  is not None else f"{'N/A':>7}"
        r1y  = f"{ev.get('Return_1YEAR_%',   'N/A'):>7}" if ev.get("Return_1YEAR_%")   is not None else f"{'N/A':>7}"
        direction = ev.get("Direction_1M", "N/A")

        try:
            r1m = f"{ev['Return_1MONTH_%']:>7.2f}"
        except (TypeError, KeyError):
            r1m = f"{'N/A':>7}"
        try:
            r3m = f"{ev['Return_3MONTH_%']:>7.2f}"
        except (TypeError, KeyError):
            r3m = f"{'N/A':>7}"
        try:
            r6m = f"{ev['Return_6MONTH_%']:>7.2f}"
        except (TypeError, KeyError):
            r6m = f"{'N/A':>7}"
        try:
            r1y = f"{ev['Return_1YEAR_%']:>7.2f}"
        except (TypeError, KeyError):
            r1y = f"{'N/A':>7}"

        print(f"  {ev['Date']:<12} "
              f"{ev['Suspicion_Score']:>6.1f} "
              f"{ev['Close_on_flag_day']:>8.2f} "
              f"{r1m} {r3m} {r6m} {r1y} "
              f"{direction:>5}")

    # Average returns across all events
    print(f"\n  AVERAGE FORWARD RETURNS (across {result['flagged_events']} event(s)):")
    print(f"  {'Window':<12} {'Avg Flagged%':>13} {'Avg Baseline%':>14} {'Lift%':>7}")
    print(f"  {'-'*48}")
    for window_name in WINDOWS:
        avg  = result["avg_returns"].get(window_name)
        base = result["baseline_returns"].get(window_name)
        lift = result["signal_lift"].get(window_name)

        avg_str  = f"{avg:>13.2f}" if avg  is not None else f"{'N/A':>13}"
        base_str = f"{base:>14.2f}" if base is not None else f"{'N/A':>14}"
        lift_str = f"{lift:>7.2f}"  if lift is not None else f"{'N/A':>7}"

        label = window_name.replace("_", " ").title()
        print(f"  {label:<12} {avg_str} {base_str} {lift_str}")

    print(f"\n  LIFT = (Avg return on flagged days) minus (Avg return on ALL days)")
    print(f"  Positive lift = flagged days outperformed the stock's own average.")
    print(f"  Negative lift = flagged days underperformed the stock's own average.")


def print_summary_table(all_results: list[dict]):
    """Prints the cross-stock summary table."""
    print(f"\n{'='*70}")
    print(f"  CROSS-STOCK BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Ticker':<14} {'Events':>7} "
          f"{'Avg 1M%':>9} {'Avg 3M%':>9} "
          f"{'Avg 6M%':>9} {'Avg 1Y%':>9} "
          f"{'Lift 1M':>8} {'Lift 1Y':>8}")
    print(f"  {'-'*74}")

    for r in all_results:
        avg  = r["avg_returns"]
        lift = r["signal_lift"]

        def fmt(val, width=9):
            return f"{val:{width}.2f}" if val is not None else f"{'N/A':>{width}}"

        print(f"  {r['ticker']:<14} "
              f"{r['flagged_events']:>7} "
              f"{fmt(avg.get('1_month'))}"
              f"{fmt(avg.get('3_month'))}"
              f"{fmt(avg.get('6_month'))}"
              f"{fmt(avg.get('1_year'))}"
              f"{fmt(lift.get('1_month'), 8)}"
              f"{fmt(lift.get('1_year'), 8)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
# INSTRUCTIONS TO ADD ENDPOINTS TO api.py:
#
#   1. Open backend/api.py
#   2. Add this import near the top (after existing imports):
#          from backend.backtest import router as backtest_router
#   3. Add this line after the CORS middleware setup:
#          app.include_router(backtest_router)
#   4. Restart the server:
#          uvicorn backend.api:app --reload --port 8000
#   5. New endpoints are live at:
#          GET http://localhost:8000/backtest
#          GET http://localhost:8000/backtest/{ticker}
#          GET http://localhost:8000/backtest/{ticker}?min_score=70
#          Interactive docs: http://localhost:8000/docs
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest: measure forward returns from model-flagged days."
    )
    parser.add_argument(
        "ticker", nargs="?", default=None,
        help="Optional: single ticker to backtest (e.g. RELIANCE). "
             "If omitted, runs for all stocks in data/processed/."
    )
    parser.add_argument(
        "--min-score", type=float, default=DEFAULT_MIN_SCORE,
        help=f"Minimum Suspicion_Score to count as a flagged event "
             f"(default: {DEFAULT_MIN_SCORE})."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Insider Trading Detector — Backtest")
    print("  Forward Return Analysis from Flagged Days")
    print("=" * 60)
    print(f"  Minimum score threshold : {args.min_score}")
    print(f"  Forward windows         : 1M / 3M / 6M / 1Y")

    if args.ticker:
        result = run_backtest(args.ticker.upper(), min_score=args.min_score)
        if result is None:
            print(f"\n  No data found for '{args.ticker}'. "
                  f"Run features.py + scoring.py first.")
        else:
            print_backtest_report(result)
    else:
        all_results = run_backtest_all(min_score=args.min_score)
        if not all_results:
            print(f"\n  No processed stock files found in {PROCESSED_DIR}/")
        else:
            for result in all_results:
                print_backtest_report(result)
            print_summary_table(all_results)

    print("\n  Done.")