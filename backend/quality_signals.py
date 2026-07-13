# backend/quality_signals.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Restructured insider trading detector that produces FEW, HIGH-
#           QUALITY signals instead of continuous daily flags.
#
# THE CORE PROBLEM WITH THE OLD APPROACH:
#   Isolation Forest on 2500 daily rows flags ~5% = 125 events per stock.
#   That is not insider trading — that is statistical noise. Insider trading
#   is a specific EVENT-DRIVEN PATTERN:
#     - Multiple signals converge TOGETHER in a SHORT window (5-15 days)
#     - The stock then moves SIGNIFICANTLY (>8%) in the next 3-6 months
#     - The pattern is RARE — a stock may have 3-8 genuine episodes in 10 years
#
# THE NEW APPROACH — THREE STAGES:
#
#   STAGE 1: STOCK SUITABILITY SCORING
#     Not every stock is equally suited for insider trading detection.
#     Mid-cap upper / small-cap lower boundary stocks are the sweet spot:
#       - Liquid enough to execute large positions without moving the price
#       - Small enough that insider information creates detectable price drift
#       - Not so small that random noise dominates the signal
#     We score each stock on a suitability index and warn if it scores low.
#
#   STAGE 2: EVENT WINDOW DETECTION
#     Instead of scoring every single day, we:
#     1. Find "candidate windows" — contiguous stretches where ≥2 signals
#        fire on the SAME or adjacent days (multi-signal convergence)
#     2. Score each WINDOW as a unit using peak signal intensity
#     3. Apply a strict combined threshold to filter out weak windows
#     This produces 3-12 candidate events per stock over 10 years.
#
#   STAGE 3: PATTERN VALIDATION WITH FORWARD RETURNS
#     For each candidate event window, measure the actual forward return
#     at 3 months and 6 months. Flag only windows followed by:
#       - >8% positive return (insider buying before good news), OR
#       - <-8% negative return (insider selling before bad news)
#     This is the decisive quality filter — it removes statistical artifacts
#     that were not followed by real price moves, leaving only windows where
#     the signal was followed by the kind of move that real insider activity
#     would cause.
#
# HOW TO RUN:
#   python backend/quality_signals.py              → all stocks
#   python backend/quality_signals.py CGPOWER      → one stock
#   python backend/quality_signals.py --suitability → just show stock rankings
#
# OUTPUT FILES:
#   data/results/quality_signals_{ticker}.csv  → confirmed events per stock
#   data/results/stock_suitability.csv         → stock ranking by model fit
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "data/results"

# ── Signal thresholds ─────────────────────────────────────────────────────────
AVR_THRESHOLD      = 2.5    # volume ratio above this = elevated
CAR_THRESHOLD      = 0.06   # 6% cumulative abnormal return = elevated
VOL_SPIKE_THRESHOLD = 1     # binary — already 0/1 from features.py
RETURN_Z_THRESHOLD  = 2.0   # z-score above this = elevated

# ── Event window parameters ───────────────────────────────────────────────────
# A "candidate window" is a stretch of consecutive days where ≥ MIN_SIGNALS_IN_WINDOW
# different signals are elevated AT THE SAME TIME or within WINDOW_GAP days of each other.
MIN_SIGNALS_IN_WINDOW = 2   # at least 2 of the 4 signals must fire together
WINDOW_GAP            = 3   # days gap allowed between signal fires within one window
MIN_WINDOW_SCORE      = 45  # minimum combined window score to be a candidate

# ── Forward return validation ─────────────────────────────────────────────────
# Only windows followed by a move THIS LARGE in EITHER direction are confirmed.
# This is the most important parameter — it separates real events from noise.
MIN_FORWARD_RETURN_PCT = 8.0   # must move ≥8% in 3M or 6M to be confirmed
FORWARD_3M_DAYS        = 66    # ~3 trading months
FORWARD_6M_DAYS        = 130   # ~6 trading months

# ── Suitability parameters ────────────────────────────────────────────────────
# Mid-cap lower / small-cap upper boundary is the sweet spot for detection.
# In Indian markets per SEBI: mid-cap = rank 101-250, small-cap = rank 251+
# Market cap proxy: we use average daily volume × average close price as a
# liquidity proxy since we don't have market cap data directly in our CSVs.
MIN_AVG_VOLUME  = 50_000    # below this = too illiquid, random noise dominates
MAX_AVG_VOLUME  = 5_000_000 # above this = too liquid, insiders can't move the price
MIN_VOLATILITY  = 0.01      # annualised daily std dev — below this = too stable
MAX_VOLATILITY  = 0.08      # above this = too chaotic for reliable signal detection

# ── Signal column names (must match features.py output exactly) ───────────────
COL_AVR       = "AVR"
COL_CAR       = "CAR_10"
COL_VOL_SPIKE = "Vol_Spike"
COL_RETURN_Z  = "Return_Z"
COL_IF_FLAG   = "IF_Flag"
COL_CLOSE     = "Close"
COL_DATE      = "Date"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STOCK SUITABILITY SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_stock_suitability(df: pd.DataFrame, ticker: str) -> dict:
    """
    Scores how well-suited a stock is for insider trading pattern detection.

    A stock is ideal when:
    - It has enough liquidity that real insider trades leave a volume footprint
      (too illiquid = random noise; too liquid = insiders can hide)
    - Its volatility is in a moderate range
      (too stable = signals never fire; too volatile = signals fire constantly)
    - Its price history is long enough for meaningful pattern detection

    Returns
    -------
    dict with keys: ticker, suitability_score (0-100), grade, verdict, details
    """
    n = len(df)

    if COL_CLOSE not in df.columns:
        return {"ticker": ticker, "suitability_score": 0,
                "grade": "F", "verdict": "Missing Close column"}

    close    = df[COL_CLOSE].dropna()
    avg_close = float(close.mean())

    # Use volume if available, otherwise estimate from AVR baseline
    if "Volume" in df.columns:
        avg_volume = float(df["Volume"].mean())
    else:
        avg_volume = 500_000  # assume moderate if not available

    # Daily return volatility (annualised)
    daily_returns = close.pct_change().dropna()
    daily_vol     = float(daily_returns.std())
    annual_vol    = daily_vol * np.sqrt(252)

    # Data length score — more history = better training
    years_of_data  = n / 252
    data_score     = min(100, years_of_data / 10 * 100)

    # Volume score — sweet spot between MIN and MAX
    if avg_volume < MIN_AVG_VOLUME:
        vol_score = max(0, avg_volume / MIN_AVG_VOLUME * 40)  # too illiquid
        vol_comment = "Too illiquid — random noise may dominate"
    elif avg_volume > MAX_AVG_VOLUME:
        vol_score = max(0, 100 - (avg_volume - MAX_AVG_VOLUME) / MAX_AVG_VOLUME * 30)
        vol_comment = "Very liquid — insiders can hide trades easily"
    else:
        # Ideal range: score peaks at ~500k-1M average volume
        vol_score = 100
        vol_comment = "Ideal liquidity range for detection"

    # Volatility score — sweet spot between MIN and MAX
    if annual_vol < MIN_VOLATILITY:
        ann_score = 30
        ann_comment = "Too stable — signals will rarely fire"
    elif annual_vol > MAX_VOLATILITY:
        ann_score = max(0, 100 - (annual_vol - MAX_VOLATILITY) / MAX_VOLATILITY * 50)
        ann_comment = "Too volatile — too much noise to isolate insider patterns"
    else:
        # Ideal: moderate volatility where abnormal moves stand out
        ann_score = 100
        ann_comment = "Ideal volatility range"

    # Signal noise ratio — how often AVR spikes vs. baseline (lower = cleaner)
    if COL_AVR in df.columns:
        avr_valid = df[COL_AVR].dropna()
        spike_rate = float((avr_valid > AVR_THRESHOLD).mean())
        # 2-5% spike rate is ideal — not too rare, not too frequent
        if spike_rate < 0.01:
            noise_score = 50
            noise_comment = "Very few AVR spikes — model may miss events"
        elif spike_rate > 0.15:
            noise_score = max(0, 100 - (spike_rate - 0.15) / 0.15 * 80)
            noise_comment = f"High spike rate ({spike_rate:.1%}) — noisy signal"
        else:
            noise_score = 100
            noise_comment = f"Clean signal rate ({spike_rate:.1%})"
    else:
        noise_score   = 50
        noise_comment = "AVR column missing"

    # Composite suitability score
    suitability = round(
        data_score   * 0.25 +
        vol_score    * 0.30 +
        ann_score    * 0.25 +
        noise_score  * 0.20,
        1
    )

    if suitability >= 80:
        grade   = "A"
        verdict = "Excellent fit — model should perform well"
    elif suitability >= 65:
        grade   = "B"
        verdict = "Good fit — expect reliable signals"
    elif suitability >= 50:
        grade   = "C"
        verdict = "Moderate fit — signals present but some noise expected"
    elif suitability >= 35:
        grade   = "D"
        verdict = "Poor fit — many false signals likely"
    else:
        grade   = "F"
        verdict = "Not suitable — consider removing from universe"

    return {
        "ticker"            : ticker,
        "suitability_score" : suitability,
        "grade"             : grade,
        "verdict"           : verdict,
        "years_of_data"     : round(years_of_data, 1),
        "avg_close"         : round(avg_close, 2),
        "annual_volatility" : round(annual_vol * 100, 2),
        "avg_volume"        : int(avg_volume),
        "avr_spike_rate_pct": round(spike_rate * 100, 2) if COL_AVR in df.columns else None,
        "data_score"        : round(data_score, 1),
        "volume_score"      : round(vol_score, 1),
        "volatility_score"  : round(ann_score, 1),
        "noise_score"       : round(noise_score, 1),
        "volume_comment"    : vol_comment,
        "volatility_comment": ann_comment,
        "noise_comment"     : noise_comment,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVENT WINDOW DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_event_windows(df: pd.DataFrame) -> list[dict]:
    """
    Finds candidate insider trading WINDOWS — stretches of days where
    multiple signals converge together, not just individual flagged days.

    Instead of asking "is today suspicious?" we ask "is this a coherent
    period of pre-event accumulation?" — a fundamentally different question
    that naturally produces far fewer, higher-quality candidates.

    Parameters
    ----------
    df : pd.DataFrame
        Fully featured stock history. Must contain Date, Close, and the
        signal columns (AVR, CAR_10, Vol_Spike, Return_Z).

    Returns
    -------
    list of dicts, one per candidate window, sorted by window score desc.
    Each dict contains the window's date range, peak signal values, and
    a combined window score.
    """
    df = df.copy().reset_index(drop=True)
    n  = len(df)

    # ── Step 1: Build a per-day signal count ──────────────────────────────────
    # For each day, count how many of the 4 signals are elevated
    signal_active = pd.DataFrame(index=df.index)

    if COL_AVR in df.columns:
        signal_active["avr"]  = (df[COL_AVR].fillna(0) > AVR_THRESHOLD).astype(int)
    else:
        signal_active["avr"] = 0

    if COL_CAR in df.columns:
        signal_active["car"]  = (df[COL_CAR].fillna(0).abs() > CAR_THRESHOLD).astype(int)
    else:
        signal_active["car"] = 0

    if COL_VOL_SPIKE in df.columns:
        signal_active["vol"]  = (df[COL_VOL_SPIKE].fillna(0) >= VOL_SPIKE_THRESHOLD).astype(int)
    else:
        signal_active["vol"] = 0

    if COL_RETURN_Z in df.columns:
        signal_active["rz"]   = (df[COL_RETURN_Z].fillna(0).abs() > RETURN_Z_THRESHOLD).astype(int)
    else:
        signal_active["rz"] = 0

    # Also include IF_Flag as a fifth signal if available
    if COL_IF_FLAG in df.columns:
        signal_active["if_flag"] = df[COL_IF_FLAG].fillna(0).astype(int)
    else:
        signal_active["if_flag"] = 0

    # Total signals firing per day
    signal_active["total"] = signal_active.sum(axis=1)

    # ── Step 2: Group consecutive active days into windows ────────────────────
    # A window starts when total >= MIN_SIGNALS_IN_WINDOW
    # and extends as long as we don't have more than WINDOW_GAP consecutive
    # quiet days (where total < MIN_SIGNALS_IN_WINDOW)
    windows    = []
    in_window  = False
    win_start  = 0
    gap_count  = 0

    for i in range(n):
        active = signal_active["total"].iloc[i] >= MIN_SIGNALS_IN_WINDOW

        if active:
            if not in_window:
                in_window = True
                win_start = i
            gap_count = 0
        else:
            if in_window:
                gap_count += 1
                if gap_count > WINDOW_GAP:
                    # Window closed
                    win_end = i - gap_count
                    if win_end > win_start:
                        windows.append((win_start, win_end))
                    in_window = False
                    gap_count = 0

    # Close any window still open at end of data
    if in_window:
        windows.append((win_start, n - 1))

    # ── Step 3: Score each window ─────────────────────────────────────────────
    scored_windows = []
    for win_start, win_end in windows:
        win_df = df.iloc[win_start: win_end + 1]
        sig_df = signal_active.iloc[win_start: win_end + 1]

        # Peak values of each signal within the window
        peak_avr    = float(win_df[COL_AVR].max())    if COL_AVR in win_df.columns else 0
        peak_car    = float(win_df[COL_CAR].abs().max()) if COL_CAR in win_df.columns else 0
        peak_rz     = float(win_df[COL_RETURN_Z].abs().max()) if COL_RETURN_Z in win_df.columns else 0
        peak_vol    = int(win_df[COL_VOL_SPIKE].max()) if COL_VOL_SPIKE in win_df.columns else 0
        max_signals = int(sig_df["total"].max())
        window_days = win_end - win_start + 1

        # Window score: combination of peak intensity + signal breadth + brevity
        # Shorter windows with more signals are more suspicious than long, scattered ones
        avr_component  = min(40, max(0, (peak_avr - AVR_THRESHOLD) / AVR_THRESHOLD * 20))
        car_component  = min(30, max(0, (peak_car - CAR_THRESHOLD) / CAR_THRESHOLD * 15))
        rz_component   = min(20, max(0, (peak_rz  - RETURN_Z_THRESHOLD) / RETURN_Z_THRESHOLD * 10))
        vol_component  = 10 if peak_vol >= VOL_SPIKE_THRESHOLD else 0

        # Bonus for multi-signal convergence
        breadth_bonus  = (max_signals - 1) * 5

        # Penalty for windows that are too long (genuine events are usually compact)
        length_penalty = max(0, (window_days - 15) * 0.5)

        window_score = round(
            avr_component + car_component + rz_component +
            vol_component + breadth_bonus - length_penalty,
            2
        )

        if window_score < MIN_WINDOW_SCORE:
            continue  # filter out weak windows

        start_date = df.iloc[win_start][COL_DATE]
        end_date   = df.iloc[win_end][COL_DATE]
        entry_price = float(df.iloc[win_start][COL_CLOSE])

        scored_windows.append({
            "window_start_idx"  : win_start,
            "window_end_idx"    : win_end,
            "window_start_date" : start_date if isinstance(start_date, str)
                                    else start_date.strftime("%Y-%m-%d"),
            "window_end_date"   : end_date if isinstance(end_date, str)
                                    else end_date.strftime("%Y-%m-%d"),
            "window_days"       : window_days,
            "peak_avr"          : round(peak_avr, 4),
            "peak_car"          : round(peak_car, 4),
            "peak_return_z"     : round(peak_rz, 4),
            "peak_vol_spike"    : peak_vol,
            "max_signals_same_day": max_signals,
            "window_score"      : window_score,
            "entry_price"       : entry_price,
        })

    scored_windows.sort(key=lambda w: w["window_score"], reverse=True)
    return scored_windows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FORWARD RETURN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_with_forward_returns(
    df: pd.DataFrame,
    windows: list[dict]
) -> list[dict]:
    """
    For each candidate window, measure the actual forward price return
    at 3M and 6M from the end of the window. Confirms only windows
    followed by a significant move (>= MIN_FORWARD_RETURN_PCT in either
    direction), discarding those that fizzled into nothing.

    WHY THIS IS THE DECISIVE QUALITY FILTER:
        A genuine insider-trading pattern should precede a real, large
        price move — because the insider was acting on information about
        an imminent event (earnings surprise, merger, regulatory approval).
        A signal window NOT followed by a significant move was probably
        just normal market noise that briefly triggered our thresholds.

    Parameters
    ----------
    df      : full stock history DataFrame
    windows : list of dicts from detect_event_windows()

    Returns
    -------
    list of confirmed windows, each enriched with forward return data.
    """
    close_arr = df[COL_CLOSE].values
    n         = len(df)
    confirmed = []

    for w in windows:
        end_idx    = w["window_end_idx"]
        exit_price = float(close_arr[end_idx])

        if exit_price == 0:
            continue

        # Forward returns from the LAST day of the window
        fwd_3m = None
        fwd_6m = None

        if end_idx + FORWARD_3M_DAYS < n:
            price_3m = float(close_arr[end_idx + FORWARD_3M_DAYS])
            fwd_3m   = round((price_3m - exit_price) / exit_price * 100, 2)

        if end_idx + FORWARD_6M_DAYS < n:
            price_6m = float(close_arr[end_idx + FORWARD_6M_DAYS])
            fwd_6m   = round((price_6m - exit_price) / exit_price * 100, 2)

        # THE KEY FILTER: was there a real move?
        # We check EITHER 3M or 6M — insider effects can take time to materialise
        max_abs_return = max(
            abs(fwd_3m) if fwd_3m is not None else 0,
            abs(fwd_6m) if fwd_6m is not None else 0,
        )

        is_confirmed = max_abs_return >= MIN_FORWARD_RETURN_PCT

        # Only include if confirmed, OR if we don't have enough future data yet
        # (window is too recent to have 6M of subsequent data — keep as "pending")
        is_pending = fwd_3m is None and fwd_6m is None

        if is_confirmed or is_pending:
            direction = "PENDING"
            if fwd_3m is not None:
                direction = "BUY SIGNAL" if fwd_3m > 0 else "SELL SIGNAL"
            elif fwd_6m is not None:
                direction = "BUY SIGNAL" if fwd_6m > 0 else "SELL SIGNAL"

            confirmed.append({
                **w,
                "exit_price"          : round(exit_price, 2),
                "fwd_return_3m_pct"   : fwd_3m,
                "fwd_return_6m_pct"   : fwd_6m,
                "max_abs_return_pct"  : round(max_abs_return, 2),
                "direction"           : direction,
                "confirmed"           : is_confirmed,
                "pending_validation"  : is_pending,
            })

    return confirmed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_quality_detection(ticker: str) -> dict | None:
    """
    Full quality signal detection pipeline for one stock.
    Runs all three stages: suitability → event windows → forward validation.
    """
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found. Run features.py first.")
        return None

    df = pd.read_csv(filepath, parse_dates=[COL_DATE])
    df = df.sort_values(COL_DATE).reset_index(drop=True)

    # Stage 1: Suitability
    suitability = score_stock_suitability(df, ticker)

    # Stage 2: Event window detection
    candidate_windows = detect_event_windows(df)

    # Stage 3: Forward return validation
    confirmed_events  = validate_with_forward_returns(df, candidate_windows)

    return {
        "ticker"           : ticker,
        "suitability"      : suitability,
        "candidate_windows": len(candidate_windows),
        "confirmed_events" : confirmed_events,
        "n_confirmed"      : len(confirmed_events),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PRINTING & SAVING
# ══════════════════════════════════════════════════════════════════════════════

def print_stock_report(result: dict):
    """Prints a clean per-stock quality signal report to the terminal."""
    t   = result["ticker"]
    s   = result["suitability"]
    evs = result["confirmed_events"]

    print(f"\n{'='*65}")
    print(f"  {t}  —  Suitability: {s['suitability_score']}/100 "
          f"[Grade {s['grade']}]  {s['verdict']}")
    print(f"  Data: {s['years_of_data']} years | "
          f"Vol: {s['annual_volatility']}%/yr | "
          f"AVR spike rate: {s.get('avr_spike_rate_pct', 'N/A')}%")
    print(f"  {s['volume_comment']} | {s['noise_comment']}")
    print(f"{'='*65}")

    print(f"  Candidate windows found : {result['candidate_windows']}")
    print(f"  Confirmed quality events: {result['n_confirmed']}")

    if not evs:
        print(f"  No confirmed events (either no pattern found, or "
              f"signals not followed by ≥{MIN_FORWARD_RETURN_PCT}% move)")
        return

    print(f"\n  {'Window':<13} {'Days':>5} {'Score':>6} "
          f"{'PeakAVR':>8} {'PeakCAR':>8} "
          f"{'3M%':>7} {'6M%':>7} {'Direction':<14} {'Status'}")
    print(f"  {'-'*85}")

    for ev in evs:
        r3m    = f"{ev['fwd_return_3m_pct']:>7.1f}" if ev["fwd_return_3m_pct"] is not None else f"{'N/A':>7}"
        r6m    = f"{ev['fwd_return_6m_pct']:>7.1f}" if ev["fwd_return_6m_pct"] is not None else f"{'N/A':>7}"
        status = "CONFIRMED" if ev["confirmed"] else "PENDING"

        print(f"  {ev['window_start_date']:<13} "
              f"{ev['window_days']:>5} "
              f"{ev['window_score']:>6.1f} "
              f"{ev['peak_avr']:>8.2f} "
              f"{ev['peak_car']:>8.4f} "
              f"{r3m} {r6m} "
              f"{ev['direction']:<14} "
              f"{status}")


def print_suitability_ranking(all_results: list[dict]):
    """Prints the cross-stock suitability ranking table."""
    print(f"\n{'='*75}")
    print(f"  STOCK SUITABILITY RANKING — Best fit for insider detection model")
    print(f"{'='*75}")
    print(f"  {'Ticker':<12} {'Score':>6} {'Grade':>6} "
          f"{'Years':>6} {'Vol%':>6} {'SpikeRate%':>11} "
          f"{'Events':>7}  Verdict")
    print(f"  {'-'*75}")

    sorted_results = sorted(
        all_results,
        key=lambda r: r["suitability"]["suitability_score"],
        reverse=True
    )

    for r in sorted_results:
        s  = r["suitability"]
        marker = "★ " if s["suitability_score"] >= 65 else "  "
        print(f"  {marker}{r['ticker']:<12} "
              f"{s['suitability_score']:>6.1f} "
              f"{s['grade']:>6} "
              f"{s['years_of_data']:>6.1f} "
              f"{s['annual_volatility']:>6.1f} "
              f"{str(s.get('avr_spike_rate_pct','N/A')):>11} "
              f"{r['n_confirmed']:>7}  "
              f"{s['verdict']}")

    good = [r for r in all_results if r["suitability"]["suitability_score"] >= 65]
    bad  = [r for r in all_results if r["suitability"]["suitability_score"] < 35]

    print(f"\n  ★ = Good fit (score ≥ 65): {len(good)} stocks")
    if bad:
        print(f"  ⚠ Consider removing (score < 35): "
              f"{[r['ticker'] for r in bad]}")


def save_results(all_results: list[dict]):
    """Saves confirmed events and suitability scores to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Per-stock confirmed events
    all_events = []
    for r in all_results:
        for ev in r["confirmed_events"]:
            all_events.append({"ticker": r["ticker"], **ev})

    if all_events:
        events_df = pd.DataFrame(all_events)
        events_df = events_df.drop(
            columns=["window_start_idx", "window_end_idx"], errors="ignore"
        )
        events_df.to_csv(
            os.path.join(RESULTS_DIR, "quality_signals_all.csv"),
            index=False
        )
        print(f"\n  All confirmed events → "
              f"{RESULTS_DIR}/quality_signals_all.csv")

    # Suitability scores
    suit_rows = [r["suitability"] for r in all_results]
    suit_df   = pd.DataFrame(suit_rows)
    suit_df   = suit_df.sort_values("suitability_score", ascending=False)
    suit_df.to_csv(
        os.path.join(RESULTS_DIR, "stock_suitability.csv"),
        index=False
    )
    print(f"  Suitability scores    → "
          f"{RESULTS_DIR}/stock_suitability.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality signal detector — few, high-confidence "
                    "insider trading patterns."
    )
    parser.add_argument(
        "ticker", nargs="?", default=None,
        help="Single ticker to analyse (e.g. CGPOWER). "
             "Omit to run all stocks."
    )
    parser.add_argument(
        "--suitability", action="store_true",
        help="Only print the stock suitability ranking, skip event detection."
    )
    parser.add_argument(
        "--min-score", type=float, default=MIN_WINDOW_SCORE,
        help=f"Minimum window score to be a candidate "
             f"(default {MIN_WINDOW_SCORE}). Raise to get fewer, "
             f"higher-confidence signals."
    )
    parser.add_argument(
        "--min-return", type=float, default=MIN_FORWARD_RETURN_PCT,
        help=f"Minimum forward return %% to confirm an event "
             f"(default {MIN_FORWARD_RETURN_PCT}). Raise to confirm "
             f"only the most dramatic moves."
    )
    args = parser.parse_args()

    # Override defaults from CLI
    MIN_WINDOW_SCORE      = args.min_score
    MIN_FORWARD_RETURN_PCT = args.min_return

    print("=" * 65)
    print("  Insider Trading Detector — Quality Signal Mode")
    print("=" * 65)
    print(f"  Min signals in window : {MIN_SIGNALS_IN_WINDOW}")
    print(f"  Min window score      : {MIN_WINDOW_SCORE}")
    print(f"  Min forward return    : {MIN_FORWARD_RETURN_PCT}%")
    print(f"  Forward windows       : 3M ({FORWARD_3M_DAYS}d) / "
          f"6M ({FORWARD_6M_DAYS}d)")
    print(f"  Confirmed = pattern + ≥{MIN_FORWARD_RETURN_PCT}% price move after")

    if not os.path.exists(PROCESSED_DIR):
        print(f"\n  Error: {PROCESSED_DIR}/ not found. Run features.py first.")
        sys.exit(1)

    if args.ticker:
        ticker_names = [args.ticker.upper()]
    else:
        ticker_names = sorted([
            f.replace("_features.csv", "")
            for f in os.listdir(PROCESSED_DIR)
            if f.endswith("_features.csv")
        ])

    all_results = []
    for ticker in ticker_names:
        result = run_quality_detection(ticker)
        if result:
            all_results.append(result)

    if args.suitability:
        print_suitability_ranking(all_results)
    else:
        for result in all_results:
            print_stock_report(result)
        if len(all_results) > 1:
            print_suitability_ranking(all_results)

    if all_results:
        save_results(all_results)

    print("\n  Done.")