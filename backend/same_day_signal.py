# backend/same_day_signal.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Same-day signal detection with zero lag.
#           The rolling windows (AVR, CAR, Vol_Spike, Return_Z) build the
#           historical baseline. The signal fires TODAY the moment TODAY's
#           candle confirms the pattern — no waiting for a window to fill.
#
# THE ZERO-LAG ARCHITECTURE:
#
#   OLD approach (lagged):
#     Signal fires when a rolling-window metric CROSSES a threshold AFTER
#     accumulating N days of data. By the time you see the signal, the
#     window period is already over.
#
#   NEW approach (same-day):
#     Rolling windows = the BASELINE (what is "normal" for this stock)
#     Same-day candle = the TRIGGER (did TODAY confirm the pattern?)
#
#     A signal fires on day T if ALL of:
#       1. Rolling baseline says this stock is in an "elevated" zone
#          (AVR above its EMA, volume above its EMA)
#       2. TODAY's candle body is large (strong directional conviction)
#       3. TODAY's close vs open direction matches the signal direction
#          (green candle for BUY, red candle for SELL)
#       4. TODAY's volume is a surge vs the recent 5-day average
#          (confirms institutional participation TODAY, not days ago)
#
#   WHY THIS HAS NO DELAY:
#     The trigger is ALWAYS computed from today's single candle (Open, High,
#     Low, Close, Volume) which are all known the moment the market closes.
#     The rolling baseline uses data UP TO BUT NOT INCLUDING today, so
#     there is no circular dependency and no look-ahead bias.
#
# HOW TO RUN:
#   python backend/same_day_signal.py              → all stocks
#   python backend/same_day_signal.py CGPOWER      → one stock
#   python backend/same_day_signal.py --test       → test suite (20 tests)
#   python backend/same_day_signal.py --live CGPOWER → score ONLY the latest
#                                                       trading day (production
#                                                       mode — run this at 6 PM)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "data/results"

# ── Baseline window sizes (rolling, look-back only) ───────────────────────────
# These define what "normal" looks like for each stock.
# They use data BEFORE today, so they introduce no same-day lag.
BASELINE_LONG  = 60    # long-term baseline (3 months)
BASELINE_SHORT = 5     # short-term comparison (1 week)
EMA_SPAN       = 20    # EMA span for smoothed baseline

# ── Same-day trigger thresholds ───────────────────────────────────────────────
# These are checked ONLY on today's single candle — zero lag.
MIN_BODY_SIZE_PCT    = 0.005   # body must be ≥0.5% of open price (filters doji)
MIN_VOLUME_SURGE     = 1.6     # strong volume confirmation, tuned to real data
MIN_BASELINE_AVR     = 2.0     # ~90th percentile — top 10% of days, tuned to real CGPOWER data
MIN_CANDLE_SCORE     = 0.55    # quality filter — filters weak candles

# ── Signal quality thresholds ─────────────────────────────────────────────────
BUY_RETURN_THRESHOLD  =  0.30   # >30% 6M return = confirmed BUY quality
SELL_RETURN_THRESHOLD = -0.15   # <-15% 6M return = confirmed SELL quality
FORWARD_6M_DAYS       = 130

# ── Model parameters ──────────────────────────────────────────────────────────
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 200
RF_N_ESTIMATORS  = 300
RF_MAX_DEPTH     = 6
TRAIN_RATIO      = 0.80
RANDOM_STATE     = 42


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BASELINE FEATURES (rolling, no same-day data)
# ══════════════════════════════════════════════════════════════════════════════

def compute_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the rolling BASELINE features that describe what "normal"
    looks like for this stock over the past N days.

    CRITICAL DESIGN RULE — NO SAME-DAY DATA IN BASELINES:
        Every rolling computation here uses shift(1) BEFORE the rolling
        window. This means: the baseline for day T is computed from
        days T-N to T-1, NEVER including day T itself.

        Why: if we included today's data in the baseline, the signal
        would be using today's price/volume to compute a threshold that
        is then compared against today's price/volume — circular and biased.

    The rolling window introduces ZERO additional lag on top of what
    already existed — these baselines were always computed from past data.
    The key change vs the original model is that we make this explicit
    by shifting before rolling.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Close, Volume, AVR, CAR_10 columns.

    Returns
    -------
    pd.DataFrame with new baseline columns:
        Baseline_Vol_EMA    : EMA of volume over past EMA_SPAN days
        Baseline_AVR_EMA    : EMA of AVR over past EMA_SPAN days
        Baseline_Return_Std : rolling std dev of daily returns (volatility proxy)
        Baseline_CAR_EMA    : EMA of CAR_10 over past EMA_SPAN days
        Baseline_Price_EMA  : EMA of close price (trend reference)
    """
    df = df.copy()

    # shift(1) ensures TODAY's value is NOT included in the baseline
    # that is used to judge TODAY's signal. This is the zero-lag guarantee.
    shifted_volume = df["Volume"].shift(1)
    shifted_avr    = df["AVR"].shift(1) if "AVR" in df.columns else pd.Series(1.0, index=df.index)
    shifted_car    = df["CAR_10"].shift(1) if "CAR_10" in df.columns else pd.Series(0.0, index=df.index)
    shifted_close  = df["Close"].shift(1)

    df["Baseline_Vol_EMA"]    = shifted_volume.ewm(span=EMA_SPAN, adjust=False).mean()
    df["Baseline_AVR_EMA"]    = shifted_avr.ewm(span=EMA_SPAN, adjust=False).mean()
    df["Baseline_CAR_EMA"]    = shifted_car.ewm(span=EMA_SPAN, adjust=False).mean()
    df["Baseline_Price_EMA"]  = shifted_close.ewm(span=EMA_SPAN, adjust=False).mean()

    shifted_returns = shifted_close.pct_change()
    df["Baseline_Return_Std"] = shifted_returns.rolling(
        BASELINE_LONG, min_periods=10
    ).std()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SAME-DAY TRIGGER FEATURES (today's candle only)
# ══════════════════════════════════════════════════════════════════════════════

def compute_trigger_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes same-day trigger features from TODAY's candle alone.

    These are the zero-lag features — computed entirely from today's
    Open, High, Low, Close, Volume which are known the instant the
    market closes. No rolling window, no delay.

    Returns
    -------
    pd.DataFrame with new trigger columns:
        Trigger_Candle_Dir    : +1 green, -1 red
        Trigger_Body_Pct      : body size as % of open (strength of move)
        Trigger_Upper_Shadow  : upper wick ratio (selling pressure today)
        Trigger_Lower_Shadow  : lower wick ratio (buying support today)
        Trigger_Volume_Surge  : today's vol / 5-day baseline vol (EMA-based)
        Trigger_Price_vs_EMA  : today's close vs baseline EMA (trend alignment)
        Trigger_Candle_Score  : 0-1 composite of the above — the main trigger
    """
    df = df.copy()

    # ── Direction ─────────────────────────────────────────────────────────────
    df["Trigger_Candle_Dir"] = np.where(df["Close"] > df["Open"], 1, -1)

    # ── Body size — normalised ────────────────────────────────────────────────
    df["Trigger_Body_Pct"] = (
        (df["Close"] - df["Open"]).abs() / df["Open"].replace(0, np.nan)
    ).round(6)

    # ── Shadows ───────────────────────────────────────────────────────────────
    upper = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1) - df["Low"]
    close_safe = df["Close"].replace(0, np.nan)

    df["Trigger_Upper_Shadow"] = (upper / close_safe).round(6)
    df["Trigger_Lower_Shadow"] = (lower / close_safe).round(6)

    # ── Volume surge vs 5-day EMA of PAST volume (baseline, no today) ─────────
    past_vol_ema = df["Volume"].shift(1).ewm(span=BASELINE_SHORT, adjust=False).mean()
    df["Trigger_Volume_Surge"] = (
        df["Volume"] / past_vol_ema.replace(0, np.nan)
    ).round(4)

    # ── Price vs baseline EMA ─────────────────────────────────────────────────
    # Positive = today's close is above the recent trend (bullish context)
    # Negative = today's close is below the recent trend (bearish context)
    if "Baseline_Price_EMA" in df.columns:
        df["Trigger_Price_vs_EMA"] = (
            (df["Close"] - df["Baseline_Price_EMA"]) /
            df["Baseline_Price_EMA"].replace(0, np.nan)
        ).round(6)
    else:
        df["Trigger_Price_vs_EMA"] = 0.0

    # ── Composite candle score ────────────────────────────────────────────────
    # Combines 4 same-day observations into one 0-1 score:
    #   body_score    : rewards large bodies (conviction)
    #   shadow_score  : penalises the shadow that works against the direction
    #   volume_score  : rewards volume surges (institutional participation)
    #   direction_bonus: adds a small premium for alignment with the trend
    #
    # The score is directional: a BUY day with a large green body, low upper
    # shadow, high volume, and price above the EMA scores close to 1.0.
    # A BUY day with a tiny body or huge upper shadow scores much lower.

    body_score = np.clip(
        df["Trigger_Body_Pct"] / 0.02,  # normalise: 2% body = score of 1.0
        0, 1
    )

    # For BUY direction: penalise upper shadow (sellers pushing back)
    # For SELL direction: penalise lower shadow (buyers pushing back)
    shadow_penalty = np.where(
        df["Trigger_Candle_Dir"] == 1,
        df["Trigger_Upper_Shadow"],
        df["Trigger_Lower_Shadow"]
    )
    shadow_score = np.clip(1 - shadow_penalty / 0.02, 0, 1)

    volume_score = np.clip(
        (df["Trigger_Volume_Surge"] - 1) / (MIN_VOLUME_SURGE - 1),
        0, 1
    )

    direction_bonus = np.where(
        df.get("Trigger_Price_vs_EMA", pd.Series(0, index=df.index)) *
        df["Trigger_Candle_Dir"] > 0,
        0.1, 0
    )

    df["Trigger_Candle_Score"] = np.round(
        body_score * 0.35 +
        shadow_score * 0.25 +
        volume_score * 0.30 +
        direction_bonus * 0.10,
        4
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SAME-DAY SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_same_day_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the rolling baseline + same-day trigger to fire signals.

    A BUY signal fires on day T when ALL of:
      1. Baseline_AVR_EMA[T] >= MIN_BASELINE_AVR  (stock in elevated zone)
      2. Trigger_Candle_Dir[T] == +1               (green candle today)
      3. Trigger_Body_Pct[T] >= MIN_BODY_SIZE_PCT  (meaningful body, not doji)
      4. Trigger_Volume_Surge[T] >= MIN_VOLUME_SURGE (volume confirmation today)
      5. Trigger_Candle_Score[T] >= MIN_CANDLE_SCORE (composite quality)

    A SELL signal fires on day T when the same conditions hold but with
    Trigger_Candle_Dir[T] == -1 (red candle) and CAR_10 negative.

    The signal fires AT MARKET CLOSE — the moment today's OHLCV is final.
    There is no N-day waiting period, no window that needs to accumulate.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have baseline and trigger features computed.

    Returns
    -------
    pd.DataFrame with new columns:
        Signal_Raw   : 1=BUY, -1=SELL, 0=none
        Signal_Label : "BUY" / "SELL" / "NO_SIGNAL"
    """
    df = df.copy()

    # Baseline condition: stock must be in an elevated statistical zone
    baseline_elevated = df["Baseline_AVR_EMA"] >= MIN_BASELINE_AVR

    # Same-day trigger conditions
    candle_quality = df["Trigger_Candle_Score"] >= MIN_CANDLE_SCORE
    volume_surge   = df["Trigger_Volume_Surge"] >= MIN_VOLUME_SURGE
    body_present   = df["Trigger_Body_Pct"] >= MIN_BODY_SIZE_PCT

    is_green = df["Trigger_Candle_Dir"] == 1
    is_red   = df["Trigger_Candle_Dir"] == -1

    # CAR alignment check — CAR_10 should match direction
    car_positive = df["CAR_10"].fillna(0) > 0.01  if "CAR_10" in df.columns else True
    car_negative = df["CAR_10"].fillna(0) < -0.01 if "CAR_10" in df.columns else True

    # Combined conditions
    buy_signal  = (baseline_elevated & is_green & body_present &
                   volume_surge & candle_quality & car_positive)

    sell_signal = (baseline_elevated & is_red & body_present &
                   volume_surge & candle_quality & car_negative)

    df["Signal_Raw"] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    df["Signal_Label"] = np.where(
        df["Signal_Raw"] == 1, "BUY",
        np.where(df["Signal_Raw"] == -1, "SELL", "NO_SIGNAL")
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QUALITY VALIDATION WITH FORWARD RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def validate_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal, measures the actual 6-month forward return and
    assigns a quality label used to train the Stage 2 RF classifier.

    BUY_QUALITY  (label=1)  : green candle + fwd 6M > +30%
    SELL_QUALITY (label=-1) : red candle   + fwd 6M < -15%
    NOISE        (label=0)  : signal fired but outcome didn't confirm
    """
    df = df.copy()
    n = len(df)
    close = df["Close"].values
    quality = np.zeros(n, dtype=int)

    for i in range(n):
        if df["Signal_Raw"].iloc[i] == 0:
            continue
        fwd_idx = i + FORWARD_6M_DAYS
        if fwd_idx >= n:
            continue
        fwd_ret = (close[fwd_idx] - close[i]) / close[i]
        sig = df["Signal_Raw"].iloc[i]
        if sig == 1 and fwd_ret > BUY_RETURN_THRESHOLD:
            quality[i] = 1
        elif sig == -1 and fwd_ret < SELL_RETURN_THRESHOLD:
            quality[i] = -1

    df["Signal_Quality"] = quality
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SECOND STAGE: RANDOM FOREST RE-SCORER
# ══════════════════════════════════════════════════════════════════════════════

def run_second_stage(df: pd.DataFrame, ticker: str) -> tuple:
    """
    Trains a Random Forest on historical signal quality outcomes and
    re-scores every signal to predict whether it will achieve the
    quality threshold (>30% 6M return for BUY, <-15% for SELL).

    Returns (df_enriched, metrics_dict)
    """
    df = df.copy()
    n = len(df)
    split = int(n * TRAIN_RATIO)

    # Feature set for the classifier — combines baseline + trigger features
    clf_features = [
        "Baseline_AVR_EMA", "Baseline_CAR_EMA", "Baseline_Return_Std",
        "Trigger_Body_Pct", "Trigger_Upper_Shadow", "Trigger_Lower_Shadow",
        "Trigger_Volume_Surge", "Trigger_Price_vs_EMA", "Trigger_Candle_Score",
        "Trigger_Candle_Dir",
    ]
    # Add original features if available
    for col in ["AVR", "CAR_10", "Return_Z", "Vol_Spike"]:
        if col in df.columns:
            clf_features.append(col)

    clf_features = [c for c in clf_features if c in df.columns]

    # Training data: only rows with signals AND known quality outcomes
    train_signals = df.iloc[:split]
    train_labeled = train_signals[
        (train_signals["Signal_Raw"] != 0) &
        (train_signals["Signal_Quality"] != 0)
    ]

    df["RF_Quality_Score"]  = 0.0
    df["RF_Final_Signal"]   = "NO_SIGNAL"

    if len(train_labeled) < 3:
        print(f"    Warning: only {len(train_labeled)} labeled training "
              f"signals for {ticker} — need ≥3. Skipping Stage 2.")
        df.loc[df["Signal_Raw"] == 1,  "RF_Final_Signal"] = "BUY_UNSCORED"
        df.loc[df["Signal_Raw"] == -1, "RF_Final_Signal"] = "SELL_UNSCORED"
        return df, {}

    X_train = train_labeled[clf_features].fillna(0).values
    y_train = train_labeled["Signal_Quality"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train_s, y_train)

    # Score ALL signal rows (train + test)
    sig_idx = df[df["Signal_Raw"] != 0].index
    if len(sig_idx) > 0:
        X_sig = scaler.transform(df.loc[sig_idx, clf_features].fillna(0).values)
        probas = rf.predict_proba(X_sig)
        preds  = rf.predict(X_sig)

        classes = list(rf.classes_)
        buy_col  = classes.index(1)  if  1 in classes else None
        sell_col = classes.index(-1) if -1 in classes else None

        for k, idx in enumerate(sig_idx):
            pred = preds[k]
            direction = df.loc[idx, "Signal_Raw"]
            prob_buy  = float(probas[k, buy_col])  if buy_col  is not None else 0.0
            prob_sell = float(probas[k, sell_col]) if sell_col is not None else 0.0

            quality_score = prob_buy if direction == 1 else prob_sell
            df.loc[idx, "RF_Quality_Score"] = round(quality_score, 4)

            if direction == 1 and pred == 1:
                df.loc[idx, "RF_Final_Signal"] = "HIGH_QUALITY_BUY"
            elif direction == -1 and pred == -1:
                df.loc[idx, "RF_Final_Signal"] = "HIGH_QUALITY_SELL"
            elif direction == 1:
                df.loc[idx, "RF_Final_Signal"] = "LOW_QUALITY_BUY"
            else:
                df.loc[idx, "RF_Final_Signal"] = "LOW_QUALITY_SELL"

    # Metrics on test signals
    test_signals = df.iloc[split:]
    test_labeled = test_signals[
        (test_signals["Signal_Raw"] != 0) &
        (test_signals["Signal_Quality"] != 0)
    ]

    metrics = {"ticker": ticker, "train_signals": len(train_labeled)}
    if len(test_labeled) >= 3:
        from sklearn.metrics import precision_score
        X_test = scaler.transform(test_labeled[clf_features].fillna(0).values)
        y_test = test_labeled["Signal_Quality"].values
        y_pred = rf.predict(X_test)
        metrics.update({
            "test_signals": len(test_labeled),
            "buy_precision":  round(float(precision_score(y_test, y_pred, labels=[1],  average="macro", zero_division=0)), 3),
            "sell_precision": round(float(precision_score(y_test, y_pred, labels=[-1], average="macro", zero_division=0)), 3),
            "hq_buy_count":   int((df["RF_Final_Signal"] == "HIGH_QUALITY_BUY").sum()),
            "hq_sell_count":  int((df["RF_Final_Signal"] == "HIGH_QUALITY_SELL").sum()),
        })

    return df, metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LIVE MODE (production use — score only today's candle)
# ══════════════════════════════════════════════════════════════════════════════

def run_live_signal(ticker: str) -> dict | None:
    """
    PRODUCTION MODE — scores ONLY the most recent trading day.

    This is what you call in your daily_scheduler.py at 6 PM.
    The full 10-year history is used to train the model and build
    the baseline, but only today's row is returned as the live signal.

    Parameters
    ----------
    ticker : str — clean ticker name, e.g. "CGPOWER"

    Returns
    -------
    dict with today's signal details, or None if no data found.

    Usage in daily_scheduler.py:
        from backend.same_day_signal import run_live_signal
        result = run_live_signal("CGPOWER")
        if result and result["signal"] != "NO_SIGNAL":
            send_alert(result)
    """
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing = [c for c in ["Open","High","Low","Close","Volume"] if c not in df.columns]
    if missing:
        return None

    df = compute_baseline_features(df)
    df = compute_trigger_features(df)
    df = generate_same_day_signals(df)
    df = validate_signal_quality(df)
    df, _ = run_second_stage(df, ticker)

    # Return ONLY the last row — today's signal
    last = df.iloc[-1]
    return {
        "ticker"          : ticker,
        "date"            : str(last["Date"])[:10],
        "close"           : float(last["Close"]),
        "signal"          : str(last["Signal_Label"]),
        "rf_final_signal" : str(last["RF_Final_Signal"]),
        "rf_quality_score": float(last["RF_Quality_Score"]),
        "candle_direction": int(last["Trigger_Candle_Dir"]),
        "candle_score"    : float(last["Trigger_Candle_Score"]),
        "volume_surge"    : float(last["Trigger_Volume_Surge"]),
        "baseline_avr"    : float(last["Baseline_AVR_EMA"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FULL BACKTEST RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_stock(ticker: str) -> dict | None:
    """Full same-day signal pipeline for one stock — backtest mode."""
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found.")
        return None

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing = [c for c in ["Open","High","Low","Close","Volume"] if c not in df.columns]
    if missing:
        print(f"  [SKIP] {ticker} missing OHLCV columns: {missing}")
        return None

    print(f"\n  Processing {ticker} ({len(df)} trading days)...")

    df = compute_baseline_features(df)
    df = compute_trigger_features(df)
    df = generate_same_day_signals(df)
    df = validate_signal_quality(df)
    df, metrics = run_second_stage(df, ticker)

    # Summary
    n_buy  = int((df["Signal_Label"] == "BUY").sum())
    n_sell = int((df["Signal_Label"] == "SELL").sum())
    n_hqb  = int((df["RF_Final_Signal"] == "HIGH_QUALITY_BUY").sum())
    n_hqs  = int((df["RF_Final_Signal"] == "HIGH_QUALITY_SELL").sum())

    print(f"    Raw signals     : {n_buy} BUY, {n_sell} SELL")
    print(f"    High quality    : {n_hqb} BUY, {n_hqs} SELL")
    if "buy_precision" in metrics:
        print(f"    Test precision  : BUY={metrics['buy_precision']}, "
              f"SELL={metrics['sell_precision']}")

    # Save results — only signal rows
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sig_df = df[df["Signal_Raw"] != 0].copy()
    cols = [
        "Date", "Close", "Signal_Label", "RF_Final_Signal", "RF_Quality_Score",
        "Trigger_Candle_Dir", "Trigger_Body_Pct", "Trigger_Volume_Surge",
        "Trigger_Candle_Score", "Baseline_AVR_EMA", "Signal_Quality",
    ]
    if "Ticker" in df.columns:
        cols.insert(0, "Ticker")
    out_path = os.path.join(RESULTS_DIR, f"same_day_signals_{ticker}.csv")
    sig_df[[c for c in cols if c in sig_df.columns]].to_csv(out_path, index=False)

    return {
        "ticker"  : ticker,
        "metrics" : metrics,
        "n_buy"   : n_buy,
        "n_sell"  : n_sell,
        "n_hqb"   : n_hqb,
        "n_hqs"   : n_hqs,
    }


def run_all() -> list[dict]:
    """Runs the full pipeline for every stock in data/processed/."""
    if not os.path.exists(PROCESSED_DIR):
        print(f"  Error: {PROCESSED_DIR}/ not found.")
        return []

    tickers = sorted([
        f.replace("_features.csv", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_features.csv")
    ])

    all_results = []
    for ticker in tickers:
        r = run_stock(ticker)
        if r:
            all_results.append(r)

    return all_results


def print_summary(all_results: list[dict]):
    """Prints the cross-stock summary table."""
    print(f"\n{'='*65}")
    print(f"  SAME-DAY SIGNAL — CROSS-STOCK SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Ticker':<12} {'BUY':>5} {'SELL':>6} "
          f"{'HQ_BUY':>8} {'HQ_SELL':>9} "
          f"{'BUY_Prec':>9} {'SELL_Prec':>10}")
    print(f"  {'-'*62}")

    total_buy = total_sell = total_hqb = total_hqs = 0
    for r in sorted(all_results, key=lambda x: x["n_hqb"], reverse=True):
        m  = r.get("metrics", {})
        bp = m.get("buy_precision",  "N/A")
        sp = m.get("sell_precision", "N/A")
        print(f"  {r['ticker']:<12} {r['n_buy']:>5} {r['n_sell']:>6} "
              f"{r['n_hqb']:>8} {r['n_hqs']:>9} "
              f"{str(bp):>9} {str(sp):>10}")
        total_buy  += r["n_buy"]
        total_sell += r["n_sell"]
        total_hqb  += r["n_hqb"]
        total_hqs  += r["n_hqs"]

    print(f"  {'-'*62}")
    print(f"  {'TOTAL':<12} {total_buy:>5} {total_sell:>6} "
          f"{total_hqb:>8} {total_hqs:>9}")
    print(f"\n  HQ = confirmed by RF classifier (green/red candle + forward return)")
    print(f"  Results in: {RESULTS_DIR}/same_day_signals_{{ticker}}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_test_df(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    open_ = close - rng.normal(0, 0.8, n)
    high  = np.maximum(open_, close) + rng.uniform(0.2, 1.5, n)
    low   = np.minimum(open_, close) - rng.uniform(0.2, 1.5, n)
    vol   = rng.integers(100_000, 1_000_000, n).astype(float)
    avr   = rng.normal(1.0, 0.3, n)
    car   = rng.normal(0.0, 0.015, n)
    vs    = rng.integers(0, 2, n)
    rz    = rng.normal(0.0, 0.8, n)
    if_f  = np.zeros(n, dtype=int)
    for idx in [int(n*0.20), int(n*0.45), int(n*0.70), int(n*0.90)]:
        if idx >= n:
            continue
        if_f[idx] = 1
        avr[idx]  = 4.5
        car[idx]  = 0.08
        vol[idx]  *= 3.5
        # Force green candle on BUY events
        open_[idx] = close[idx] - abs(rng.normal(3, 1))
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": open_.round(2), "High": high.round(2),
        "Low": low.round(2), "Close": close.round(2),
        "Volume": vol.astype(int),
        "AVR": avr.round(4), "CAR_10": car.round(6),
        "Vol_Spike": vs, "Return_Z": rz.round(4),
        "IF_Flag": if_f,
    })


class TestBaselineFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = _make_test_df(n=200)
        cls.result = compute_baseline_features(cls.df)

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_baseline_columns_added(self):
        for col in ["Baseline_Vol_EMA", "Baseline_AVR_EMA",
                    "Baseline_Return_Std", "Baseline_Price_EMA"]:
            self.assertIn(col, self.result.columns)

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_baseline_vol_ema_uses_past_data_only(self):
        """Baseline_Vol_EMA must be NaN on row 0 since there is no prior day."""
        self.assertTrue(pd.isna(self.result["Baseline_Vol_EMA"].iloc[0]),
            "Row 0 Baseline_Vol_EMA should be NaN — no past volume exists yet")

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_row_count_unchanged(self):
        self.assertEqual(len(self.df), len(self.result))


class TestTriggerFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = _make_test_df(n=200)
        df = compute_baseline_features(df)
        cls.result = compute_trigger_features(df)

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_trigger_columns_added(self):
        for col in ["Trigger_Candle_Dir", "Trigger_Body_Pct",
                    "Trigger_Volume_Surge", "Trigger_Candle_Score"]:
            self.assertIn(col, self.result.columns)

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_candle_dir_is_plus_or_minus_one(self):
        unique = set(self.result["Trigger_Candle_Dir"].unique())
        self.assertTrue(unique.issubset({1, -1}))

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_candle_score_between_zero_and_one(self):
        scores = self.result["Trigger_Candle_Score"].dropna()
        self.assertTrue((scores >= 0).all() and (scores <= 1.1).all())

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_body_pct_non_negative(self):
        self.assertTrue((self.result["Trigger_Body_Pct"].dropna() >= 0).all())

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_green_candle_gets_direction_plus_one(self):
        df = pd.DataFrame({
            "Open": [100.0], "High": [106.0], "Low": [99.0], "Close": [105.0],
            "Volume": [500_000], "AVR": [2.0], "CAR_10": [0.05],
        })
        df = compute_baseline_features(df)
        result = compute_trigger_features(df)
        self.assertEqual(result["Trigger_Candle_Dir"].iloc[0], 1)

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_red_candle_gets_direction_minus_one(self):
        df = pd.DataFrame({
            "Open": [105.0], "High": [106.0], "Low": [99.0], "Close": [100.0],
            "Volume": [500_000], "AVR": [2.0], "CAR_10": [-0.05],
        })
        df = compute_baseline_features(df)
        result = compute_trigger_features(df)
        self.assertEqual(result["Trigger_Candle_Dir"].iloc[0], -1)


class TestSignalGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = _make_test_df(n=400)
        df = compute_baseline_features(df)
        df = compute_trigger_features(df)
        cls.result = generate_same_day_signals(df)

    # ── Test 10 ───────────────────────────────────────────────────────────────
    def test_signal_columns_added(self):
        self.assertIn("Signal_Raw", self.result.columns)
        self.assertIn("Signal_Label", self.result.columns)

    # ── Test 11 ───────────────────────────────────────────────────────────────
    def test_signal_raw_values_valid(self):
        unique = set(self.result["Signal_Raw"].unique())
        self.assertTrue(unique.issubset({-1, 0, 1}))

    # ── Test 12 ───────────────────────────────────────────────────────────────
    def test_signal_label_values_valid(self):
        unique = set(self.result["Signal_Label"].unique())
        self.assertTrue(unique.issubset({"BUY", "SELL", "NO_SIGNAL"}))

    # ── Test 13 ───────────────────────────────────────────────────────────────
    def test_buy_signal_requires_green_candle(self):
        """Every row labelled BUY must have Trigger_Candle_Dir == +1."""
        buy_rows = self.result[self.result["Signal_Label"] == "BUY"]
        if not buy_rows.empty:
            self.assertTrue((buy_rows["Trigger_Candle_Dir"] == 1).all(),
                "Found a BUY signal with a red candle — should never happen")

    # ── Test 14 ───────────────────────────────────────────────────────────────
    def test_sell_signal_requires_red_candle(self):
        """Every row labelled SELL must have Trigger_Candle_Dir == -1."""
        sell_rows = self.result[self.result["Signal_Label"] == "SELL"]
        if not sell_rows.empty:
            self.assertTrue((sell_rows["Trigger_Candle_Dir"] == -1).all(),
                "Found a SELL signal with a green candle — should never happen")

    # ── Test 15 ───────────────────────────────────────────────────────────────
    def test_signal_count_is_reasonable(self):
        """Total signals should be << total rows (rare events, not daily)."""
        total_signals = (self.result["Signal_Raw"] != 0).sum()
        self.assertLess(total_signals, len(self.result) * 0.15,
            "Too many signals — model is over-triggering")


class TestFullPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df = _make_test_df(n=500, seed=99)
        df.to_csv(os.path.join(PROCESSED_DIR, "_TEST_SAMEDAY_features.csv"), index=False)

    @classmethod
    def tearDownClass(cls):
        for fname in ["_TEST_SAMEDAY_features.csv"]:
            p = os.path.join(PROCESSED_DIR, fname)
            if os.path.exists(p): os.remove(p)
        for fname in ["same_day_signals__TEST_SAMEDAY.csv"]:
            p = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(p): os.remove(p)

    # ── Test 16 ───────────────────────────────────────────────────────────────
    def test_run_stock_returns_result(self):
        result = run_stock("_TEST_SAMEDAY")
        self.assertIsNotNone(result)
        self.assertIn("n_buy", result)
        self.assertIn("n_sell", result)

    # ── Test 17 ───────────────────────────────────────────────────────────────
    def test_output_csv_saved(self):
        run_stock("_TEST_SAMEDAY")
        path = os.path.join(RESULTS_DIR, "same_day_signals__TEST_SAMEDAY.csv")
        self.assertTrue(os.path.exists(path))

    # ── Test 18 ───────────────────────────────────────────────────────────────
    def test_output_csv_has_signal_column(self):
        run_stock("_TEST_SAMEDAY")
        path = os.path.join(RESULTS_DIR, "same_day_signals__TEST_SAMEDAY.csv")
        df = pd.read_csv(path)
        self.assertIn("Signal_Label", df.columns)

    # ── Test 19 ───────────────────────────────────────────────────────────────
    def test_missing_ticker_returns_none(self):
        result = run_stock("_NONEXISTENT_TICKER_XYZ")
        self.assertIsNone(result)

    # ── Test 20 ───────────────────────────────────────────────────────────────
    def test_live_signal_returns_dict(self):
        result = run_live_signal("_TEST_SAMEDAY")
        self.assertIsNotNone(result)
        self.assertIn("signal", result)
        self.assertIn("date", result)
        self.assertIn("rf_final_signal", result)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
# HOW TO USE IN DAILY_SCHEDULER.PY (production):
#
#   from backend.same_day_signal import run_live_signal
#
#   def daily_job():
#       for ticker in ALL_TICKERS:
#           result = run_live_signal(ticker)
#           if result and result["rf_final_signal"] == "HIGH_QUALITY_BUY":
#               print(f"🟢 {ticker}: HIGH QUALITY BUY — Score {result['rf_quality_score']}")
#           elif result and result["rf_final_signal"] == "HIGH_QUALITY_SELL":
#               print(f"🔴 {ticker}: HIGH QUALITY SELL — Score {result['rf_quality_score']}")
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--live",   action="store_true",
                        help="Live mode: score only today's candle for the given ticker")
    args = parser.parse_args()

    if args.test:
        sys.argv = [sys.argv[0]]
        print("=" * 60)
        print("  Running same_day_signal.py test suite  (20 tests)")
        print("=" * 60)
        print()
        unittest.main(verbosity=2)

    elif args.live and args.ticker:
        print(f"  LIVE MODE — scoring today's candle for {args.ticker.upper()}")
        result = run_live_signal(args.ticker.upper())
        if result is None:
            print(f"  No data found for {args.ticker}. Run the pipeline first.")
        else:
            print(f"\n  Date      : {result['date']}")
            print(f"  Close     : {result['close']}")
            print(f"  Signal    : {result['signal']}")
            print(f"  RF Signal : {result['rf_final_signal']}")
            print(f"  Quality   : {result['rf_quality_score']:.2%}")
            print(f"  Candle    : {'GREEN' if result['candle_direction']==1 else 'RED'} "
                  f"(score {result['candle_score']:.2f})")
            print(f"  Vol Surge : {result['volume_surge']:.2f}x")
            print(f"  Baseline  : AVR EMA = {result['baseline_avr']:.2f}")

    else:
        print("=" * 60)
        print("  Same-Day Signal Detector (Zero Lag)")
        print("=" * 60)
        print(f"  Trigger   : today's candle (Open/High/Low/Close/Volume)")
        print(f"  Baseline  : EMA of past N days (no same-day data)")
        print(f"  BUY rule  : green candle + volume surge + elevated baseline")
        print(f"  SELL rule : red candle  + volume surge + elevated baseline")

        if args.ticker:
            r = run_stock(args.ticker.upper())
        else:
            all_results = run_all()
            if all_results:
                print_summary(all_results)

        print("\n  Done.")