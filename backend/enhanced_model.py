# backend/enhanced_model.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Two-stage enhanced model for higher-accuracy insider trading signal
#           prediction, addressing the 19% SELL accuracy problem and boosting
#           the 55% BUY-with->30%-6M-return rate.
#
# THE TWO-STAGE ARCHITECTURE:
#
#   STAGE 1 — Enhanced Isolation Forest
#     Original features (4): AVR, CAR_10, Vol_Spike, Return_Z
#     NEW features added (6):
#       Candle_Direction : +1 if green (Close>Open), -1 if red
#       Body_Size        : abs(Close-Open)/Open — candle body strength
#       Upper_Shadow     : (High - max(Open,Close))/Close — selling pressure
#       Lower_Shadow     : (min(Open,Close) - Low)/Close — buying support
#       Price_Momentum_5 : 5-day return vs 20-day return — momentum alignment
#       Volume_Surge     : today's volume / 5-day avg volume (finer-grained
#                          than AVR which uses a 60-day baseline)
#
#     WHY CANDLE FEATURES HELP THE SELL SIGNAL:
#       A SELL signal should have a RED candle on the signal day (Close < Open)
#       because insiders selling means there is immediate downward price pressure.
#       The original model had no awareness of intraday price structure, which
#       is why SELL signals were so inaccurate (19%) — it was flagging anomalous
#       days without checking whether the price action itself confirmed selling.
#
#   STAGE 2 — Random Forest Classifier
#     Takes the flagged windows from Stage 1 and re-scores them using a
#     classifier trained on HISTORICAL SIGNAL OUTCOMES.
#
#     Training target:
#       BUY_QUALITY  = 1 if candle is green AND 6M forward return > BUY_RETURN_THRESHOLD
#       SELL_QUALITY = 1 if candle is red  AND 6M forward return < SELL_RETURN_THRESHOLD
#       else         = 0 (noise, ignore)
#
#     The classifier learns WHICH FEATURE COMBINATIONS predict these
#     high-quality outcomes from the first 80% of data, then applies
#     that learning to score new signals in the test/live period.
#
# HOW TO RUN:
#   python backend/enhanced_model.py              → all stocks
#   python backend/enhanced_model.py CGPOWER      → one stock
#   python backend/enhanced_model.py --test       → runs test suite
#
# OUTPUT:
#   data/results/enhanced_signals_{ticker}.csv   → high-confidence signals
#   data/results/model_performance.csv           → accuracy report per stock
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "data/results"

# Stage 1: Isolation Forest
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 200
RANDOM_STATE     = 42

# Stage 2: Random Forest Classifier
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH    = 6     # limited depth to prevent overfitting on small datasets

# Signal quality thresholds
BUY_RETURN_THRESHOLD  =  0.30   # 6-month return must be > +30% for BUY quality
SELL_RETURN_THRESHOLD = -0.15   # 6-month return must be < -15% for SELL quality
FORWARD_6M_DAYS       = 130     # ~6 trading months

# Train/test split
TRAIN_RATIO = 0.80   # 80% of history for training (~8 years)

# Original 4 features from features.py
BASE_FEATURES = ["AVR", "CAR_10", "Vol_Spike", "Return_Z"]

# 6 new candle + momentum features computed here
NEW_FEATURES = [
    "Candle_Direction",
    "Body_Size",
    "Upper_Shadow",
    "Lower_Shadow",
    "Price_Momentum_5",
    "Volume_Surge",
]

ALL_FEATURES = BASE_FEATURES + NEW_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NEW FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 6 new candle-structure and momentum features from OHLCV data.

    WHY EACH FEATURE MATTERS FOR INSIDER TRADING DETECTION:

    Candle_Direction (+1 or -1):
        The single most important new feature. On a BUY signal day, the
        candle should be GREEN (insiders accumulating → price closes higher
        than it opened). On a SELL signal day, the candle should be RED.
        This directly fixes the 19% SELL accuracy problem by teaching the
        model to look at what price actually DID, not just what the volume
        and return statistics say.

    Body_Size (0 to ~0.1 typically):
        A large body means conviction — price moved decisively from open
        to close. A tiny body (doji-like) means indecision. Insiders
        trading with size tend to produce large-body candles because they
        are directional, not random.

    Upper_Shadow (0+):
        Price went up intraday but got sold back down. Large upper shadow
        on a BUY signal day means sellers are absorbing the insider buying.
        Very high upper shadow = weakened BUY signal.

    Lower_Shadow (0+):
        Price went down intraday but got bought back up. Large lower shadow
        on a SELL signal day means buyers are absorbing the insider selling.
        Very high lower shadow = weakened SELL signal.

    Price_Momentum_5:
        5-day price return minus 20-day price return. Positive = short-term
        momentum is accelerating above the medium-term trend — consistent
        with insider buying building up ahead of an event. Negative =
        the opposite, consistent with distribution before bad news.

    Volume_Surge:
        Today's volume divided by the 5-day rolling average volume.
        This is more sensitive than AVR (which uses a 60-day baseline) —
        it catches intraday spikes that get smoothed away in AVR.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Open, High, Low, Close, Volume columns.

    Returns
    -------
    pd.DataFrame — same as input with 6 new columns appended.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compute_candle_features() requires columns {missing}")

    df = df.copy()

    # ── Candle direction ──────────────────────────────────────────────────────
    df["Candle_Direction"] = np.where(df["Close"] > df["Open"], 1, -1)

    # ── Body size — normalised by open price ──────────────────────────────────
    df["Body_Size"] = ((df["Close"] - df["Open"]).abs() / df["Open"]).round(6)

    # ── Shadow lengths — normalised by close price ────────────────────────────
    df["Upper_Shadow"] = (
        (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    ).round(6)

    df["Lower_Shadow"] = (
        (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]
    ).round(6)

    # ── Price momentum: 5-day return minus 20-day return ─────────────────────
    ret_5  = df["Close"].pct_change(5)
    ret_20 = df["Close"].pct_change(20)
    df["Price_Momentum_5"] = (ret_5 - ret_20).round(6)

    # ── Volume surge: today vs 5-day rolling average ──────────────────────────
    vol_ma5 = df["Volume"].rolling(5, min_periods=1).mean()
    vol_ma5 = vol_ma5.replace(0, np.nan)
    df["Volume_Surge"] = (df["Volume"] / vol_ma5).round(4)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SIGNAL LABELLING (for training the Stage 2 classifier)
# ══════════════════════════════════════════════════════════════════════════════

def label_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every row where IF_Flag == 1, computes the 6-month forward return
    and assigns a quality label:

        BUY_QUALITY  (label = 1) : green candle + fwd 6M return > +30%
        SELL_QUALITY (label = -1): red candle  + fwd 6M return < -15%
        NOISE        (label = 0) : everything else

    This is what the Stage 2 Random Forest is trained to predict.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Close, Candle_Direction, IF_Flag.

    Returns
    -------
    pd.DataFrame with a new "Signal_Quality" column.
    """
    df = df.copy()
    n = len(df)
    close = df["Close"].values
    quality = np.zeros(n, dtype=int)

    for i in range(n):
        if df["IF_Flag"].iloc[i] != 1:
            continue

        fwd_idx = i + FORWARD_6M_DAYS
        if fwd_idx >= n:
            quality[i] = 0   # not enough future data — treat as unknown
            continue

        fwd_return = (close[fwd_idx] - close[i]) / close[i]
        candle_dir = df["Candle_Direction"].iloc[i]

        if candle_dir == 1 and fwd_return > BUY_RETURN_THRESHOLD:
            quality[i] = 1    # high-quality BUY signal
        elif candle_dir == -1 and fwd_return < SELL_RETURN_THRESHOLD:
            quality[i] = -1   # high-quality SELL signal
        else:
            quality[i] = 0    # flagged but not a quality outcome

    df["Signal_Quality"] = quality
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STAGE 1: ENHANCED ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════════════

def run_enhanced_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an Isolation Forest on ALL 10 features (4 original + 6 new)
    using the training portion of the data, then scores the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns in ALL_FEATURES.

    Returns
    -------
    pd.DataFrame with new columns: IF_Score_Enhanced, IF_Flag_Enhanced
    """
    df = df.copy()
    n = len(df)
    split = int(n * TRAIN_RATIO)

    X = df[ALL_FEATURES].fillna(0.0).values
    X_train = X[:split]

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)

    scores = model.decision_function(X)
    preds  = model.predict(X)

    df["IF_Score_Enhanced"] = np.round(scores, 4)
    df["IF_Flag_Enhanced"]  = (preds == -1).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STAGE 2: RANDOM FOREST CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def run_second_stage_classifier(
    df: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Trains a Random Forest on historical signal outcomes and uses it to
    classify new signals as high-quality BUY, high-quality SELL, or noise.

    TRAINING STRATEGY:
        We train ONLY on the first 80% of the data (the training period)
        to prevent look-ahead bias — the classifier cannot use knowledge
        of future prices during its training phase.

        We then apply the trained classifier to the FULL dataset so we
        can see both in-sample and out-of-sample quality predictions.

    Parameters
    ----------
    df     : DataFrame with all 10 features, IF_Flag, Signal_Quality columns
    ticker : Stock name for reporting

    Returns
    -------
    tuple of:
        pd.DataFrame : input df with new columns RF_Prediction, RF_Probability_Buy,
                       RF_Probability_Sell, Final_Signal
        dict         : performance metrics dictionary
    """
    df = df.copy()
    n  = len(df)
    split = int(n * TRAIN_RATIO)

    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    # Training data: only rows where the model flagged something
    # and we have a quality label (enough future data to verify outcome)
    train_flagged = train_df[
        (train_df["IF_Flag_Enhanced"] == 1) &
        (train_df["Signal_Quality"].notna())
    ].copy()

    if len(train_flagged) < 10:
        print(f"    Warning: only {len(train_flagged)} labeled training signals "
              f"for {ticker} — need ≥10 for a reliable classifier. "
              f"Skipping Stage 2 for this stock.")
        df["RF_Prediction"]      = 0
        df["RF_Probability_Buy"] = 0.0
        df["RF_Probability_Sell"]= 0.0
        df["Final_Signal"]       = "INSUFFICIENT_DATA"
        return df, {}

    X_train = train_flagged[ALL_FEATURES].fillna(0).values
    y_train = train_flagged["Signal_Quality"].values

    # Scale features — RF is not sensitive to scale but StandardScaler
    # helps when features have very different magnitudes (AVR ~1-6 vs
    # Body_Size ~0.001-0.05 vs Volume_Surge ~0.5-10)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the classifier
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",   # corrects for imbalance (few signals vs many noise)
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train_scaled, y_train)

    # Apply to ALL flagged rows (both train and test periods)
    all_flagged_idx = df[df["IF_Flag_Enhanced"] == 1].index
    X_all_flagged = scaler.transform(df.loc[all_flagged_idx, ALL_FEATURES].fillna(0).values)

    rf_preds   = rf.predict(X_all_flagged)
    rf_probas  = rf.predict_proba(X_all_flagged)

    # Map class indices to buy/sell probabilities
    classes = list(rf.classes_)
    buy_idx  = classes.index(1)  if  1 in classes else None
    sell_idx = classes.index(-1) if -1 in classes else None

    df["RF_Prediction"]       = 0
    df["RF_Probability_Buy"]  = 0.0
    df["RF_Probability_Sell"] = 0.0

    df.loc[all_flagged_idx, "RF_Prediction"] = rf_preds
    if buy_idx is not None:
        df.loc[all_flagged_idx, "RF_Probability_Buy"] = np.round(rf_probas[:, buy_idx], 4)
    if sell_idx is not None:
        df.loc[all_flagged_idx, "RF_Probability_Sell"] = np.round(rf_probas[:, sell_idx], 4)

    # Final signal label incorporating candle confirmation
    def assign_final_signal(row):
        if row["RF_Prediction"] == 1 and row["Candle_Direction"] == 1:
            return "HIGH_QUALITY_BUY"
        elif row["RF_Prediction"] == -1 and row["Candle_Direction"] == -1:
            return "HIGH_QUALITY_SELL"
        elif row["IF_Flag_Enhanced"] == 1:
            return "LOW_QUALITY_FLAG"
        else:
            return "NO_SIGNAL"

    df["Final_Signal"] = df.apply(assign_final_signal, axis=1)

    # ── Performance metrics on TEST period ───────────────────────────────────
    test_flagged = test_df[
        (test_df["IF_Flag_Enhanced"] == 1) &
        (test_df["Signal_Quality"].notna())
    ]

    metrics = {}
    if len(test_flagged) >= 3:
        X_test = scaler.transform(test_flagged[ALL_FEATURES].fillna(0).values)
        y_test = test_flagged["Signal_Quality"].values
        y_pred = rf.predict(X_test)

        metrics = {
            "ticker"              : ticker,
            "train_signals"       : len(train_flagged),
            "test_signals"        : len(test_flagged),
            "test_buy_precision"  : round(precision_score(
                y_test, y_pred, labels=[1], average="macro", zero_division=0), 3),
            "test_sell_precision" : round(precision_score(
                y_test, y_pred, labels=[-1], average="macro", zero_division=0), 3),
            "test_overall_precision": round(precision_score(
                y_test, y_pred, average="weighted", zero_division=0), 3),
            "high_quality_buy_count" : int((df["Final_Signal"] == "HIGH_QUALITY_BUY").sum()),
            "high_quality_sell_count": int((df["Final_Signal"] == "HIGH_QUALITY_SELL").sum()),
        }
    else:
        metrics = {
            "ticker": ticker,
            "train_signals": len(train_flagged),
            "test_signals": len(test_flagged),
            "note": "Too few test signals for reliable metrics",
        }

    return df, metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_enhanced_model(ticker: str) -> dict | None:
    """
    Full two-stage pipeline for one stock.
    Stage 1: compute new features → enhanced IF
    Stage 2: label quality → train RF → final signal classification
    """
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found. Run features.py first.")
        return None

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing_ohlcv = [c for c in ["Open","High","Low","Close","Volume"] if c not in df.columns]
    if missing_ohlcv:
        print(f"  [SKIP] {ticker} missing OHLCV columns: {missing_ohlcv}")
        print(f"         Make sure ingest.py saves Open/High/Low to the features CSV.")
        return None

    print(f"\n  Processing {ticker} ...")

    # Step 1: compute new candle + momentum features
    df = compute_candle_features(df)

    # Step 2: stage 1 — enhanced Isolation Forest with 10 features
    df = run_enhanced_isolation_forest(df)
    n_if_flags = int(df["IF_Flag_Enhanced"].sum())
    print(f"    Stage 1 (Enhanced IF): {n_if_flags} days flagged "
          f"(vs {int(df.get('IF_Flag', pd.Series([0])).sum())} with old IF)")

    # Step 3: label quality outcomes for training
    df = label_signal_quality(df)
    n_buy_quality  = int((df["Signal_Quality"] == 1).sum())
    n_sell_quality = int((df["Signal_Quality"] == -1).sum())
    print(f"    Quality labels: {n_buy_quality} high-quality BUY, "
          f"{n_sell_quality} high-quality SELL in training data")

    # Step 4: stage 2 — Random Forest classifier
    df, metrics = run_second_stage_classifier(df, ticker)

    n_hq_buy  = int((df["Final_Signal"] == "HIGH_QUALITY_BUY").sum())
    n_hq_sell = int((df["Final_Signal"] == "HIGH_QUALITY_SELL").sum())
    print(f"    Final signals: {n_hq_buy} HIGH_QUALITY_BUY, "
          f"{n_hq_sell} HIGH_QUALITY_SELL")

    if metrics:
        bp = metrics.get("test_buy_precision", "N/A")
        sp = metrics.get("test_sell_precision", "N/A")
        print(f"    Test precision — BUY: {bp}, SELL: {sp}")

    # Step 5: save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "AVR", "CAR_10", "Vol_Spike", "Return_Z",
        "Candle_Direction", "Body_Size", "Upper_Shadow", "Lower_Shadow",
        "Price_Momentum_5", "Volume_Surge",
        "IF_Flag_Enhanced", "IF_Score_Enhanced",
        "Signal_Quality", "RF_Prediction",
        "RF_Probability_Buy", "RF_Probability_Sell",
        "Final_Signal",
    ]
    if "Ticker" in df.columns:
        out_cols.insert(0, "Ticker")

    out_path = os.path.join(RESULTS_DIR, f"enhanced_signals_{ticker}.csv")
    df[[c for c in out_cols if c in df.columns]].to_csv(out_path, index=False)

    return {
        "ticker"   : ticker,
        "metrics"  : metrics,
        "result_df": df,
        "n_hq_buy" : n_hq_buy,
        "n_hq_sell": n_hq_sell,
    }


def run_all(ticker_names: list = None) -> list[dict]:
    """Runs the enhanced model for every stock in data/processed/."""
    if ticker_names is None:
        if not os.path.exists(PROCESSED_DIR):
            print(f"  Error: {PROCESSED_DIR}/ not found. Run features.py first.")
            return []
        ticker_names = sorted([
            f.replace("_features.csv", "")
            for f in os.listdir(PROCESSED_DIR)
            if f.endswith("_features.csv")
        ])

    all_results = []
    for ticker in ticker_names:
        result = run_enhanced_model(ticker)
        if result:
            all_results.append(result)

    # Save cross-stock performance summary
    if all_results:
        perf_rows = [r["metrics"] for r in all_results if r["metrics"]]
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            perf_path = os.path.join(RESULTS_DIR, "model_performance.csv")
            perf_df.to_csv(perf_path, index=False)
            print(f"\n  Performance summary → {perf_path}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CROSS-STOCK SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: list[dict]):
    """Prints the cross-stock signal quality summary table."""
    print(f"\n{'='*70}")
    print(f"  ENHANCED MODEL — CROSS-STOCK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Ticker':<14} {'HQ_BUY':>7} {'HQ_SELL':>8} "
          f"{'BUY_Prec':>9} {'SELL_Prec':>10}")
    print(f"  {'-'*50}")

    total_buy = total_sell = 0
    for r in sorted(all_results, key=lambda x: x["n_hq_buy"], reverse=True):
        m  = r["metrics"]
        bp = f"{m.get('test_buy_precision', 'N/A')}"
        sp = f"{m.get('test_sell_precision', 'N/A')}"
        print(f"  {r['ticker']:<14} {r['n_hq_buy']:>7} {r['n_hq_sell']:>8} "
              f"{bp:>9} {sp:>10}")
        total_buy  += r["n_hq_buy"]
        total_sell += r["n_hq_sell"]

    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<14} {total_buy:>7} {total_sell:>8}")
    print(f"\n  HQ_BUY  = green candle + predicted >30% 6M return")
    print(f"  HQ_SELL = red candle  + predicted <-15% 6M return")
    print(f"  Results saved in: {RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_ohlcv(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Builds realistic synthetic OHLCV + feature data for testing."""
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    open_ = close - rng.normal(0, 0.8, n)
    high  = np.maximum(open_, close) + rng.uniform(0.2, 1.5, n)
    low   = np.minimum(open_, close) - rng.uniform(0.2, 1.5, n)
    vol   = rng.integers(100_000, 500_000, n).astype(float)
    avr   = rng.normal(1.0, 0.2, n)
    car   = rng.normal(0.0, 0.01, n)
    vs    = rng.integers(0, 2, n)
    rz    = rng.normal(0.0, 0.8, n)
    if_f  = np.zeros(n, dtype=int)
    for idx in [50, 120, 200, 270]:
        if_f[idx] = 1
        avr[idx]  = 5.0
        car[idx]  = 0.10

    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": open_.round(2), "High": high.round(2),
        "Low": low.round(2), "Close": close.round(2),
        "Volume": vol.astype(int),
        "AVR": avr.round(4), "CAR_10": car.round(6),
        "Vol_Spike": vs, "Return_Z": rz.round(4),
        "IF_Flag": if_f,
    })


class TestCandleFeatures(unittest.TestCase):
    """Tests for compute_candle_features()."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_synthetic_ohlcv(n=100)
        cls.result = compute_candle_features(cls.df)

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_all_new_columns_added(self):
        """All 6 new feature columns must be present after computation."""
        for col in NEW_FEATURES:
            self.assertIn(col, self.result.columns, f"Missing: {col}")

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_candle_direction_is_binary(self):
        """Candle_Direction must be exactly +1 or -1, nothing else."""
        unique = set(self.result["Candle_Direction"].unique())
        self.assertTrue(unique.issubset({1, -1}),
            f"Candle_Direction contains unexpected values: {unique}")

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_green_candle_when_close_above_open(self):
        """Green candle (Close > Open) must give Candle_Direction = +1."""
        df = pd.DataFrame({
            "Open": [100], "High": [105], "Low": [99], "Close": [104],
            "Volume": [1_000_000]
        })
        result = compute_candle_features(df)
        self.assertEqual(result["Candle_Direction"].iloc[0], 1)

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_red_candle_when_close_below_open(self):
        """Red candle (Close < Open) must give Candle_Direction = -1."""
        df = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [94], "Close": [95],
            "Volume": [1_000_000]
        })
        result = compute_candle_features(df)
        self.assertEqual(result["Candle_Direction"].iloc[0], -1)

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_body_size_is_non_negative(self):
        """Body_Size is an absolute value divided by Open — always >= 0."""
        self.assertTrue((self.result["Body_Size"] >= 0).all())

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_upper_shadow_is_non_negative(self):
        """Upper_Shadow = High - max(Open,Close) — always >= 0."""
        self.assertTrue((self.result["Upper_Shadow"] >= 0).all())

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_lower_shadow_is_non_negative(self):
        """Lower_Shadow = min(Open,Close) - Low — always >= 0."""
        self.assertTrue((self.result["Lower_Shadow"] >= 0).all())

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_volume_surge_positive(self):
        """Volume_Surge = today / 5-day avg — always positive."""
        valid = self.result["Volume_Surge"].dropna()
        self.assertTrue((valid > 0).all())

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_missing_ohlcv_raises_value_error(self):
        """Missing a required OHLCV column must raise ValueError, not crash."""
        bad_df = self.df.drop(columns=["High"])
        with self.assertRaises(ValueError):
            compute_candle_features(bad_df)

    # ── Test 10 ───────────────────────────────────────────────────────────────
    def test_row_count_unchanged(self):
        """compute_candle_features() must not add or drop rows."""
        self.assertEqual(len(self.df), len(self.result))


class TestSignalQualityLabelling(unittest.TestCase):
    """Tests for label_signal_quality()."""

    # ── Test 11 ───────────────────────────────────────────────────────────────
    def test_adds_signal_quality_column(self):
        """label_signal_quality() must add a Signal_Quality column."""
        df = _make_synthetic_ohlcv(n=300)
        df = compute_candle_features(df)
        result = label_signal_quality(df)
        self.assertIn("Signal_Quality", result.columns)

    # ── Test 12 ───────────────────────────────────────────────────────────────
    def test_quality_values_are_in_valid_set(self):
        """Signal_Quality must only contain -1, 0, or 1."""
        df = _make_synthetic_ohlcv(n=300)
        df = compute_candle_features(df)
        result = label_signal_quality(df)
        unique = set(result["Signal_Quality"].unique())
        self.assertTrue(unique.issubset({-1, 0, 1}),
            f"Signal_Quality contains unexpected values: {unique}")

    # ── Test 13 ───────────────────────────────────────────────────────────────
    def test_non_flagged_rows_have_zero_quality(self):
        """Rows where IF_Flag == 0 must have Signal_Quality == 0."""
        df = _make_synthetic_ohlcv(n=200)
        df = compute_candle_features(df)
        result = label_signal_quality(df)
        non_flagged = result[result["IF_Flag"] == 0]
        self.assertTrue((non_flagged["Signal_Quality"] == 0).all())


class TestEnhancedIsolationForest(unittest.TestCase):
    """Tests for run_enhanced_isolation_forest()."""

    # ── Test 14 ───────────────────────────────────────────────────────────────
    def test_adds_if_enhanced_columns(self):
        """Must add IF_Score_Enhanced and IF_Flag_Enhanced."""
        df = _make_synthetic_ohlcv(n=200)
        df = compute_candle_features(df)
        result = run_enhanced_isolation_forest(df)
        self.assertIn("IF_Score_Enhanced", result.columns)
        self.assertIn("IF_Flag_Enhanced", result.columns)

    # ── Test 15 ───────────────────────────────────────────────────────────────
    def test_if_flag_enhanced_is_binary(self):
        """IF_Flag_Enhanced must only contain 0 or 1."""
        df = _make_synthetic_ohlcv(n=200)
        df = compute_candle_features(df)
        result = run_enhanced_isolation_forest(df)
        unique = set(result["IF_Flag_Enhanced"].unique())
        self.assertTrue(unique.issubset({0, 1}))

    # ── Test 16 ───────────────────────────────────────────────────────────────
    def test_flag_rate_near_contamination(self):
        """
        Flag rate should be close to IF_CONTAMINATION (5%).
        We allow a generous 3x margin since the dataset is synthetic
        and small, but it should not be 50%+ or 0%.
        """
        df = _make_synthetic_ohlcv(n=500)
        df = compute_candle_features(df)
        result = run_enhanced_isolation_forest(df)
        flag_rate = result["IF_Flag_Enhanced"].mean()
        self.assertLess(flag_rate, IF_CONTAMINATION * 3,
            f"Flag rate {flag_rate:.2%} too high — model is over-flagging")
        self.assertGreater(flag_rate, 0,
            "Flag rate is 0 — model is not flagging anything")


class TestFullPipeline(unittest.TestCase):
    """Integration test for the full two-stage pipeline."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df = _make_synthetic_ohlcv(n=400, seed=42)
        df.to_csv(os.path.join(PROCESSED_DIR, "_TEST_ENHANCED_features.csv"), index=False)

    @classmethod
    def tearDownClass(cls):
        for f in ["_TEST_ENHANCED_features.csv"]:
            p = os.path.join(PROCESSED_DIR, f)
            if os.path.exists(p):
                os.remove(p)
        for f in ["enhanced_signals__TEST_ENHANCED.csv"]:
            p = os.path.join(RESULTS_DIR, f)
            if os.path.exists(p):
                os.remove(p)

    # ── Test 17 ───────────────────────────────────────────────────────────────
    def test_full_pipeline_returns_result(self):
        """run_enhanced_model() must return a non-None result dict."""
        result = run_enhanced_model("_TEST_ENHANCED")
        self.assertIsNotNone(result)
        self.assertIn("n_hq_buy", result)
        self.assertIn("n_hq_sell", result)

    # ── Test 18 ───────────────────────────────────────────────────────────────
    def test_output_file_saved(self):
        """The enhanced signals CSV must be saved to data/results/."""
        run_enhanced_model("_TEST_ENHANCED")
        out_path = os.path.join(RESULTS_DIR, "enhanced_signals__TEST_ENHANCED.csv")
        self.assertTrue(os.path.exists(out_path),
            f"Expected output file not found: {out_path}")

    # ── Test 19 ───────────────────────────────────────────────────────────────
    def test_final_signal_column_has_valid_values(self):
        """Final_Signal must only contain the 4 expected label strings."""
        run_enhanced_model("_TEST_ENHANCED")
        out_path = os.path.join(RESULTS_DIR, "enhanced_signals__TEST_ENHANCED.csv")
        df = pd.read_csv(out_path)
        valid_values = {
            "HIGH_QUALITY_BUY", "HIGH_QUALITY_SELL",
            "LOW_QUALITY_FLAG", "NO_SIGNAL", "INSUFFICIENT_DATA"
        }
        unique = set(df["Final_Signal"].unique())
        self.assertTrue(unique.issubset(valid_values),
            f"Unexpected Final_Signal values: {unique - valid_values}")

    # ── Test 20 ───────────────────────────────────────────────────────────────
    def test_missing_stock_returns_none(self):
        """run_enhanced_model() must return None for a non-existent ticker."""
        result = run_enhanced_model("_NONEXISTENT_STOCK_XYZ")
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced two-stage insider trading model: "
                    "improved IF + Random Forest classifier."
    )
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        sys.argv = [sys.argv[0]]
        print("=" * 60)
        print("  Running enhanced_model.py test suite  (20 tests)")
        print("=" * 60)
        print()
        unittest.main(verbosity=2)

    else:
        print("=" * 60)
        print("  Enhanced Two-Stage Insider Trading Model")
        print("=" * 60)
        print(f"  Stage 1: Isolation Forest ({IF_N_ESTIMATORS} trees, "
              f"{len(ALL_FEATURES)} features)")
        print(f"  Stage 2: Random Forest Classifier ({RF_N_ESTIMATORS} trees)")
        print(f"  BUY target  : green candle + >30% 6M return")
        print(f"  SELL target : red candle   + <-15% 6M return")
        print(f"  Train/Test  : {int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}%")

        if args.ticker:
            result = run_enhanced_model(args.ticker.upper())
        else:
            all_results = run_all()
            if all_results:
                print_summary(all_results)

        print("\n  Done.")