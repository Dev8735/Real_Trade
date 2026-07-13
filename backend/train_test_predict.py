# backend/train_test_predict.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Train anomaly detection models on 8 years of historical data,
#           test their performance on a held-out 2-year test set, and then
#           generate predictions (Suspicion Scores) for every stock.
#
# WHY TRAIN/TEST SPLIT FOR AN UNSUPERVISED MODEL:
#   Isolation Forest and Z-score are unsupervised — they have no labeled
#   "this was definitely insider trading" data to learn from. Instead:
#
#   TRAIN (first 8 years, ~2000 rows):
#       The model learns what "normal" looks like for each stock —
#       the typical distribution of AVR, CAR_10, Vol_Spike, Return_Z.
#
#   TEST (last 2 years, ~500 rows):
#       The model scores the UNSEEN recent period using only the baseline
#       it learned from the training set. This is the honest evaluation:
#       we check whether the model flags the same days as suspicious when
#       it only knows the stock's older history, not the test period itself.
#
#   PREDICT:
#       Generates the final suspicion score for every day in the FULL
#       10-year window, sorted by most suspicious — ready for reporting.
#
# HOW TO RUN:
#   python backend/train_test_predict.py              → processes all stocks
#   python backend/train_test_predict.py RELIANCE     → one specific stock
#   python backend/train_test_predict.py --top 10     → show top 10 flagged days
#
# DEPENDENCY:
#   data/processed/{ticker}_features.csv must exist with columns:
#   AVR, CAR_10, Vol_Spike, Return_Z  (produced by features.py)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROCESSED_DIR    = "data/processed"
RESULTS_DIR      = "data/results"

MODEL_FEATURES   = ["AVR", "CAR_10", "Vol_Spike", "Return_Z"]
CONTAMINATION    = 0.05      # expected proportion of anomalous days
N_ESTIMATORS     = 200       # more trees = more stable scores on 10y data
RANDOM_STATE     = 42

TRAIN_RATIO      = 0.8       # 80% train (~8 years), 20% test (~2 years)
SUSPICION_THRESHOLD = 65     # score >= 65 is flagged in the final output

# Scoring weights (must sum to 100)
WEIGHT_AVR       = 25
WEIGHT_CAR       = 25
WEIGHT_IF        = 30
WEIGHT_PROXIMITY = 20

AVR_THRESHOLD    = 2.5
CAR_THRESHOLD    = 0.08
PROXIMITY_WINDOW = 10


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: scale a raw signal value to 0–1
# ══════════════════════════════════════════════════════════════════════════════

def scale_component(value: float, threshold: float, cap_multiple: float = 3.0) -> float:
    """Scales a raw signal to 0-1 based on how far past its threshold it is."""
    if pd.isna(value) or value <= threshold:
        return 0.0
    cap = threshold * cap_multiple
    if value >= cap:
        return 1.0
    return round((value - threshold) / (cap - threshold), 4)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: compute event proximity (recency of nearest anomaly)
# ══════════════════════════════════════════════════════════════════════════════

def compute_proximity(if_flags: np.ndarray, car_10: np.ndarray) -> np.ndarray:
    """Returns a 0-1 proximity score per row based on distance to nearest event."""
    n = len(if_flags)
    is_event = (if_flags == 1) | (np.abs(car_10) > CAR_THRESHOLD)
    proximity = np.zeros(n)

    for t in range(n):
        window_start = max(0, t - PROXIMITY_WINDOW + 1)
        window = is_event[window_start: t + 1]
        if not window.any():
            continue
        event_positions = np.where(window)[0]
        nearest = event_positions[-1]
        days_back = t - (window_start + nearest)
        proximity[t] = round(1 - (days_back / PROXIMITY_WINDOW), 4)

    return proximity


# ══════════════════════════════════════════════════════════════════════════════
# CORE: train, test, predict for one stock
# ══════════════════════════════════════════════════════════════════════════════

def run_stock(ticker: str, top_n: int = 20) -> dict | None:
    """
    Full train → test → predict pipeline for a single stock.

    Returns a dict with keys:
        ticker, train_rows, test_rows,
        test_anomalies_detected, test_flag_rate,
        top_suspicious_days (DataFrame)
    """
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found — run features.py first.")
        return None

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        print(f"  [SKIP] {ticker} missing columns: {missing} — run features.py first.")
        return None

    # ── Train / Test split ────────────────────────────────────────────────────
    n = len(df)
    split_idx = int(n * TRAIN_RATIO)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"\n  {'='*50}")
    print(f"  Stock        : {ticker}")
    print(f"  Total rows   : {n} trading days "
          f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    print(f"  Train set    : {len(train_df)} rows "
          f"({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  Test  set    : {len(test_df)} rows "
          f"({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    # ── Prepare inputs ────────────────────────────────────────────────────────
    def prep(frame):
        return frame[MODEL_FEATURES].fillna(0.0).values

    X_train = prep(train_df)
    X_test  = prep(test_df)
    X_full  = prep(df)

    # ── Train model on training set ───────────────────────────────────────────
    print(f"\n  TRAINING Isolation Forest on {len(train_df)} rows ...")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)
    print(f"  Training complete. ({N_ESTIMATORS} trees, "
          f"contamination={CONTAMINATION})")

    # ── TEST: score the held-out test set ─────────────────────────────────────
    print(f"\n  TESTING on held-out {len(test_df)} rows (last 20% of history)...")
    test_scores  = model.decision_function(X_test)
    test_preds   = model.predict(X_test)   # -1 = anomaly, +1 = normal

    test_df["IF_Score"] = np.round(test_scores, 4)
    test_df["IF_Flag"]  = (test_preds == -1).astype(int)

    n_anomalies = test_df["IF_Flag"].sum()
    flag_rate   = round(test_df["IF_Flag"].mean() * 100, 2)

    print(f"  Anomalous days detected in test period : {n_anomalies}")
    print(f"  Flag rate in test period               : {flag_rate}%")
    print(f"  (Expected ~{CONTAMINATION*100:.0f}% based on contamination setting)")

    # Show the most anomalous days in the test period
    top_test = (
        test_df[test_df["IF_Flag"] == 1]
        [["Date", "AVR", "CAR_10", "Vol_Spike", "Return_Z", "IF_Score"]]
        .sort_values("IF_Score")
        .head(5)
    )
    if not top_test.empty:
        print(f"\n  Top anomalous days in TEST period:")
        print(f"  {'Date':<12} {'AVR':>6} {'CAR_10':>8} "
              f"{'Vol_Spike':>10} {'Return_Z':>9} {'IF_Score':>9}")
        print(f"  {'-'*56}")
        for _, row in top_test.iterrows():
            print(f"  {str(row['Date'].date()):<12} "
                  f"{row['AVR']:>6.2f} "
                  f"{row['CAR_10']:>8.4f} "
                  f"{int(row['Vol_Spike']):>10} "
                  f"{row['Return_Z']:>9.2f} "
                  f"{row['IF_Score']:>9.4f}")

    # ── PREDICT: score every day in the full 10-year window ──────────────────
    print(f"\n  PREDICTING on full 10-year window ({n} rows)...")
    full_scores = model.decision_function(X_full)
    full_preds  = model.predict(X_full)

    df["IF_Score"] = np.round(full_scores, 4)
    df["IF_Flag"]  = (full_preds == -1).astype(int)
    df["IF_Rank"]  = df["IF_Score"].rank(method="min", ascending=True).astype(int)

    # ── Compute suspicion score ───────────────────────────────────────────────
    df["AVR_Component"] = df["AVR"].apply(
        lambda v: scale_component(v, AVR_THRESHOLD)
    )
    df["CAR_Component"] = df["CAR_10"].abs().apply(
        lambda v: scale_component(v, CAR_THRESHOLD)
    )
    df["Proximity_Component"] = compute_proximity(
        df["IF_Flag"].values,
        df["CAR_10"].fillna(0).values
    )

    df["Suspicion_Score"] = (
        df["AVR_Component"]       * WEIGHT_AVR
        + df["CAR_Component"]     * WEIGHT_CAR
        + df["IF_Flag"]           * WEIGHT_IF
        + df["Proximity_Component"] * WEIGHT_PROXIMITY
    ).round(2)

    df["Suspicion_Flag"] = (df["Suspicion_Score"] >= SUSPICION_THRESHOLD).astype(int)

    # ── Save full predictions ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{ticker}_predictions.csv")
    df.to_csv(out_path, index=False)

    # ── Print top suspicious days ─────────────────────────────────────────────
    flagged_total = df["Suspicion_Flag"].sum()
    max_score     = df["Suspicion_Score"].max()
    print(f"\n  PREDICTIONS complete.")
    print(f"  Total flagged days (score >= {SUSPICION_THRESHOLD}) : {flagged_total}")
    print(f"  Highest Suspicion Score             : {max_score}")
    print(f"  Results saved to                    : {out_path}")

    top_days = (
        df[df["Suspicion_Flag"] == 1]
        [["Date", "AVR", "CAR_10", "Vol_Spike", "Return_Z",
          "IF_Flag", "Suspicion_Score"]]
        .sort_values("Suspicion_Score", ascending=False)
        .head(top_n)
    )

    if not top_days.empty:
        print(f"\n  TOP {min(top_n, len(top_days))} MOST SUSPICIOUS DAYS "
              f"(score >= {SUSPICION_THRESHOLD}):")
        print(f"  {'Date':<12} {'AVR':>6} {'CAR_10':>8} "
              f"{'Spike':>6} {'RetZ':>6} {'IF':>3} {'Score':>7}")
        print(f"  {'-'*57}")
        for _, row in top_days.iterrows():
            print(f"  {str(row['Date'].date()):<12} "
                  f"{row['AVR']:>6.2f} "
                  f"{row['CAR_10']:>8.4f} "
                  f"{int(row['Vol_Spike']):>6} "
                  f"{row['Return_Z']:>6.2f} "
                  f"{int(row['IF_Flag']):>3} "
                  f"{row['Suspicion_Score']:>7.2f}")
    else:
        print(f"\n  No days scored above {SUSPICION_THRESHOLD} threshold.")
        print(f"  Showing top 5 by score regardless:")
        top5 = df.nlargest(5, "Suspicion_Score")[
            ["Date", "AVR", "CAR_10", "Suspicion_Score"]
        ]
        print(top5.to_string(index=False))

    return {
        "ticker"                  : ticker,
        "train_rows"              : len(train_df),
        "test_rows"               : len(test_df),
        "test_anomalies_detected" : int(n_anomalies),
        "test_flag_rate"          : flag_rate,
        "total_flagged_days"      : int(flagged_total),
        "max_suspicion_score"     : float(max_score),
        "top_suspicious_days"     : top_days,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: run all stocks and print a cross-stock summary
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train, test and predict insider trading anomalies "
                    "on 10 years of data."
    )
    parser.add_argument(
        "ticker", nargs="?", default=None,
        help="Optional: run for a single ticker only (e.g. RELIANCE). "
             "If omitted, runs for every stock in data/processed/."
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top suspicious days to display per stock (default: 20)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Insider Trading Pattern Detector")
    print("  Train / Test / Predict — 10-Year Data")
    print("=" * 60)
    print(f"  Model        : Isolation Forest "
          f"({N_ESTIMATORS} trees, contamination={CONTAMINATION})")
    print(f"  Split        : {int(TRAIN_RATIO*100)}% train / "
          f"{int((1-TRAIN_RATIO)*100)}% test")
    print(f"  Features     : {MODEL_FEATURES}")
    print(f"  Alert cutoff : Suspicion Score >= {SUSPICION_THRESHOLD}")

    if args.ticker:
        ticker_names = [args.ticker.upper()]
    else:
        if not os.path.exists(PROCESSED_DIR):
            print(f"\n  Error: {PROCESSED_DIR}/ not found. "
                  f"Run features.py first.")
            sys.exit(1)
        ticker_names = sorted([
            f.replace("_features.csv", "")
            for f in os.listdir(PROCESSED_DIR)
            if f.endswith("_features.csv")
        ])

    if not ticker_names:
        print(f"\n  No feature files found in {PROCESSED_DIR}/")
        sys.exit(1)

    print(f"\n  Stocks to process: {ticker_names}")

    # Run pipeline for each stock
    all_results = []
    for ticker in ticker_names:
        result = run_stock(ticker, top_n=args.top)
        if result:
            all_results.append(result)

    # ── Cross-stock summary ───────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n")
        print("=" * 60)
        print("  CROSS-STOCK SUMMARY")
        print("=" * 60)
        print(f"  {'Ticker':<14} {'Train':>6} {'Test':>6} "
              f"{'TestFlags':>10} {'FlagRate':>9} "
              f"{'TotalFlags':>11} {'MaxScore':>9}")
        print(f"  {'-'*68}")

        all_results_sorted = sorted(
            all_results, key=lambda r: r["max_suspicion_score"], reverse=True
        )

        for r in all_results_sorted:
            print(f"  {r['ticker']:<14} "
                  f"{r['train_rows']:>6} "
                  f"{r['test_rows']:>6} "
                  f"{r['test_anomalies_detected']:>10} "
                  f"{r['test_flag_rate']:>8.1f}% "
                  f"{r['total_flagged_days']:>11} "
                  f"{r['max_suspicion_score']:>9.2f}")

        print(f"\n  Results saved in: {RESULTS_DIR}/")
        print(f"  Each file: {{ticker}}_predictions.csv "
              f"(full 10-year scored history)")

    print("\n  Done.")


if __name__ == "__main__":
    main()