# backend/models/zscore_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Apply Z-score based statistical anomaly detection on top of the
#           feature columns produced by features.py: AVR, CAR_10, Vol_Spike,
#           and Return_Z.
#
# WHY Z-SCORE, AND WHY AS A SEPARATE MODEL FROM features.py's OWN FLAGS:
#   features.py already raises individual flags using FIXED thresholds
#   (e.g. AVR > 2.5 everywhere). Those thresholds are the same for every
#   stock, which is too rigid — a highly liquid stock like RELIANCE
#   naturally has different volume variability than a smaller-cap stock.
#
#   Z-score instead asks: "how many standard deviations away from THIS
#   STOCK'S OWN normal distribution is this value?" This adapts the
#   threshold per-stock automatically, which is exactly what a second,
#   independent detection layer should do — it catches different cases
#   than the fixed-threshold flags, and agreement between the two methods
#   makes a flagged day far more trustworthy.
#
# COLUMN NAMES — IMPORTANT:
#   This file matches the EXACT column names your features.py produces:
#       AVR        (not lowercase "avr")
#       CAR_10     (not "car")
#       Vol_Spike  (not "vol_spike")
#       Return_Z   (already a Z-score itself, but we re-include it here
#                   so a combined "any extreme signal" flag considers it too)
#
# CONTAINS:
#   Section 1 — Imports & Configuration
#   Section 2 — compute_zscore()         : core Z-score formula on any column
#   Section 3 — apply_zscore_detection() : applies it to AVR, CAR_10, Vol_Spike
#   Section 4 — run_zscore_on_all()      : runs detection for every processed stock
#   Section 5 — Tests                    : unittest, run with --test
#   Section 6 — Entry point
#
# HOW TO RUN:
#   python backend/models/zscore_detector.py           → runs on all processed stocks
#   python backend/models/zscore_detector.py --test    → runs the test suite
#
# DEPENDENCY:
#   This file expects data/processed/{ticker}_features.csv to already exist,
#   containing AVR, CAR_10, Vol_Spike columns (produced by features.py).
#   Run `python backend/features.py` first if those files are missing.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import unittest

import numpy as np
import pandas as pd

PROCESSED_DIR = "data/processed"

# Z-score threshold — a value is flagged if |z-score| exceeds this.
# 2.5 standard deviations corresponds to roughly the top/bottom 0.6% of a
# normal distribution — a sensible "statistically rare" cutoff for finance.
ZSCORE_THRESHOLD = 2.5

# Which feature columns get a Z-score computed.
# THESE MUST MATCH features.py's ACTUAL output column names exactly,
# including capitalization - pandas column lookups are case-sensitive.
ZSCORE_COLUMNS = ["AVR", "CAR_10", "Vol_Spike"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — compute_zscore()
# ══════════════════════════════════════════════════════════════════════════════

def compute_zscore(series: pd.Series) -> pd.Series:
    """
    Computes the Z-score for every value in a pandas Series.

    FORMULA:
        z = (x - mean) / standard_deviation

    INTERPRETATION:
        z = 0    -> value is exactly the average
        z = 1    -> value is 1 standard deviation above average
        z = -2   -> value is 2 standard deviations below average
        |z| > 2.5 is considered statistically rare (see ZSCORE_THRESHOLD)

    IMPORTANT DESIGN DECISION:
        Mean and standard deviation are computed using ONLY the non-NaN
        values in the series (pandas .mean() and .std() skip NaN by
        default). This matters because AVR/CAR_10/Vol_Spike all have NaN
        for the first N rows (not enough rolling-window history yet) -
        those NaNs must not distort the baseline statistics used to judge
        every other row.

    Parameters
    ----------
    series : pd.Series
        Any numeric column, e.g. df["AVR"], df["CAR_10"], df["Vol_Spike"].
        May contain NaN values.

    Returns
    -------
    pd.Series
        Same length and index as input.
        NaN input values produce NaN output (we never invent a Z-score
        for a row that didn't have a valid feature value to begin with).
        Returns an all-NaN series if standard deviation is 0 or undefined
        (e.g. fewer than 2 valid values, or all valid values identical -
        this can genuinely happen with Vol_Spike, which is often 0 for
        most rows).
    """
    mean = series.mean()
    std = series.std()

    # Guard against division by zero - happens when all valid values are
    # identical (zero variance) or there's only 0-1 valid data points.
    # Vol_Spike in particular is frequently constant (mostly 0s), so this
    # guard is exercised often in real data, not just edge cases.
    if std == 0 or pd.isna(std):
        return pd.Series([np.nan] * len(series), index=series.index)

    z = (series - mean) / std
    return z.round(4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — apply_zscore_detection()
# ══════════════════════════════════════════════════════════════════════════════

def apply_zscore_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Z-score anomaly detection to all configured feature columns
    (ZSCORE_COLUMNS = AVR, CAR_10, Vol_Spike) and combines them into one
    overall flag.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in ZSCORE_COLUMNS.
        This is the output of features.py's build_features().

    Returns
    -------
    pd.DataFrame
        Same as input, with new columns added:
        - "{col}_Zscore"     : Z-score for each column in ZSCORE_COLUMNS
                                e.g. "AVR_Zscore", "CAR_10_Zscore",
                                "Vol_Spike_Zscore"
        - "{col}_Zscore_Flag": 1 if |zscore| > ZSCORE_THRESHOLD, else 0
        - "Zscore_Anomaly"   : 1 if ANY of the individual zscore flags is 1,
                                else 0. This is the single combined signal
                                scoring.py can optionally fold in later.
    """
    df = df.copy()

    individual_flag_columns = []

    for col in ZSCORE_COLUMNS:
        if col not in df.columns:
            print(f"  Warning: column '{col}' not found in DataFrame - skipping.")
            continue

        zscore_col = f"{col}_Zscore"
        flag_col = f"{col}_Zscore_Flag"

        df[zscore_col] = compute_zscore(df[col])
        df[flag_col] = df[zscore_col].apply(
            lambda z: 1 if pd.notna(z) and abs(z) > ZSCORE_THRESHOLD else 0
        )

        individual_flag_columns.append(flag_col)

    if individual_flag_columns:
        df["Zscore_Anomaly"] = (
            df[individual_flag_columns].sum(axis=1) > 0
        ).astype(int)
    else:
        # No valid columns were found at all - still produce the column
        # (filled with 0) so downstream code never has to special-case
        # a missing "Zscore_Anomaly" column.
        df["Zscore_Anomaly"] = 0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — run_zscore_on_all()
# ══════════════════════════════════════════════════════════════════════════════

def run_zscore_on_all(ticker_names: list = None) -> dict:
    """
    Loads every processed feature CSV from data/processed/, applies Z-score
    detection, and saves the result back (overwriting with the extra columns
    added) so scoring.py can later read one fully-enriched file per stock.

    Parameters
    ----------
    ticker_names : list, optional
        Clean ticker names to process, e.g. ["RELIANCE", "INFY"].
        If None, auto-detects every "{name}_features.csv" file in
        data/processed/.

    Returns
    -------
    dict  {ticker_name: pd.DataFrame}  — only successfully processed stocks
    """
    print("=" * 55)
    print("  Insider Trading Detector - Z-score Detection")
    print("=" * 55)

    if ticker_names is None:
        if not os.path.exists(PROCESSED_DIR):
            print(f"  Error: {PROCESSED_DIR}/ does not exist. Run features.py first.")
            return {}

        ticker_names = [
            f.replace("_features.csv", "")
            for f in os.listdir(PROCESSED_DIR)
            if f.endswith("_features.csv")
        ]

    results = {}

    for ticker_name in ticker_names:
        filepath = os.path.join(PROCESSED_DIR, f"{ticker_name}_features.csv")

        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found - skipping {ticker_name}.")
            continue

        print(f"  Processing {ticker_name} ...")
        df = pd.read_csv(filepath)

        missing = [c for c in ZSCORE_COLUMNS if c not in df.columns]
        if missing:
            print(f"    Warning: {ticker_name} is missing columns {missing} - "
                  f"make sure features.py has run on this file.")

        df = apply_zscore_detection(df)

        df.to_csv(filepath, index=False)
        results[ticker_name] = df

        anomaly_count = int(df["Zscore_Anomaly"].sum())
        print(f"    Done: {anomaly_count} anomalous day(s) flagged -> {filepath}")

    print()
    print("=" * 55)
    print(f"  Z-score detection complete: {len(results)} stocks processed")
    print(f"  Files updated in: {PROCESSED_DIR}/")
    print("=" * 55)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TESTS
# ══════════════════════════════════════════════════════════════════════════════
#
# HOW TO RUN:
#   python backend/models/zscore_detector.py --test
#
# These tests use small, hand-controlled synthetic data with the CORRECT
# (real) column names: AVR, CAR_10, Vol_Spike - matching exactly what
# features.py actually produces, so these tests catch the kind of
# column-name mismatch that caused the original bug.
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeZScore(unittest.TestCase):
    """Tests for compute_zscore()."""

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_known_values_match_manual_calculation(self):
        """
        For the series [10, 20, 30, 40, 50]:
            mean = 30, std (sample, ddof=1) = 15.811...
            z for value 50 = (50 - 30) / 15.811... = 1.2649...
        This is the single most important test - it proves the formula
        itself is correct, not just that it runs without crashing.
        """
        series = pd.Series([10, 20, 30, 40, 50])
        result = compute_zscore(series)

        expected_mean = series.mean()
        expected_std = series.std()
        expected_z_last = (50 - expected_mean) / expected_std

        self.assertAlmostEqual(
            result.iloc[-1], round(expected_z_last, 4), places=3,
            msg=f"Z-score mismatch. Expected ~{expected_z_last:.4f}, got {result.iloc[-1]}"
        )

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_mean_value_has_zscore_near_zero(self):
        """
        A value exactly equal to the series mean must have a Z-score of 0.
        """
        series = pd.Series([10, 20, 30, 40, 50])  # mean is exactly 30
        result = compute_zscore(series)
        zscore_of_mean_value = result.iloc[2]  # the value 30 is at index 2

        self.assertAlmostEqual(
            zscore_of_mean_value, 0.0, places=4,
            msg=f"Expected Z-score 0 for the mean value, got {zscore_of_mean_value}"
        )

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_constant_series_returns_all_nan(self):
        """
        If every value in the series is identical, standard deviation is 0.
        Dividing by zero would crash or produce inf - compute_zscore must
        instead return all-NaN, signalling 'no meaningful variation to judge.'

        This is exercised heavily in real Vol_Spike data, which is often
        all-zero for stretches of a stock's history.
        """
        series = pd.Series([0, 0, 0, 0, 0])  # mirrors a quiet Vol_Spike stretch
        result = compute_zscore(series)

        self.assertTrue(
            result.isna().all(),
            "Expected all-NaN result for a constant (zero-variance) series"
        )

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_nan_input_values_produce_nan_output(self):
        """
        NaN values in the input (e.g. early rows of AVR/CAR_10/Vol_Spike
        that don't have enough history yet) must remain NaN in the
        output - never silently converted to 0 or any fabricated number.
        """
        series = pd.Series([10, 20, np.nan, 40, 50])
        result = compute_zscore(series)

        self.assertTrue(
            pd.isna(result.iloc[2]),
            "Expected NaN output at the position where input was NaN"
        )
        self.assertFalse(
            pd.isna(result.iloc[0]),
            "Non-NaN input values should not produce NaN output"
        )

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_nan_values_excluded_from_mean_and_std_calculation(self):
        """
        The presence of a NaN in the series must not distort the mean/std
        used to score the OTHER values.
        """
        series_with_nan = pd.Series([10, 20, np.nan, 40, 50])
        series_without_nan = pd.Series([10, 20, 40, 50])

        result_with_nan = compute_zscore(series_with_nan)
        result_without_nan = compute_zscore(series_without_nan)

        self.assertAlmostEqual(
            result_with_nan.iloc[0], result_without_nan.iloc[0], places=4,
            msg="Z-score of a valid value should be unaffected by NaN elsewhere"
        )

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_symmetric_extreme_values_have_opposite_sign_zscores(self):
        """
        A value far above the mean must have a positive Z-score, and a
        value far below the mean must have a negative Z-score.
        """
        series = pd.Series([50, 51, 49, 50, 200, 50, -100, 50])
        result = compute_zscore(series)

        high_value_zscore = result.iloc[4]   # corresponds to 200
        low_value_zscore = result.iloc[6]    # corresponds to -100

        self.assertGreater(high_value_zscore, 0,
            "Value far above the mean should have a positive Z-score")
        self.assertLess(low_value_zscore, 0,
            "Value far below the mean should have a negative Z-score")


class TestApplyZscoreDetection(unittest.TestCase):
    """Tests for apply_zscore_detection() using REAL features.py column names."""

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_adds_expected_columns_with_correct_names(self):
        """
        For each column in ZSCORE_COLUMNS (AVR, CAR_10, Vol_Spike), the
        function must add "{col}_Zscore" and "{col}_Zscore_Flag" using the
        EXACT real column names - this is precisely the bug that broke
        the original version of this file.
        """
        df = pd.DataFrame({
            "AVR": [1.0, 1.1, 0.9, 1.2, 1.0, 10.0],
            "CAR_10": [0.01, 0.02, -0.01, 0.0, 0.01, 0.02],
            "Vol_Spike": [0, 0, 1, 0, 0, 1],
        })
        result = apply_zscore_detection(df)

        self.assertIn("AVR_Zscore", result.columns)
        self.assertIn("AVR_Zscore_Flag", result.columns)
        self.assertIn("CAR_10_Zscore", result.columns)
        self.assertIn("CAR_10_Zscore_Flag", result.columns)
        self.assertIn("Vol_Spike_Zscore", result.columns)
        self.assertIn("Vol_Spike_Zscore_Flag", result.columns)
        self.assertIn("Zscore_Anomaly", result.columns)

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_injected_avr_outlier_is_flagged(self):
        """
        Build a DataFrame where AVR is calm (around 1.0) for 9 rows, then
        spikes to 10.0 on the 10th row. The Z-score flag for that row
        must be 1, and Zscore_Anomaly for that row must also be 1.
        """
        avr_values = [1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 10.0]
        df = pd.DataFrame({
            "AVR": avr_values,
            "CAR_10": [0.0] * 10,
            "Vol_Spike": [0] * 10,
        })

        result = apply_zscore_detection(df)
        last_row_flag = result["AVR_Zscore_Flag"].iloc[-1]
        last_row_anomaly = result["Zscore_Anomaly"].iloc[-1]

        self.assertEqual(last_row_flag, 1,
            "Expected AVR_Zscore_Flag=1 for the injected outlier row")
        self.assertEqual(last_row_anomaly, 1,
            "Expected Zscore_Anomaly=1 when at least one underlying flag fires")

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_normal_rows_are_not_flagged(self):
        """
        Rows with no extreme values in any column should have
        Zscore_Anomaly = 0.
        """
        avr_values = [1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 1.04]
        df = pd.DataFrame({
            "AVR": avr_values,
            "CAR_10": [0.0, 0.01, -0.01, 0.005, -0.005, 0.0, 0.01, -0.01, 0.0, 0.005],
            "Vol_Spike": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })

        result = apply_zscore_detection(df)
        anomaly_count = result["Zscore_Anomaly"].sum()

        self.assertLessEqual(
            anomaly_count, 1,
            f"Expected at most 1 false-positive anomaly in calm data, got {anomaly_count}"
        )

    # ── Test 10 ───────────────────────────────────────────────────────────────
    def test_missing_column_is_skipped_gracefully(self):
        """
        If the input DataFrame is missing one of the expected columns
        (e.g. "CAR_10" wasn't computed), the function must skip it with
        a warning, not crash, and still process the columns that ARE present.
        """
        df = pd.DataFrame({
            "AVR": [1.0, 1.1, 0.9, 1.2, 1.0, 10.0],
            # "CAR_10" deliberately missing
            "Vol_Spike": [0, 1, 0, 1, 0, 1],
        })

        result = apply_zscore_detection(df)

        self.assertIn("AVR_Zscore", result.columns)
        self.assertNotIn("CAR_10_Zscore", result.columns)
        self.assertIn("Vol_Spike_Zscore", result.columns)
        self.assertIn("Zscore_Anomaly", result.columns)

    # ── Test 11 ───────────────────────────────────────────────────────────────
    def test_nan_feature_rows_do_not_count_as_anomalies(self):
        """
        Early rows in real feature data have NaN for AVR/CAR_10 (not
        enough history yet). Those rows must NOT be counted as anomalies
        just because they're NaN.
        """
        df = pd.DataFrame({
            "AVR": [np.nan, np.nan, 1.0, 1.1, 0.9],
            "CAR_10": [np.nan, np.nan, 0.01, 0.02, -0.01],
            "Vol_Spike": [0, 0, 0, 1, 0],
        })

        result = apply_zscore_detection(df)
        first_two_rows_anomaly = result["Zscore_Anomaly"].iloc[:2]

        self.assertTrue(
            (first_two_rows_anomaly == 0).all(),
            "Expected Zscore_Anomaly=0 for rows where the underlying feature was NaN"
        )

    # ── Test 12 ───────────────────────────────────────────────────────────────
    def test_vol_spike_all_zero_does_not_crash(self):
        """
        Vol_Spike is frequently ALL ZEROS for long stretches of real data
        (no volatility spike most days). This must not crash - it should
        produce all-NaN Z-scores for that column (zero variance) and
        contribute 0 to Zscore_Anomaly, not raise a division-by-zero error.
        """
        df = pd.DataFrame({
            "AVR": [1.0, 1.05, 0.95, 1.02, 0.98],
            "CAR_10": [0.0, 0.01, -0.01, 0.005, -0.005],
            "Vol_Spike": [0, 0, 0, 0, 0],  # all zero, zero variance
        })

        # Should not raise any exception
        result = apply_zscore_detection(df)

        self.assertTrue(
            result["Vol_Spike_Zscore"].isna().all(),
            "Expected all-NaN Vol_Spike_Zscore for a zero-variance column"
        )
        self.assertTrue(
            (result["Vol_Spike_Zscore_Flag"] == 0).all(),
            "Expected Vol_Spike_Zscore_Flag=0 everywhere when the column has no variance"
        )


class TestRunZscoreOnAll(unittest.TestCase):
    """Tests for run_zscore_on_all() — the integration / file-handling layer."""

    @classmethod
    def setUpClass(cls):
        cls.test_ticker = "_TESTSTOCK_ZSCORE"
        cls.test_filepath = os.path.join(PROCESSED_DIR, f"{cls.test_ticker}_features.csv")

        os.makedirs(PROCESSED_DIR, exist_ok=True)

        # Build a small synthetic "already feature-engineered" CSV using
        # the REAL column names, as features.py would actually produce.
        n = 70
        rng = np.random.default_rng(seed=11)
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=n, freq="B"),
            "AVR": rng.normal(1.0, 0.1, n),
            "CAR_10": rng.normal(0.0, 0.01, n),
            "Vol_Spike": rng.integers(0, 2, n),
        })
        df.loc[n - 1, "AVR"] = 15.0  # inject one clear outlier

        df.to_csv(cls.test_filepath, index=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_filepath):
            os.remove(cls.test_filepath)

    # ── Test 13 ───────────────────────────────────────────────────────────────
    def test_processes_given_ticker_and_returns_dict(self):
        """
        run_zscore_on_all() called with our synthetic test ticker must
        process it successfully and return a dict containing it.
        """
        results = run_zscore_on_all(ticker_names=[self.test_ticker])
        self.assertIn(self.test_ticker, results)
        self.assertIsInstance(results[self.test_ticker], pd.DataFrame)

    # ── Test 14 ───────────────────────────────────────────────────────────────
    def test_saved_csv_contains_new_columns(self):
        """
        After running, the CSV on disk must be overwritten with the new
        Z-score columns included, using the correct real column-name
        convention.
        """
        run_zscore_on_all(ticker_names=[self.test_ticker])
        saved_df = pd.read_csv(self.test_filepath)

        self.assertIn("Zscore_Anomaly", saved_df.columns)
        self.assertIn("AVR_Zscore", saved_df.columns)

    # ── Test 15 ───────────────────────────────────────────────────────────────
    def test_injected_outlier_detected_end_to_end(self):
        """
        The 15.0 AVR outlier injected in setUpClass must be flagged as
        an anomaly after the full run_zscore_on_all() pipeline.
        """
        run_zscore_on_all(ticker_names=[self.test_ticker])
        saved_df = pd.read_csv(self.test_filepath)

        last_row_anomaly = saved_df["Zscore_Anomaly"].iloc[-1]
        self.assertEqual(
            last_row_anomaly, 1,
            "Expected the injected AVR outlier to be flagged end-to-end"
        )

    # ── Test 16 ───────────────────────────────────────────────────────────────
    def test_missing_ticker_is_skipped_not_crashed(self):
        """
        Requesting a ticker whose CSV doesn't exist must be skipped with
        a warning, not raise an exception.
        """
        results = run_zscore_on_all(ticker_names=["_NONEXISTENT_TICKER_ABC"])
        self.assertNotIn("_NONEXISTENT_TICKER_ABC", results)
        self.assertEqual(len(results), 0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
#   python backend/models/zscore_detector.py           → run on all processed stocks
#   python backend/models/zscore_detector.py --test    → run the 16-test suite
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if "--test" in sys.argv:
        sys.argv.remove("--test")

        print("=" * 55)
        print("  Running zscore_detector.py test suite  (16 tests)")
        print("=" * 55)
        print()

        unittest.main(verbosity=2)

    else:
        run_zscore_on_all()