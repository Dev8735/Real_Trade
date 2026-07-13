# backend/features.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Read raw OHLCV CSVs produced by ingest.py and compute four
#           statistical signals that together describe "abnormal" trading
#           behaviour — the footprints of potential insider activity.
#
# THE FOUR SIGNALS COMPUTED HERE:
#
#   1. AVR  — Abnormal Volume Ratio
#              Did unusually high/low trading volume occur recently?
#
#   2. CAR  — Cumulative Abnormal Return
#              Did the stock move more than the overall market (Nifty 50)?
#
#   3. VOL_SPIKE — Volatility Spike Flag
#              Is the stock swinging more wildly than its own history?
#
#   4. RETURN_Z  — Daily Return Z-score
#              Is today's return a statistical outlier vs the past 60 days?
#
# CONTAINS:
#   Section 1 — Imports & Constants
#   Section 2 — load_stock()         : read one stock CSV from disk
#   Section 3 — compute_avr()        : Abnormal Volume Ratio
#   Section 4 — compute_car()        : Cumulative Abnormal Return
#   Section 5 — compute_vol_spike()  : Volatility Spike
#   Section 6 — compute_return_z()   : Return Z-score
#   Section 7 — build_features()     : master function — runs all 4 signals
#   Section 8 — Tests                : 24 integrated tests using unittest
#   Section 9 — Entry point          : run features OR tests from CLI
#
# HOW TO RUN:
#   python backend/features.py           → computes features for all stocks
#   python backend/features.py --test    → runs all 24 tests
#
# DEPENDENCY:
#   ingest.py must have been run first so data/raw/*.csv files exist.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import unittest

import numpy as np
import pandas as pd


# Paths — must match what ingest.py wrote to
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

# Rolling window sizes (in trading days)
# SHORT_WINDOW : the "suspicious" period just before a corporate event
# LONG_WINDOW  : the historical baseline we compare the short window against
SHORT_WINDOW = 5    # ~1 trading week  (unchanged — best suspicious window size)
LONG_WINDOW  = 252  # ~1 full trading year (was 60 days/3 months — much stronger
                    # baseline with 10 years of data available)

# Z-score threshold: daily returns beyond ±2.5 standard deviations are flagged
Z_SCORE_THRESHOLD = 2.5

# Volatility spike: if short-term std dev is more than this multiple of
# long-term std dev, flag it as a volatility spike
VOL_SPIKE_RATIO = 2.0

# Tickers to process — must match those downloaded by ingest.py
TICKERS = [
    # Defence
    "PARAS", "DATAPATTNS", "ASTRAMICRO", "SIKAINTERP", "AVANTEL",
    # Power (PIGL = Power & Instrumentation Gujarat, listed Apr 2018)
    "CESC", "GENUSPOWER", "SKIPPER", "RTNPOWER", "PIGL",
    # Capital Goods (RITES and TEXRAIL also cover Railways — not duplicated)
    "CGPOWER", "APARINDS", "RITES", "VESUVIUS", "TEXRAIL",
    # Pharma
    "LAURUSLABS", "AARTIDRUGS", "INNOVACAP", "SOLARA", "FDC",
    # Chemicals
    "AARTIIND", "DEEPAKNTR", "NEONAMINES", "BALAMINES", "PRIVISCL",
    # NBFC
    "POONAWALLA", "FIVESTAR", "MUTHOOTFIN", "CHOLAFIN", "FEDFINA",
    # Railways (RITES and TEXRAIL already above — not duplicated)
    "TITAGARH", "RVNL", "IRCON",
    # Auto
    "SONACOMS", "ENDURANCE", "MSUMI", "TUBEINVEST", "BHARATFORG",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — load_stock()
# ══════════════════════════════════════════════════════════════════════════════

def load_stock(ticker: str) -> pd.DataFrame | None:
    """
    Reads the raw OHLCV CSV for one ticker from disk.

    Parameters
    ----------
    ticker : str
        Clean ticker name WITHOUT the .NS suffix, e.g. "RELIANCE"
        (this is how ingest.py saved the files)

    Returns
    -------
    pd.DataFrame
        Columns: [Date, Open, High, Low, Close, Volume]
        Date is parsed as a proper datetime object (not a string) so we can
        do date arithmetic (subtraction, comparison, groupby) downstream.

    Returns None if the CSV file does not exist on disk.

    WHY WE PARSE DATES HERE:
        ingest.py saved Date as a YYYY-MM-DD string. Here we convert it
        to a pandas Timestamp so we can do things like:
            df[df["Date"] >= pd.Timestamp("2024-01-01")]
        String comparison of dates is unreliable and error-prone.
    """
    filepath = os.path.join(RAW_DIR, f"{ticker}_ohlcv.csv")

    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath}  — run ingest.py first")
        return None

    df = pd.read_csv(filepath, parse_dates=["Date"])

    # Sort by date ascending — oldest row first.
    # Rolling window calculations depend on correct chronological order.
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_nifty() -> pd.DataFrame | None:
    """
    Reads the Nifty 50 benchmark CSV saved by ingest.py.

    Returns
    -------
    pd.DataFrame
        Columns: [Date, Nifty_Close, Nifty_Return]
        Date is a datetime object.

    Returns None if the file does not exist.
    """
    filepath = os.path.join(RAW_DIR, "NIFTY50_benchmark.csv")

    if not os.path.exists(filepath):
        print(f"  ⚠️  Nifty benchmark not found at {filepath} — run ingest.py first")
        return None

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — compute_avr()
# ══════════════════════════════════════════════════════════════════════════════

def compute_avr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Abnormal Volume Ratio (AVR) for every trading day.

    WHAT IS AVR?
        AVR measures whether today's trading volume is unusually high
        compared to the stock's own historical average volume.

        Formula for each row t:
            AVR(t) = Volume(t) / RollingMean(Volume, last 60 days)(t)

        An AVR of 1.0 means volume is exactly average.
        An AVR of 3.0 means 3x the normal volume — suspicious.
        An AVR of 0.4 means only 40% of normal volume — also notable.

    WHY AVR MATTERS FOR INSIDER DETECTION:
        Insiders often buy/sell in large quantities just before an
        announcement. This drives volume up sharply while prices may
        look normal, making volume the earliest detectable signal.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a "Volume" column (integers).
        Typically the output of load_stock().

    Returns
    -------
    pd.DataFrame
        Original df with two new columns added:
        - "Volume_MA60"  : 60-day rolling mean of Volume (the baseline)
        - "AVR"          : Volume / Volume_MA60

        Rows where Volume_MA60 is NaN (first 59 rows — not enough history)
        will have AVR = NaN. This is correct and expected.

    Notes
    -----
    min_periods=1 is intentional on Volume_MA60: it allows partial windows
    at the start of the dataset. Without it, the first 59 rows would all
    be NaN for Volume_MA60, which would also make AVR NaN for those rows.
    We still get NaN for rows where Volume itself is 0 (division by zero
    is handled by replacing 0-volume baseline with NaN before dividing).
    """
    df = df.copy()  # never mutate the input DataFrame

    # 60-day rolling mean of Volume.
    # min_periods=1 means: start computing from the very first row,
    # using however many rows are available (even if fewer than 60).
    df["Volume_MA60"] = (
        df["Volume"]
        .rolling(window=LONG_WINDOW, min_periods=1)
        .mean()
    )

    # Guard against division by zero: if Volume_MA60 is 0 (extremely rare
    # but theoretically possible for illiquid stocks), set it to NaN so
    # the division produces NaN rather than inf.
    df["Volume_MA60"] = df["Volume_MA60"].replace(0, np.nan)

    # AVR = today's volume divided by the rolling 60-day average
    df["AVR"] = (df["Volume"] / df["Volume_MA60"]).round(4)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — compute_car()
# ══════════════════════════════════════════════════════════════════════════════

def compute_car(df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Cumulative Abnormal Return (CAR) over a rolling 10-day window.

    WHAT IS CAR?
        CAR measures how much a stock's price movement exceeds or lags
        the overall market (Nifty 50) over a rolling window.

        Step 1 — Daily stock return:
            Stock_Return(t) = (Close(t) - Close(t-1)) / Close(t-1)

        Step 2 — Daily abnormal return:
            Abnormal_Return(t) = Stock_Return(t) - Nifty_Return(t)

        Step 3 — Cumulative abnormal return (10-day rolling sum):
            CAR(t) = sum of Abnormal_Return for the last 10 days

    WHY CAR MATTERS:
        If a stock climbs 8% over 10 days while Nifty only gained 1%,
        the CAR = +7%. That excess return suggests someone is buying
        aggressively — possibly on non-public information.

        A large negative CAR before a bad announcement (earnings miss,
        fraud revelation) is equally suspicious — insiders selling short.

    Parameters
    ----------
    df : pd.DataFrame
        Stock OHLCV data. Must have "Date" and "Close" columns.
    nifty_df : pd.DataFrame
        Nifty 50 benchmark. Must have "Date" and "Nifty_Return" columns.
        Produced by ingest.py's fetch_nifty().

    Returns
    -------
    pd.DataFrame
        Original df with three new columns:
        - "Stock_Return"     : daily % return of the stock (decimal)
        - "Abnormal_Return"  : Stock_Return minus Nifty_Return for the same day
        - "CAR_10"           : 10-day rolling sum of Abnormal_Return
    """
    df = df.copy()

    # ── Step 1: Daily stock return ────────────────────────────────────────────
    # pct_change() = (today - yesterday) / yesterday
    # First row is NaN because there's no "yesterday" — that's correct.
    df["Stock_Return"] = df["Close"].pct_change().round(6)

    # ── Step 2: Merge Nifty returns onto the stock DataFrame by date ──────────
    # We need Nifty_Return on the same date as each stock row.
    # merge with how="left" keeps all stock rows even if Nifty had no trading
    # that day (unlikely but safe).
    df = df.merge(
        nifty_df[["Date", "Nifty_Return"]],
        on="Date",
        how="left"
    )

    # ── Step 3: Daily abnormal return ─────────────────────────────────────────
    # If Nifty_Return is NaN for a date (e.g. holiday), abnormal return = stock return
    # We fill NaN Nifty_Return with 0 so the subtraction still works.
    df["Nifty_Return"] = df["Nifty_Return"].fillna(0)
    df["Abnormal_Return"] = (df["Stock_Return"] - df["Nifty_Return"]).round(6)

    # ── Step 4: 10-day rolling cumulative abnormal return ─────────────────────
    # sum() over 10 rows = total abnormal movement over the past 2 weeks
    df["CAR_10"] = (
        df["Abnormal_Return"]
        .rolling(window=10, min_periods=1)
        .sum()
        .round(6)
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — compute_vol_spike()
# ══════════════════════════════════════════════════════════════════════════════

def compute_vol_spike(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects volatility spikes — days when a stock is swinging more
    wildly than its own historical norm.

    WHAT IS A VOLATILITY SPIKE?
        We compare:
            Short-term volatility : std dev of Close returns over last 5 days
            Long-term volatility  : std dev of Close returns over last 60 days

        If short-term std dev > 2× long-term std dev → spike = 1 (flagged)

        Formula:
            Vol_Short(t) = std(Stock_Return, last 5 days)
            Vol_Long(t)  = std(Stock_Return, last 60 days)
            Vol_Ratio(t) = Vol_Short(t) / Vol_Long(t)
            Vol_Spike(t) = 1  if Vol_Ratio(t) > VOL_SPIKE_RATIO (2.0)
                         = 0  otherwise

    WHY THIS MATTERS:
        Insiders trading large positions cause erratic intraday and
        daily price movements. A sudden surge in volatility — especially
        when it wasn't there the week before — is a red flag.

    Parameters
    ----------
    df : pd.DataFrame
        Must have "Stock_Return" column.
        Call compute_car() first (it adds Stock_Return).

    Returns
    -------
    pd.DataFrame
        Three new columns added:
        - "Vol_Short"  : 5-day rolling std dev of returns
        - "Vol_Long"   : 60-day rolling std dev of returns
        - "Vol_Ratio"  : Vol_Short / Vol_Long
        - "Vol_Spike"  : 1 if Vol_Ratio > 2.0, else 0
    """
    if "Stock_Return" not in df.columns:
        raise ValueError(
            "compute_vol_spike() requires 'Stock_Return' column. "
            "Call compute_car() before calling compute_vol_spike()."
        )

    df = df.copy()

    # std() with min_periods=2 requires at least 2 data points to compute
    # (std dev of a single value is undefined)
    df["Vol_Short"] = (
        df["Stock_Return"]
        .rolling(window=SHORT_WINDOW, min_periods=2)
        .std()
        .round(6)
    )

    df["Vol_Long"] = (
        df["Stock_Return"]
        .rolling(window=LONG_WINDOW, min_periods=2)
        .std()
        .round(6)
    )

    # Guard against division by zero:
    # If Vol_Long is 0 (stock had zero variance — impossible in practice but
    # theoretically possible in test data), set to NaN so we get NaN ratio
    # rather than inf, which would break downstream scoring.
    safe_vol_long = df["Vol_Long"].replace(0, np.nan)

    df["Vol_Ratio"] = (df["Vol_Short"] / safe_vol_long).round(4)

    # Binary spike flag: 1 = suspicious spike, 0 = normal
    df["Vol_Spike"] = (df["Vol_Ratio"] > VOL_SPIKE_RATIO).astype(int)

    # Where Vol_Ratio is NaN (insufficient history), spike = 0 not NaN
    df["Vol_Spike"] = df["Vol_Spike"].fillna(0).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — compute_return_z()
# ══════════════════════════════════════════════════════════════════════════════

def compute_return_z(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a Z-score for each day's stock return relative to its own
    60-day rolling distribution.

    WHAT IS A RETURN Z-SCORE?
        Z-score measures how many standard deviations today's return
        is away from the rolling 60-day mean return.

        Formula:
            Z(t) = (Stock_Return(t) - RollingMean(Return, 60)(t))
                   / RollingStd(Return, 60)(t)

        Z = 0   → perfectly average day
        Z = 2.5 → return is 2.5 std devs above average → flagged
        Z = -3  → return is 3 std devs below average   → flagged

    WHY THIS MATTERS:
        A large positive Z-score (stock jumping far more than usual)
        or large negative Z-score (sharp drop) combined with other
        signals makes a strong case for insider activity.

        Unlike AVR which looks at volume, Return Z-score looks at
        price — both dimensions together are far more telling than either alone.

    Parameters
    ----------
    df : pd.DataFrame
        Must have "Stock_Return" column (added by compute_car()).

    Returns
    -------
    pd.DataFrame
        Two new columns:
        - "Return_Z"     : Z-score of today's return vs 60-day rolling dist
        - "Return_Z_Flag": 1 if |Return_Z| > Z_SCORE_THRESHOLD (2.5), else 0
    """
    if "Stock_Return" not in df.columns:
        raise ValueError(
            "compute_return_z() requires 'Stock_Return' column. "
            "Call compute_car() before calling compute_return_z()."
        )

    df = df.copy()

    # 60-day rolling mean of returns
    rolling_mean = (
        df["Stock_Return"]
        .rolling(window=LONG_WINDOW, min_periods=2)
        .mean()
    )

    # 60-day rolling standard deviation of returns
    rolling_std = (
        df["Stock_Return"]
        .rolling(window=LONG_WINDOW, min_periods=2)
        .std()
    )

    # Guard: if rolling_std is 0, set to NaN to avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    # Z-score formula
    df["Return_Z"] = ((df["Stock_Return"] - rolling_mean) / rolling_std).round(4)

    # Binary flag: 1 if the absolute Z-score exceeds the threshold
    df["Return_Z_Flag"] = (df["Return_Z"].abs() > Z_SCORE_THRESHOLD).astype(int)

    # Where Z is NaN (insufficient history), flag = 0
    df["Return_Z_Flag"] = df["Return_Z_Flag"].fillna(0).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — build_features()
# ══════════════════════════════════════════════════════════════════════════════

def build_features(ticker: str, nifty_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Master pipeline function — loads one stock and runs all four
    feature computations in the correct order, returning a single
    clean DataFrame ready for the anomaly detection models.

    COLUMN ORDER IN OUTPUT:
        Date | Open | High | Low | Close | Volume
        | Volume_MA60 | AVR
        | Stock_Return | Nifty_Return | Abnormal_Return | CAR_10
        | Vol_Short | Vol_Long | Vol_Ratio | Vol_Spike
        | Return_Z | Return_Z_Flag

    Parameters
    ----------
    ticker : str
        Clean ticker name, e.g. "RELIANCE" (no .NS suffix)
    nifty_df : pd.DataFrame
        Output of load_nifty() — must have Date and Nifty_Return columns.

    Returns
    -------
    pd.DataFrame  with all 18 columns described above.
    Returns None  if the stock's CSV does not exist on disk.
    """
    # Step 1: Load raw OHLCV from disk
    df = load_stock(ticker)
    if df is None:
        return None

    # Step 2: Abnormal Volume Ratio
    # (only needs Volume — no dependency on other steps)
    df = compute_avr(df)

    # Step 3: Cumulative Abnormal Return
    # (needs Close and Nifty_Return — adds Stock_Return as a side effect)
    df = compute_car(df, nifty_df)

    # Step 4: Volatility Spike
    # (needs Stock_Return — must come after compute_car)
    df = compute_vol_spike(df)

    # Step 5: Return Z-score
    # (needs Stock_Return — must come after compute_car)
    df = compute_return_z(df)

    # Add ticker name as a column so we know which stock each row belongs to
    # when we later concatenate all stocks into one DataFrame for the models
    df.insert(0, "Ticker", ticker)

    return df


def run_feature_engineering(tickers: list = None) -> dict:
    """
    Runs build_features() for every ticker and saves results to disk.

    Parameters
    ----------
    tickers : list, optional
        List of clean ticker names. Defaults to the TICKERS list above.

    Returns
    -------
    dict   {ticker_name: featured_DataFrame}
           Only successfully processed tickers are included.
    """
    if tickers is None:
        tickers = TICKERS

    print("=" * 55)
    print("  Insider Trading Detector — Feature Engineering")
    print("=" * 55)

    # Load Nifty once and reuse for all stocks
    nifty_df = load_nifty()
    if nifty_df is None:
        print("  ❌ Cannot proceed — Nifty benchmark missing.")
        print("     Run:  python backend/ingest.py")
        return {}

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    results = {}

    for ticker in tickers:
        print(f"\n  Processing {ticker} ...")
        df = build_features(ticker, nifty_df)

        if df is None:
            print(f"    ⚠️  Skipping {ticker} — CSV not found in {RAW_DIR}/")
            continue

        # Save featured DataFrame to data/processed/
        out_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
        df.to_csv(out_path, index=False)
        print(f"    ✅ {len(df)} rows, {len(df.columns)} columns → {out_path}")

        results[ticker] = df

    print()
    print("=" * 55)
    print(f"  ✅ Feature engineering complete: {len(results)} stocks")
    print(f"  📁 Files saved in: {PROCESSED_DIR}/")
    print("=" * 55)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TESTS
# ══════════════════════════════════════════════════════════════════════════════
#
# HOW TO RUN:
#   python backend/features.py --test
#
# These tests use synthetic (hand-crafted) DataFrames — they do NOT depend
# on internet access or the data/raw/ CSV files being present.
#
# WHY SYNTHETIC DATA FOR TESTS?
#   ingest.py's tests already verified that real data downloads correctly.
#   Here we want to test pure arithmetic: given these exact input numbers,
#   do we get exactly the right output numbers?
#   Synthetic data lets us calculate the expected answer by hand and
#   assert against it precisely, with no network dependency.
# ══════════════════════════════════════════════════════════════════════════════

def _make_stock_df(n: int = 80) -> pd.DataFrame:
    """
    Creates a synthetic OHLCV DataFrame with n rows for testing.

    Volume is crafted so the last 5 rows have 3× the earlier volume
    (to produce a detectable AVR spike).
    Prices follow a gentle uptrend with a sharp jump in the last 10 rows
    (to produce a detectable CAR spike).
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")  # business days

    # Base price: gentle uptrend from 100 to 120 over n days
    close = np.linspace(100, 120, n) + np.random.normal(0, 0.3, n)

    # Sharp jump in the last 10 rows (simulates pre-announcement drift)
    close[-10:] += np.linspace(0, 8, 10)

    close  = np.round(close, 2)
    open_  = np.round(close - np.random.uniform(0.1, 0.5, n), 2)
    high   = np.round(close + np.random.uniform(0.2, 0.8, n), 2)
    low    = np.round(close - np.random.uniform(0.2, 0.8, n), 2)

    # Base volume: 100,000 shares/day
    # Last 5 rows: 300,000 shares/day  →  AVR ≈ 3.0
    volume = np.full(n, 100_000)
    volume[-5:] = 300_000

    return pd.DataFrame({
        "Date"  : dates,
        "Open"  : open_,
        "High"  : high,
        "Low"   : low,
        "Close" : close,
        "Volume": volume,
    })


def _make_nifty_df(n: int = 80) -> pd.DataFrame:
    """
    Creates a synthetic Nifty 50 benchmark DataFrame with n rows for testing.
    Returns are small random values centred around 0 (realistic daily returns).
    """
    np.random.seed(99)
    dates          = pd.date_range(start="2024-01-01", periods=n, freq="B")
    nifty_close    = np.round(np.linspace(21000, 22000, n), 2)
    nifty_return   = np.round(np.random.normal(0.0003, 0.008, n), 6)
    nifty_return[0] = 0  # first row has no prior day

    return pd.DataFrame({
        "Date"         : dates,
        "Nifty_Close"  : nifty_close,
        "Nifty_Return" : nifty_return,
    })


# ── Test classes ──────────────────────────────────────────────────────────────

class TestLoadFunctions(unittest.TestCase):
    """Tests for load_stock() and load_nifty()."""

    def test_load_stock_returns_none_for_missing_file(self):
        """
        load_stock() must return None when the CSV doesn't exist,
        not raise a FileNotFoundError that would crash the pipeline.
        """
        result = load_stock("NONEXISTENT_TICKER_XYZ")
        self.assertIsNone(result,
            "load_stock() should return None for a missing file, not raise an error")

    def test_load_nifty_returns_none_for_missing_file(self):
        """
        Same graceful None for load_nifty() when benchmark CSV is missing.

        WHY sys.modules[__name__] INSTEAD OF "import backend.features":
        This file is run directly as `python backend/features.py --test`,
        which means it executes as the __main__ module, not as the
        package module backend.features. Doing `import backend.features`
        in that situation creates a SECOND, independent copy of this file
        with its own separate RAW_DIR variable. Patching that copy's
        RAW_DIR has no effect on the load_nifty() function actually being
        called here (which belongs to __main__), so the real RAW_DIR is
        still used, the real NIFTY50_benchmark.csv is found, and the test
        fails because it gets a real DataFrame instead of None.

        sys.modules[__name__] always refers to whichever module is
        currently executing — __main__ when run directly, or
        backend.features when imported normally — so the patch always
        lands on the correct, active copy of RAW_DIR.
        """
        this_module = sys.modules[__name__]
        original = this_module.RAW_DIR
        this_module.RAW_DIR = "data/definitely_does_not_exist"
        try:
            result = load_nifty()
        finally:
            # finally guarantees RAW_DIR is restored even if the
            # assertion below fails — otherwise every test that runs
            # after this one would silently use the wrong RAW_DIR.
            this_module.RAW_DIR = original

        self.assertIsNone(result,
            "load_nifty() should return None for a missing file")


class TestComputeAVR(unittest.TestCase):
    """Tests for compute_avr()."""

    @classmethod
    def setUpClass(cls):
        cls.df_raw      = _make_stock_df(n=80)
        cls.df_featured = compute_avr(cls.df_raw)

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_avr_columns_added(self):
        """
        compute_avr() must add exactly "Volume_MA60" and "AVR" columns.
        No other columns should be added or removed.
        """
        self.assertIn("Volume_MA60", self.df_featured.columns,
            "Volume_MA60 column missing after compute_avr()")
        self.assertIn("AVR", self.df_featured.columns,
            "AVR column missing after compute_avr()")

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_avr_does_not_mutate_input(self):
        """
        compute_avr() must not modify the original DataFrame it received.
        We always .copy() inputs — this test verifies that contract.
        """
        original_cols = list(self.df_raw.columns)
        self.assertEqual(original_cols, ["Date","Open","High","Low","Close","Volume"],
            "compute_avr() mutated the input DataFrame's columns")

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_avr_row_count_unchanged(self):
        """
        compute_avr() must not add or drop any rows.
        Same number of rows in as out.
        """
        self.assertEqual(len(self.df_raw), len(self.df_featured),
            "Row count changed after compute_avr()")

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_avr_spike_detected_in_last_rows(self):
        """
        The synthetic data has 300,000 volume in the last 5 rows vs
        100,000 everywhere else. AVR in those last rows should be > 2.0.
        This is the core insider signal — high volume before an event.
        """
        last_5_avr = self.df_featured["AVR"].iloc[-5:]
        self.assertTrue(
            (last_5_avr > 2.0).all(),
            f"Expected AVR > 2.0 in last 5 rows. Got:\n{last_5_avr.values}"
        )

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_avr_normal_rows_near_one(self):
        """
        In the "normal" period (rows 30–60), volume is constant at 100,000.
        AVR should be very close to 1.0 in these rows (within ±0.05).
        """
        normal_avr = self.df_featured["AVR"].iloc[30:60]
        self.assertTrue(
            ((normal_avr > 0.95) & (normal_avr < 1.05)).all(),
            f"AVR in normal rows should be ~1.0. Got min={normal_avr.min():.3f}, max={normal_avr.max():.3f}"
        )

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_avr_no_negative_values(self):
        """
        AVR = Volume / MA60. Both are always positive,
        so AVR can never be negative.
        """
        avr_values = self.df_featured["AVR"].dropna()
        self.assertTrue(
            (avr_values >= 0).all(),
            "AVR contains negative values — this is impossible and indicates a bug"
        )

    # ── Test 6b ───────────────────────────────────────────────────────────────
    def test_avr_zero_volume_handled(self):
        """
        If Volume_MA60 is 0 (edge case), division should produce NaN
        instead of inf. Our replace(0, NaN) guard handles this.
        """
        df_zero = _make_stock_df(n=10)
        df_zero["Volume"] = 0
        result = compute_avr(df_zero)
        # AVR should be NaN (0/NaN = NaN), never inf
        self.assertFalse(
            np.isinf(result["AVR"].fillna(0)).any(),
            "AVR produced inf values when Volume_MA60 is 0"
        )


class TestComputeCAR(unittest.TestCase):
    """Tests for compute_car()."""

    @classmethod
    def setUpClass(cls):
        cls.stock_df  = _make_stock_df(n=80)
        cls.nifty_df  = _make_nifty_df(n=80)
        cls.df        = compute_car(cls.stock_df, cls.nifty_df)

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_car_columns_added(self):
        """
        compute_car() must add Stock_Return, Nifty_Return,
        Abnormal_Return, and CAR_10 columns.
        """
        for col in ["Stock_Return", "Nifty_Return", "Abnormal_Return", "CAR_10"]:
            self.assertIn(col, self.df.columns,
                f"Expected column '{col}' not found after compute_car()")

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_stock_return_first_row_is_nan(self):
        """
        The first row's Stock_Return must be NaN — there is no prior day
        to compute a return from. pct_change() naturally gives NaN here.
        """
        self.assertTrue(
            pd.isna(self.df["Stock_Return"].iloc[0]),
            "First row of Stock_Return should be NaN (no prior close to compare)"
        )

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_stock_return_values_are_small_decimals(self):
        """
        Daily stock returns are stored as decimals (not percentages).
        1% return = 0.01, not 1.0.
        For our synthetic data, returns should all be within ±0.15 (15%).
        """
        returns = self.df["Stock_Return"].dropna()
        self.assertTrue(
            (returns.abs() < 0.15).all(),
            f"Some Stock_Return values look like percentages not decimals. "
            f"Max abs value: {returns.abs().max():.4f}"
        )

    # ── Test 10 ───────────────────────────────────────────────────────────────
    def test_abnormal_return_equals_stock_minus_nifty(self):
        """
        For a specific row where both returns are known, verify:
            Abnormal_Return = Stock_Return - Nifty_Return
        Checks the arithmetic is correct.
        """
        # Use row index 10 (well past the first NaN row)
        row = self.df.iloc[10]
        expected = round(row["Stock_Return"] - row["Nifty_Return"], 6)
        actual   = row["Abnormal_Return"]
        self.assertAlmostEqual(actual, expected, places=4,
            msg=f"Abnormal_Return arithmetic wrong at row 10. "
                f"Expected {expected}, got {actual}")

    # ── Test 11 ───────────────────────────────────────────────────────────────
    def test_car_positive_during_price_jump(self):
        """
        Our synthetic stock has a deliberate 8-point price jump in the last
        10 rows. The CAR_10 in those rows should be positive (stock
        outperforming Nifty during the suspicious period).
        """
        car_last_10 = self.df["CAR_10"].iloc[-10:]
        self.assertTrue(
            (car_last_10 > 0).any(),
            "CAR_10 should be positive during the synthetic price jump window"
        )

    # ── Test 12 ───────────────────────────────────────────────────────────────
    def test_car_row_count_unchanged(self):
        """compute_car() must not add or drop rows."""
        self.assertEqual(len(self.stock_df), len(self.df),
            "Row count changed after compute_car()")


class TestComputeVolSpike(unittest.TestCase):
    """Tests for compute_vol_spike()."""

    @classmethod
    def setUpClass(cls):
        stock_df      = _make_stock_df(n=80)
        nifty_df      = _make_nifty_df(n=80)
        df_with_car   = compute_car(stock_df, nifty_df)
        cls.df        = compute_vol_spike(df_with_car)

    # ── Test 13 ───────────────────────────────────────────────────────────────
    def test_vol_spike_columns_added(self):
        """compute_vol_spike() must add Vol_Short, Vol_Long, Vol_Ratio, Vol_Spike."""
        for col in ["Vol_Short", "Vol_Long", "Vol_Ratio", "Vol_Spike"]:
            self.assertIn(col, self.df.columns,
                f"Column '{col}' missing after compute_vol_spike()")

    # ── Test 14 ───────────────────────────────────────────────────────────────
    def test_vol_spike_is_binary(self):
        """
        Vol_Spike must contain only 0 or 1 — it is a binary flag.
        Any other value indicates a bug in the flag computation.
        """
        unique_vals = set(self.df["Vol_Spike"].unique())
        self.assertTrue(
            unique_vals.issubset({0, 1}),
            f"Vol_Spike contains non-binary values: {unique_vals}"
        )

    # ── Test 15 ───────────────────────────────────────────────────────────────
    def test_vol_spike_no_nulls(self):
        """
        Vol_Spike must have zero NaN values.
        We explicitly fillna(0) in compute_vol_spike() — this test verifies it.
        """
        null_count = self.df["Vol_Spike"].isnull().sum()
        self.assertEqual(null_count, 0,
            f"Vol_Spike has {null_count} NaN values — fillna(0) not working")

    # ── Test 16 ───────────────────────────────────────────────────────────────
    def test_missing_stock_return_raises(self):
        """
        compute_vol_spike() must raise a ValueError with a helpful message
        if called on a DataFrame that doesn't have Stock_Return yet.
        """
        bare_df = _make_stock_df(n=30)
        with self.assertRaises(ValueError) as ctx:
            compute_vol_spike(bare_df)
        self.assertIn("Stock_Return", str(ctx.exception),
            "ValueError message should mention the missing column name")


class TestComputeReturnZ(unittest.TestCase):
    """Tests for compute_return_z()."""

    @classmethod
    def setUpClass(cls):
        stock_df    = _make_stock_df(n=80)
        nifty_df    = _make_nifty_df(n=80)
        df_with_car = compute_car(stock_df, nifty_df)
        cls.df      = compute_return_z(df_with_car)

    # ── Test 17 ───────────────────────────────────────────────────────────────
    def test_return_z_columns_added(self):
        """compute_return_z() must add Return_Z and Return_Z_Flag."""
        self.assertIn("Return_Z", self.df.columns,
            "Return_Z column missing after compute_return_z()")
        self.assertIn("Return_Z_Flag", self.df.columns,
            "Return_Z_Flag column missing after compute_return_z()")

    # ── Test 18 ───────────────────────────────────────────────────────────────
    def test_return_z_flag_is_binary(self):
        """Return_Z_Flag must contain only 0 or 1."""
        unique_vals = set(self.df["Return_Z_Flag"].unique())
        self.assertTrue(
            unique_vals.issubset({0, 1}),
            f"Return_Z_Flag contains non-binary values: {unique_vals}"
        )

    # ── Test 19 ───────────────────────────────────────────────────────────────
    def test_return_z_flag_no_nulls(self):
        """Return_Z_Flag must have zero NaN values after fillna(0)."""
        null_count = self.df["Return_Z_Flag"].isnull().sum()
        self.assertEqual(null_count, 0,
            f"Return_Z_Flag has {null_count} NaN values")

    # ── Test 20 ───────────────────────────────────────────────────────────────
    def test_missing_stock_return_raises(self):
        """
        compute_return_z() must raise ValueError if Stock_Return column
        is missing — same guard as compute_vol_spike().
        """
        bare_df = _make_stock_df(n=30)
        with self.assertRaises(ValueError) as ctx:
            compute_return_z(bare_df)
        self.assertIn("Stock_Return", str(ctx.exception))


class TestBuildFeatures(unittest.TestCase):
    """Tests for build_features() — the master pipeline function."""

    @classmethod
    def setUpClass(cls):
        """
        Write a synthetic stock CSV to disk so build_features() can load it.
        This simulates what ingest.py produces without needing internet access.
        """
        os.makedirs(RAW_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        # Write synthetic stock CSV
        stock_df = _make_stock_df(n=80)
        stock_df["Date"] = stock_df["Date"].dt.strftime("%Y-%m-%d")
        stock_df.to_csv(os.path.join(RAW_DIR, "TEST_TICKER_ohlcv.csv"), index=False)

        # Write synthetic Nifty CSV
        nifty_df = _make_nifty_df(n=80)
        nifty_df["Date"] = nifty_df["Date"].dt.strftime("%Y-%m-%d")
        nifty_df.to_csv(os.path.join(RAW_DIR, "NIFTY50_benchmark.csv"), index=False)

        cls.nifty_df = load_nifty()
        cls.result   = build_features("TEST_TICKER", cls.nifty_df)

    # ── Test 21 ───────────────────────────────────────────────────────────────
    def test_build_features_returns_dataframe(self):
        """build_features() must return a DataFrame for a valid ticker."""
        self.assertIsNotNone(self.result,
            "build_features() returned None for TEST_TICKER")
        self.assertIsInstance(self.result, pd.DataFrame,
            "build_features() did not return a DataFrame")

    # ── Test 22 ───────────────────────────────────────────────────────────────
    def test_build_features_has_all_signal_columns(self):
        """
        The output must contain all 4 signal columns:
        AVR, CAR_10, Vol_Spike, Return_Z_Flag.
        These are what the anomaly models in models/ will use.
        """
        required = ["AVR", "CAR_10", "Vol_Spike", "Return_Z_Flag"]
        for col in required:
            self.assertIn(col, self.result.columns,
                f"Signal column '{col}' missing from build_features() output")

    # ── Test 23 ───────────────────────────────────────────────────────────────
    def test_build_features_has_ticker_column(self):
        """
        build_features() adds a "Ticker" column so rows are identifiable
        when multiple stocks are concatenated for model training.
        """
        self.assertIn("Ticker", self.result.columns,
            "Ticker column missing from build_features() output")
        self.assertTrue(
            (self.result["Ticker"] == "TEST_TICKER").all(),
            "Ticker column values do not match the requested ticker"
        )

    # ── Test 24 ───────────────────────────────────────────────────────────────
    def test_build_features_returns_none_for_missing_ticker(self):
        """
        build_features() must return None (not crash) for a ticker
        whose CSV doesn't exist on disk.
        """
        result = build_features("TICKER_THAT_DOES_NOT_EXIST", self.nifty_df)
        self.assertIsNone(result,
            "build_features() should return None for a missing ticker CSV")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
#   python backend/features.py           → runs feature engineering for all stocks
#   python backend/features.py --test    → runs all 24 tests
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if "--test" in sys.argv:
        sys.argv.remove("--test")

        print("=" * 55)
        print("  Running features.py test suite  (24 tests)")
        print("=" * 55)
        print()

        unittest.main(verbosity=2)

    else:
        run_feature_engineering()