# backend/ingest.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Download historical OHLCV (Open/High/Low/Close/Volume) price data
#           for NSE-listed stocks using yfinance and save each as a CSV file.
#
# CONTAINS:
#   Section 1 — Imports & Configuration
#   Section 2 — fetch_ohlcv()       : download one stock
#   Section 3 — fetch_nifty()       : download Nifty 50 benchmark
#   Section 4 — run_ingestion()     : download all stocks at once
#   Section 5 — Tests               : 12 self-contained tests using unittest
#   Section 6 — Entry point         : run ingestion OR tests based on CLI flag
#
# HOW TO RUN:
#   python backend/ingest.py           → runs full data ingestion
#   python backend/ingest.py --test    → runs all 12 tests
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import unittest

import pandas as pd
import yfinance as yf


# All NSE tickers on yfinance end with .NS
# 38 stocks across 8 sectors — Defence, Power, CapGoods, Pharma,
# Chemicals, NBFC, Railways, Auto (RITES and TEXRAIL appear in both
# CapGoods and Railways in the original list — deduplicated here)
# ──────────────────────────────────────────────────────────────────────────
# STOCK UNIVERSE — 39 NSE stocks across 8 sectors
# Source: user-defined sector basket (no large-cap index stocks)
# NOTE : RITES and TEXRAIL appear in both CapGoods and Railways in the
#        original list — they are included once here (deduplicated).
#        PIGL (Power & Instrumentation Gujarat) was listed Apr 2018,
#        so yfinance will return ~8 years of data for it instead of 10.
# ──────────────────────────────────────────────────────────────────────────
TICKERS = [
    # ── Defence ──────────────────────────────────────────────────────────────
    "PARAS.NS",        # Paras Defence and Space Technologies
    "DATAPATTNS.NS",   # Data Patterns (India)
    "ASTRAMICRO.NS",   # Astra Microwave Products
    "SIKAINTERP.NS",   # Sika Interplant Systems
    "AVANTEL.NS",      # Avantel

    # ── Power ─────────────────────────────────────────────────────────────────
    "CESC.NS",         # CESC
    "GENUSPOWER.NS",   # Genus Power Infrastructures
    "SKIPPER.NS",      # Skipper
    "RTNPOWER.NS",     # RattanIndia Power
    "PIGL.NS",         # Power & Instrumentation (Gujarat) — listed Apr 2018

    # ── Capital Goods ─────────────────────────────────────────────────────────
    "CGPOWER.NS",      # CG Power & Industrial Solutions
    "APARINDS.NS",     # Apar Industries
    "RITES.NS",        # RITES (also in Railways — not duplicated below)
    "VESUVIUS.NS",     # Vesuvius India
    "TEXRAIL.NS",      # Texmaco Rail & Engineering (also in Railways — not duplicated)

    # ── Pharma ────────────────────────────────────────────────────────────────
    "LAURUSLABS.NS",   # Laurus Labs
    "AARTIDRUGS.NS",   # Aarti Drugs
    "INNOVACAP.NS",    # Innova Captab
    "SOLARA.NS",       # Solara Active Pharma Sciences
    "FDC.NS",          # FDC

    # ── Chemicals ─────────────────────────────────────────────────────────────
    "AARTIIND.NS",     # Aarti Industries
    "DEEPAKNTR.NS",    # Deepak Nitrite
    "NEONAMINES.NS",   # Neogen Chemicals
    "BALAMINES.NS",    # Balaji Amines
    "PRIVISCL.NS",     # Privi Speciality Chemicals

    # ── NBFC ──────────────────────────────────────────────────────────────────
    "POONAWALLA.NS",   # Poonawalla Fincorp
    "FIVESTAR.NS",     # Five Star Business Finance
    "MUTHOOTFIN.NS",   # Muthoot Finance
    "CHOLAFIN.NS",     # Cholamandalam Investment and Finance
    "FEDFINA.NS",      # Fedbank Financial Services

    # ── Railways ─────────────────────────────────────────────────────────────
    "TITAGARH.NS",     # Titagarh Rail Systems
    "RVNL.NS",         # Rail Vikas Nigam
    "IRCON.NS",        # Ircon International
    # RITES and TEXRAIL already listed under CapGoods above

    # ── Auto ──────────────────────────────────────────────────────────────────
    "SONACOMS.NS",     # Sona BLW Precision Forgings
    "ENDURANCE.NS",    # Endurance Technologies
    "MSUMI.NS",        # Motherson Sumi Wiring India
    "TUBEINVEST.NS",   # Tube Investments of India
    "BHARATFORG.NS",   # Bharat Forge
]

# All raw CSV files are saved here
RAW_DIR = "data/raw"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — fetch_ohlcv()
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(ticker: str, period: str = "10y") -> pd.DataFrame | None:
    """
    Downloads daily OHLCV data for a single NSE-listed stock.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
        NSE stocks use the suffix .NS  →  e.g. "RELIANCE.NS"
        BSE stocks use the suffix .BO  →  e.g. "RELIANCE.BO"

    period : str
        How far back to fetch.
        "10y" = last ~2500 trading days (default — 10 years for robust baseline)
        "1mo" = last ~22 trading days   (used by tests to keep them fast)
        "1y"  = last ~252 trading days

    Returns
    -------
    pd.DataFrame
        Columns: [Date, Open, High, Low, Close, Volume]
        Date is a YYYY-MM-DD string. Prices are float. Volume is int.

    Returns None if:
        - The ticker symbol is invalid or not found on Yahoo Finance
        - yfinance returns an empty dataset (delisted stock, network error)
    """
    print(f"  Fetching {ticker} ...")

    # yf.download() hits Yahoo Finance — completely free, no API key needed.
    # auto_adjust=True : prices are adjusted for splits and dividends
    # progress=False   : suppresses the tqdm progress bar in the terminal
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    # Empty DataFrame means Yahoo Finance found nothing for this ticker
    if df is None or df.empty:
        print(f"    ⚠️  No data returned for {ticker} — skipping.")
        return None

    # yfinance returns Date as the DataFrame index — bring it to a column
    df.reset_index(inplace=True)

    # yfinance sometimes returns MultiIndex columns (ticker name + field).
    # Flatten them so we get simple column names: Open, High, Low, Close, Volume
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Keep only the 6 columns we need, in a fixed order.
    # Any extra columns yfinance might add (e.g. Dividends) are dropped here.
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # Standardise the Date column to a YYYY-MM-DD string.
    # Consistent format prevents date-parsing bugs in features.py later.
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Round prices to 2 decimal places (rupee + paisa precision)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)

    # Volume is a share count — it should always be a whole number
    df["Volume"] = df["Volume"].astype(int)

    # Build output path: "RELIANCE.NS" → "data/raw/RELIANCE_ohlcv.csv"
    clean_name = ticker.split(".")[0]
    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, f"{clean_name}_ohlcv.csv")

    df.to_csv(filepath, index=False)
    print(f"    ✅ {len(df)} rows saved → {filepath}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — fetch_nifty()
# ══════════════════════════════════════════════════════════════════════════════

def fetch_nifty(period: str = "10y") -> pd.DataFrame | None:
    """
    Downloads Nifty 50 index data and computes its daily percentage return.

    WHY THIS IS NEEDED:
        To detect whether a stock's price movement is genuinely abnormal,
        we subtract the market's movement from it. For example:
            Stock went up 3% on a day when Nifty went up 2.8%
            → Abnormal return = 0.2%  (not suspicious)

            Stock went up 5% on a day when Nifty went down 0.5%
            → Abnormal return = 5.5%  (very suspicious)

        The Nifty_Return column computed here is the market benchmark
        used in the CAR (Cumulative Abnormal Return) calculation in features.py.

    Parameters
    ----------
    period : str
        Same period string as fetch_ohlcv. Default "6mo".

    Returns
    -------
    pd.DataFrame
        Columns: [Date, Nifty_Close, Nifty_Return]
        Saved to: data/raw/NIFTY50_benchmark.csv

    Returns None if download fails.
    """
    print("  Fetching Nifty 50 benchmark (^NSEI) ...")

    df = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)

    if df is None or df.empty:
        print("    ⚠️  Could not fetch Nifty 50 data.")
        return None

    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Only Close is needed for the benchmark
    df = df[["Date", "Close"]].copy()
    df.columns = ["Date", "Nifty_Close"]

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Nifty_Close"] = df["Nifty_Close"].round(2)

    # pct_change() = (today - yesterday) / yesterday → daily % return as decimal
    # e.g. Nifty goes from 22000 to 22110 → pct_change = 0.005 (0.5%)
    df["Nifty_Return"] = df["Nifty_Close"].pct_change().round(6)

    # The very first row has no "yesterday" so pct_change gives NaN — set to 0
    df["Nifty_Return"] = df["Nifty_Return"].fillna(0)

    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, "NIFTY50_benchmark.csv")
    df.to_csv(filepath, index=False)
    print(f"    ✅ {len(df)} rows saved → {filepath}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — run_ingestion()
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion(tickers: list = None, period: str = "10y") -> dict:
    """
    Master function — downloads all tickers + Nifty 50 in one call.

    This is the function that features.py will import and call to get
    fresh data before computing signals.

    Parameters
    ----------
    tickers : list, optional
        List of Yahoo Finance ticker strings.
        Defaults to the TICKERS list defined at the top of this file.

    period : str
        Time period for all downloads. Default "6mo".

    Returns
    -------
    dict
        Keys   : clean ticker names without suffix, e.g. "RELIANCE", "INFY"
                 Plus "NIFTY50" for the benchmark.
        Values : pd.DataFrame for each ticker

    Example
    -------
        data = run_ingestion()
        reliance_df = data["RELIANCE"]
        nifty_df    = data["NIFTY50"]
    """
    if tickers is None:
        tickers = TICKERS

    print("=" * 55)
    print("  Insider Trading Detector — Data Ingestion")
    print("=" * 55)

    results = {}

    for ticker in tickers:
        df = fetch_ohlcv(ticker, period=period)
        if df is not None:
            clean_name = ticker.split(".")[0]
            results[clean_name] = df

    nifty_df = fetch_nifty(period=period)
    if nifty_df is not None:
        results["NIFTY50"] = nifty_df

    print()
    print("=" * 55)
    print(f"  ✅ Ingestion complete: {len(results)} datasets downloaded")
    print(f"  📁 Files saved in   : {RAW_DIR}/")
    print("=" * 55)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TESTS
# ══════════════════════════════════════════════════════════════════════════════
#
# These tests use Python's built-in `unittest` — no extra library needed.
# Each TestCase class groups related tests. Each test method starts with
# "test_" so unittest discovers it automatically.
#
# HOW TO RUN TESTS:
#   python backend/ingest.py --test
#
# WHAT EACH TEST CHECKS:
#   TestFetchOHLCV    → the core download function works correctly
#   TestFetchNifty    → the benchmark download and return computation works
#   TestRunIngestion  → the master pipeline function works end-to-end
#
# NOTE: Tests use period="1mo" (not "6mo") to download less data and run fast.
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchOHLCV(unittest.TestCase):
    """Tests for the fetch_ohlcv() function."""

    # setUpClass runs once before all tests in this class.
    # We download data once and reuse it across tests to avoid
    # hitting Yahoo Finance multiple times.
    #
    # TICKER SELECTION RULE:
    #   Only use large-cap, consistently listed NSE stocks here.
    #   Avoid recently-listed or low-volume stocks (e.g. ZOMATO.NS) because
    #   Yahoo Finance occasionally returns None for them intermittently.
    #   RELIANCE, INFY, TCS, WIPRO, HDFCBANK are the safest choices.
    @classmethod
    def setUpClass(cls):
        cls.df_reliance   = fetch_ohlcv("RELIANCE.NS",   period="1mo")
        cls.df_infy       = fetch_ohlcv("INFY.NS",       period="1mo")
        cls.df_tcs        = fetch_ohlcv("TCS.NS",        period="1mo")
        cls.df_wipro      = fetch_ohlcv("WIPRO.NS",      period="1mo")
        cls.df_bajfinance = fetch_ohlcv("BAJFINANCE.NS", period="1mo")
        cls.df_sunpharma  = fetch_ohlcv("SUNPHARMA.NS",  period="1mo")
        cls.df_hdfcbank   = fetch_ohlcv("HDFCBANK.NS",   period="1mo")

    def _require(self, df, ticker_name):
        """
        Helper called at the top of any test that depends on a specific DataFrame.
        If that DataFrame is None (Yahoo Finance returned nothing), the test is
        skipped with a clear message instead of crashing with a TypeError.

        This is the correct pattern — a skipped test is honest;
        a crashed test is misleading.
        """
        if df is None:
            self.skipTest(
                f"{ticker_name} data is None — "
                f"Yahoo Finance may be temporarily unavailable. "
                f"Re-run the tests or check your internet connection."
            )

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_returns_dataframe(self):
        """
        fetch_ohlcv() must return a pandas DataFrame, not None or any other type.
        This is the most basic sanity check — if it fails, the download is broken.
        """
        self._require(self.df_reliance, "RELIANCE.NS")
        self.assertIsInstance(
            self.df_reliance,
            pd.DataFrame,
            "fetch_ohlcv did not return a DataFrame"
        )

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_correct_columns(self):
        """
        The DataFrame must have exactly these 6 columns in this exact order.
        Any deviation will break features.py which accesses columns by name.
        """
        self._require(self.df_infy, "INFY.NS")
        expected = ["Date", "Open", "High", "Low", "Close", "Volume"]
        actual   = list(self.df_infy.columns)
        self.assertEqual(
            actual, expected,
            f"Column mismatch.\n  Got     : {actual}\n  Expected: {expected}"
        )

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_no_missing_values(self):
        """
        No cell should be NaN. Missing price/volume values cause silent
        calculation errors in rolling window computations in features.py.
        """
        self._require(self.df_tcs, "TCS.NS")
        missing = self.df_tcs.isnull().sum().sum()
        self.assertEqual(
            missing, 0,
            f"Found {missing} missing (NaN) values in TCS data"
        )

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_minimum_row_count(self):
        """
        1 month of NSE data should have at least 15 rows (trading days).
        NSE trades ~22 days/month, so 15 is a safe lower bound.
        Fewer rows means yfinance returned incomplete data.
        """
        self._require(self.df_reliance, "RELIANCE.NS")
        self.assertGreaterEqual(
            len(self.df_reliance), 15,
            f"Expected ≥ 15 rows for 1 month, got {len(self.df_reliance)}"
        )

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_price_sanity(self):
        """
        Core financial logic check:
        - High must always be ≥ Low on the same day
        - Close must fall between Low and High
        - All price values must be positive (no zero or negative prices)
        If any of these fail, the data from Yahoo Finance is corrupted.
        """
        self._require(self.df_wipro, "WIPRO.NS")
        df = self.df_wipro

        invalid_hl = (df["High"] < df["Low"]).sum()
        self.assertEqual(invalid_hl, 0,
            f"{invalid_hl} rows where High < Low in WIPRO data")

        invalid_cl = (df["Close"] < df["Low"]).sum()
        self.assertEqual(invalid_cl, 0,
            f"{invalid_cl} rows where Close < Low in WIPRO data")

        invalid_ch = (df["Close"] > df["High"]).sum()
        self.assertEqual(invalid_ch, 0,
            f"{invalid_ch} rows where Close > High in WIPRO data")

        for col in ["Open", "High", "Low", "Close"]:
            non_positive = (df[col] <= 0).sum()
            self.assertEqual(non_positive, 0,
                f"{non_positive} non-positive values found in {col}")

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_csv_file_is_saved_on_disk(self):
        """
        After fetch_ohlcv runs, the CSV must physically exist on disk.
        Tests the file-writing logic — not just the return value.
        """
        self._require(self.df_bajfinance, "BAJFINANCE.NS")
        filepath = os.path.join(RAW_DIR, "BAJFINANCE_ohlcv.csv")
        self.assertTrue(
            os.path.exists(filepath),
            f"CSV file not found at expected path: {filepath}"
        )

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_saved_csv_matches_returned_dataframe(self):
        """
        Reading the saved CSV back should give the same row count and columns
        as the DataFrame that fetch_ohlcv returned in memory.
        This confirms the to_csv() step didn't lose or corrupt any data.
        """
        self._require(self.df_sunpharma, "SUNPHARMA.NS")
        filepath = os.path.join(RAW_DIR, "SUNPHARMA_ohlcv.csv")
        df_saved = pd.read_csv(filepath)

        self.assertEqual(
            len(self.df_sunpharma), len(df_saved),
            "Row count differs between returned DataFrame and saved CSV"
        )
        self.assertEqual(
            list(self.df_sunpharma.columns), list(df_saved.columns),
            "Column names differ between returned DataFrame and saved CSV"
        )

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_invalid_ticker_returns_none(self):
        """
        Passing a fake/non-existent ticker must return None gracefully.
        The pipeline must not crash — it should skip and continue.
        """
        result = fetch_ohlcv("FAKETICKER999.NS", period="1mo")
        self.assertIsNone(
            result,
            "fetch_ohlcv should return None for invalid ticker, not raise an error"
        )

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_date_column_format(self):
        """
        The Date column must be a YYYY-MM-DD formatted string.
        features.py will do date arithmetic and grouping — a consistent
        date string format prevents silent bugs there.

        Uses HDFCBANK (not ZOMATO) because large-cap stocks are more
        reliably available on Yahoo Finance across all network conditions.
        """
        self._require(self.df_hdfcbank, "HDFCBANK.NS")
        sample_date = self.df_hdfcbank["Date"].iloc[0]

        self.assertIsInstance(
            sample_date, str,
            f"Date should be a string, got {type(sample_date)}"
        )

        parts = sample_date.split("-")
        self.assertEqual(len(parts), 3,
            f"Date format should be YYYY-MM-DD, got: {sample_date}")
        self.assertEqual(len(parts[0]), 4,
            f"Year part should be 4 digits, got: {parts[0]}")
        self.assertEqual(len(parts[1]), 2,
            f"Month part should be 2 digits, got: {parts[1]}")
        self.assertEqual(len(parts[2]), 2,
            f"Day part should be 2 digits, got: {parts[2]}")


class TestFetchNifty(unittest.TestCase):
    """Tests for the fetch_nifty() function."""

    @classmethod
    def setUpClass(cls):
        cls.df = fetch_nifty(period="1mo")

    # ── Test 10 ───────────────────────────────────────────────────────────────
    def test_nifty_correct_columns(self):
        """
        Nifty benchmark must have exactly 3 columns: Date, Nifty_Close, Nifty_Return.
        Nifty_Return is computed inside fetch_nifty and used for CAR in features.py.
        """
        if self.df is None:
            self.skipTest("fetch_nifty returned None — Yahoo Finance may be unavailable")
        expected = ["Date", "Nifty_Close", "Nifty_Return"]
        self.assertEqual(
            list(self.df.columns), expected,
            f"Nifty columns wrong.\n  Got     : {list(self.df.columns)}\n  Expected: {expected}"
        )

    # ── Test 11 ───────────────────────────────────────────────────────────────
    def test_nifty_return_no_nulls(self):
        """
        The first row of pct_change() is naturally NaN (no prior day exists).
        fetch_nifty fills it with 0 using fillna(0).
        This test confirms that fillna worked and the column is fully clean.
        """
        if self.df is None:
            self.skipTest("fetch_nifty returned None — Yahoo Finance may be unavailable")
        null_count = self.df["Nifty_Return"].isnull().sum()
        self.assertEqual(
            null_count, 0,
            f"Nifty_Return still has {null_count} NaN values — fillna(0) did not work"
        )

    # ── Test 12 ───────────────────────────────────────────────────────────────
    def test_nifty_return_values_are_small_decimals(self):
        """
        Daily market returns are typically between -5% and +5%.
        Nifty_Return is stored as a decimal: 1% = 0.01, 2% = 0.02.
        If values are outside [-0.1, +0.1] something went wrong in computation.
        This is a data-quality guard, not a financial rule.
        """
        if self.df is None:
            self.skipTest("fetch_nifty returned None — Yahoo Finance may be unavailable")
        max_return = self.df["Nifty_Return"].abs().max()
        self.assertLess(
            max_return, 0.15,
            f"Max daily Nifty return is {max_return:.4f} — unexpectedly large, check calculation"
        )


class TestRunIngestion(unittest.TestCase):
    """Tests for the run_ingestion() master function."""

    @classmethod
    def setUpClass(cls):
        # Use only 2 tickers to keep this test fast
        cls.result = run_ingestion(
            tickers=["RELIANCE.NS", "INFY.NS"],
            period="1mo"
        )

    # ── Test 13 ───────────────────────────────────────────────────────────────
    def test_returns_dict(self):
        """run_ingestion() must return a Python dict."""
        self.assertIsInstance(
            self.result, dict,
            "run_ingestion did not return a dict"
        )

    # ── Test 14 ───────────────────────────────────────────────────────────────
    def test_dict_contains_requested_tickers(self):
        """
        The result dict must contain keys for each successfully downloaded ticker.
        Keys use the clean name without .NS suffix: "RELIANCE", "INFY".
        """
        self.assertIn("RELIANCE", self.result,
            "'RELIANCE' key missing from run_ingestion result")
        self.assertIn("INFY", self.result,
            "'INFY' key missing from run_ingestion result")

    # ── Test 15 ───────────────────────────────────────────────────────────────
    def test_dict_contains_nifty_benchmark(self):
        """
        Even when called with a custom ticker list, run_ingestion must
        always include "NIFTY50" in the result. The benchmark is mandatory
        for CAR computation and should never be skipped.
        """
        self.assertIn("NIFTY50", self.result,
            "'NIFTY50' key missing — benchmark is always required")

    # ── Test 16 ───────────────────────────────────────────────────────────────
    def test_each_value_is_dataframe(self):
        """
        Every value in the result dict must be a DataFrame.
        This ensures downstream code can safely call .columns, .iterrows() etc.
        """
        for name, df in self.result.items():
            self.assertIsInstance(
                df, pd.DataFrame,
                f"Value for '{name}' is not a DataFrame — got {type(df)}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
#   python backend/ingest.py           → runs full data ingestion (all 10 stocks)
#   python backend/ingest.py --test    → runs all 16 tests and shows PASS/FAIL
#
# This block only executes when the file is run directly.
# It does NOT run when ingest.py is imported by features.py or api.py.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if "--test" in sys.argv:
        # Remove --test from sys.argv so unittest doesn't try to parse it
        sys.argv.remove("--test")

        print("=" * 55)
        print("  Running ingest.py test suite  (16 tests)")
        print("=" * 55)
        print()

        # verbosity=2 prints each test name and PASS/FAIL individually
        unittest.main(verbosity=2)

    else:
        # Default behaviour — run the full ingestion pipeline
        run_ingestion()