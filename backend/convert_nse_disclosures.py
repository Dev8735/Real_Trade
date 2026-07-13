# backend/convert_nse_disclosures.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Convert a RAW insider trading disclosure CSV downloaded manually
#           from NSE's website (or BSE's equivalent page) into the clean
#           3-column format that dbscan_cluster.py expects:
#               Date, Price, Entity
#
# WHY THIS FILE EXISTS:
#   NSE/BSE's insider disclosure CSV export has 12-15 columns with long,
#   inconsistent header names (e.g. "NAME OF THE ACQUIRER/DISPOSER",
#   "VALUE OF SECURITY (ACQUIRED/DISPLOSED)") and reports VALUE and QUANTITY
#   separately rather than a per-share PRICE. dbscan_cluster.py needs a
#   derived price-per-share to cluster trades meaningfully in price-space.
#
# HOW TO GET THE RAW INPUT FILE:
#   1. Go to https://www.nseindia.com/companies-listing/corporate-filings-insider-trading
#   2. Search for your ticker (e.g. "RELIANCE") in the Company box
#   3. Set a date range (6M to match ingest.py's default window)
#   4. Click "Download (.csv)"
#   5. Save the file as data/bse_disclosures/{TICKER}_raw_nse_export.csv
#
# HOW TO RUN:
#   python backend/convert_nse_disclosures.py RELIANCE
#   python backend/convert_nse_disclosures.py --test
#
# OUTPUT:
#   Writes data/bse_disclosures/{TICKER}_trades.csv — exactly the format
#   dbscan_cluster.py's run_clustering_on_all() expects to find.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import unittest

import pandas as pd

DISCLOSURES_DIR = "data/bse_disclosures"

# Column name candidates — NSE has changed its exact header wording across
# different export versions over the years, so we check several known
# variants for each logical field rather than hardcoding one exact name.
ENTITY_COLUMN_CANDIDATES = [
    "NAME OF THE ACQUIRER/DISPOSER",
    "Name of the Acquirer/Disposer",
    "NAME OF ACQUIRER/DISPOSER",
]
QUANTITY_COLUMN_CANDIDATES = [
    "NO. OF SECURITIES (ACQUIRED/DISPLOSED)",
    "NO. OF SECURITY (ACQUIRED/DISPLOSED)",
    "No. of Securities (Acquired/Disposed)",
]
VALUE_COLUMN_CANDIDATES = [
    "VALUE OF SECURITY (ACQUIRED/DISPLOSED)",
    "Value of Security (Acquired/Disposed)",
]
DATE_COLUMN_CANDIDATES = [
    "DATE OF ALLOTMENT/ACQUISITION FROM",
    "Date of Allotment/Acquisition From",
    "DATE OF INITMATION TO COMPANY",
]


def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    """
    Returns the first column name from `candidates` that actually exists
    in df.columns, or None if none of them match.

    WHY THIS HELPER EXISTS:
        Real-world exported CSVs from government/exchange websites are
        notoriously inconsistent in header casing and exact wording across
        different download sessions or years. Rather than crashing on the
        first mismatch, we check a list of known variants and use whichever
        one is actually present.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def convert_nse_export(ticker: str) -> pd.DataFrame | None:
    """
    Reads data/bse_disclosures/{ticker}_raw_nse_export.csv and converts it
    to the clean Date/Price/Entity format dbscan_cluster.py requires.

    Parameters
    ----------
    ticker : str
        Clean ticker name, e.g. "RELIANCE". Expects the raw file at:
        data/bse_disclosures/{ticker}_raw_nse_export.csv

    Returns
    -------
    pd.DataFrame with columns [Date, Price, Entity], or None if the raw
    file doesn't exist or is missing required columns.

    PRICE DERIVATION:
        NSE reports total transaction VALUE and QUANTITY separately, not a
        per-share price directly. We derive it:
            Price = Value / Quantity
        This is the actual average price per share for that disclosed trade.
    """
    raw_path = os.path.join(DISCLOSURES_DIR, f"{ticker}_raw_nse_export.csv")

    if not os.path.exists(raw_path):
        print(f"  Error: {raw_path} not found. Download it manually from NSE first.")
        return None

    raw_df = pd.read_csv(raw_path)

    entity_col = _find_column(raw_df, ENTITY_COLUMN_CANDIDATES)
    qty_col = _find_column(raw_df, QUANTITY_COLUMN_CANDIDATES)
    value_col = _find_column(raw_df, VALUE_COLUMN_CANDIDATES)
    date_col = _find_column(raw_df, DATE_COLUMN_CANDIDATES)

    missing = []
    if entity_col is None:
        missing.append("Entity/Acquirer name column")
    if qty_col is None:
        missing.append("Quantity column")
    if value_col is None:
        missing.append("Value column")
    if date_col is None:
        missing.append("Date column")

    if missing:
        print(f"  Error: could not find these required columns in the raw "
              f"export: {missing}")
        print(f"  Available columns were: {list(raw_df.columns)}")
        return None

    clean_df = pd.DataFrame()
    clean_df["Entity"] = raw_df[entity_col].astype(str).str.strip()

    # Derive per-share price: total value / quantity. Guard against
    # division by zero for any malformed rows (quantity == 0).
    quantity = raw_df[qty_col].astype(float)
    value = raw_df[value_col].astype(float)
    clean_df["Price"] = (value / quantity.replace(0, pd.NA)).round(2)

    # NSE dates are typically in DD-Mon-YYYY format (e.g. "10-Mar-2025").
    # dayfirst handling via format inference; falls back gracefully if the
    # format differs slightly between export versions.
    clean_df["Date"] = pd.to_datetime(
        raw_df[date_col], dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Drop any row where conversion failed (bad date, zero quantity, etc.)
    # rather than silently keeping malformed rows that would break DBSCAN.
    before = len(clean_df)
    clean_df = clean_df.dropna(subset=["Date", "Price", "Entity"])
    dropped = before - len(clean_df)
    if dropped > 0:
        print(f"  Note: dropped {dropped} row(s) with unparseable date/price/entity.")

    clean_df = clean_df[["Date", "Price", "Entity"]].reset_index(drop=True)

    return clean_df


def run_conversion(ticker: str) -> bool:
    """
    Runs convert_nse_export() for one ticker and saves the result to
    data/bse_disclosures/{ticker}_trades.csv — the exact filename
    dbscan_cluster.py's run_clustering_on_all() looks for.

    Returns
    -------
    bool : True if conversion succeeded and file was written, False otherwise.
    """
    print(f"Converting {ticker} insider disclosure export...")

    clean_df = convert_nse_export(ticker)
    if clean_df is None:
        return False

    if clean_df.empty:
        print(f"  Warning: conversion produced 0 valid rows for {ticker}. "
              f"Check the raw export file content.")
        return False

    out_path = os.path.join(DISCLOSURES_DIR, f"{ticker}_trades.csv")
    os.makedirs(DISCLOSURES_DIR, exist_ok=True)
    clean_df.to_csv(out_path, index=False)

    print(f"  Done: {len(clean_df)} trade(s) converted -> {out_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_nse_export(ticker: str) -> str:
    """Writes a realistic fake raw NSE export CSV for testing, returns its path."""
    df = pd.DataFrame({
        "SYMBOL": [ticker] * 4,
        "NAME OF THE ACQUIRER/DISPOSER": [
            "Test Person A", "Test Person B", "Test Person A", "Test Person C"
        ],
        "CATEGORY OF PERSON": ["Promoter", "Promoter", "Promoter", "Director"],
        "NO. OF SECURITIES (ACQUIRED/DISPLOSED)": [5000, 3000, 2000, 1000],
        "VALUE OF SECURITY (ACQUIRED/DISPLOSED)": [12500000, 7500000, 5100000, 2600000],
        "ACQUISITION/DISPOSAL TRANSACTION TYPE": ["Buy", "Buy", "Sell", "Buy"],
        "DATE OF ALLOTMENT/ACQUISITION FROM": [
            "10-Mar-2025", "11-Mar-2025", "20-Apr-2025", "21-Apr-2025"
        ],
    })

    os.makedirs(DISCLOSURES_DIR, exist_ok=True)
    path = os.path.join(DISCLOSURES_DIR, f"{ticker}_raw_nse_export.csv")
    df.to_csv(path, index=False)
    return path


class TestConvertNseExport(unittest.TestCase):
    """Tests for convert_nse_export() and run_conversion()."""

    @classmethod
    def setUpClass(cls):
        cls.test_ticker = "_TESTSTOCK_NSECONVERT"
        cls.raw_path = _make_fake_nse_export(cls.test_ticker)

    @classmethod
    def tearDownClass(cls):
        for suffix in ["_raw_nse_export.csv", "_trades.csv"]:
            path = os.path.join(DISCLOSURES_DIR, f"{cls.test_ticker}{suffix}")
            if os.path.exists(path):
                os.remove(path)

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_returns_correct_columns(self):
        """Output must have exactly Date, Price, Entity columns."""
        result = convert_nse_export(self.test_ticker)
        self.assertIsNotNone(result)
        self.assertEqual(list(result.columns), ["Date", "Price", "Entity"])

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_correct_row_count(self):
        """All 4 fake rows should convert successfully."""
        result = convert_nse_export(self.test_ticker)
        self.assertEqual(len(result), 4)

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_price_correctly_derived_from_value_and_quantity(self):
        """
        First row: value=12500000, quantity=5000 -> price=2500.00
        This is the core arithmetic check - proves price derivation is correct.
        """
        result = convert_nse_export(self.test_ticker)
        first_row_price = result.iloc[0]["Price"]
        self.assertAlmostEqual(first_row_price, 2500.00, places=2)

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_date_format_is_yyyy_mm_dd(self):
        """Dates must be converted to YYYY-MM-DD string format."""
        result = convert_nse_export(self.test_ticker)
        first_date = result.iloc[0]["Date"]
        self.assertEqual(first_date, "2025-03-10")

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_entity_names_preserved(self):
        """Entity column must contain the acquirer/disposer names, unmodified."""
        result = convert_nse_export(self.test_ticker)
        entities = set(result["Entity"])
        self.assertIn("Test Person A", entities)
        self.assertIn("Test Person B", entities)
        self.assertIn("Test Person C", entities)

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_missing_raw_file_returns_none(self):
        """A ticker with no raw export file must return None, not crash."""
        result = convert_nse_export("_NONEXISTENT_TICKER_XYZ")
        self.assertIsNone(result)

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_missing_required_columns_returns_none(self):
        """
        A raw CSV missing one of the required source columns (e.g. no
        entity name column at all) must return None with a clear error,
        not crash with a KeyError.
        """
        broken_ticker = "_TESTSTOCK_BROKEN_NSE"
        broken_path = os.path.join(DISCLOSURES_DIR, f"{broken_ticker}_raw_nse_export.csv")
        pd.DataFrame({"SYMBOL": ["X"], "SOME_OTHER_COL": [1]}).to_csv(broken_path, index=False)

        try:
            result = convert_nse_export(broken_ticker)
            self.assertIsNone(result)
        finally:
            if os.path.exists(broken_path):
                os.remove(broken_path)

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_run_conversion_writes_output_file(self):
        """run_conversion() must write a {ticker}_trades.csv file to disk."""
        success = run_conversion(self.test_ticker)
        self.assertTrue(success)

        out_path = os.path.join(DISCLOSURES_DIR, f"{self.test_ticker}_trades.csv")
        self.assertTrue(os.path.exists(out_path))

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_output_file_is_directly_loadable_by_dbscan_format(self):
        """
        The saved {ticker}_trades.csv must, when re-read, have exactly the
        TRADE_COLUMNS that dbscan_cluster.py requires - proving the two
        files are actually compatible end-to-end, not just superficially similar.
        """
        run_conversion(self.test_ticker)
        out_path = os.path.join(DISCLOSURES_DIR, f"{self.test_ticker}_trades.csv")
        reloaded = pd.read_csv(out_path)

        for required_col in ["Date", "Price", "Entity"]:
            self.assertIn(required_col, reloaded.columns)


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        print("=" * 55)
        print("  Running convert_nse_disclosures.py test suite  (9 tests)")
        print("=" * 55)
        print()
        unittest.main(verbosity=2)

    elif len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        run_conversion(ticker)

    else:
        print("Usage:")
        print("  python backend/convert_nse_disclosures.py TICKER_NAME")
        print("  python backend/convert_nse_disclosures.py --test")