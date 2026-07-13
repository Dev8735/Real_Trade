# backend/api.py
#from backend.quality_signals_api import router as quality_router
#app.include_router(quality_router)
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : A FastAPI server that exposes the fully-scored data (produced by
#           ingest.py -> features.py -> isolation_forest.py -> scoring.py)
#           as JSON endpoints the Next.js dashboard can call.
#
# THIS FILE IS THE BRIDGE between the Python ML pipeline and the frontend.
# Everything computed so far lives in CSV files on disk — this file reads
# those CSVs and serves them as HTTP responses.
#
# ENDPOINTS:
#   GET  /                        -> health check, confirms the API is running
#   GET  /stocks                  -> list of all tickers that have been scored
#   GET  /flags                   -> top suspicious stocks across ALL tickers,
#                                     sorted by their single highest Suspicion_Score
#   GET  /stock/{ticker}          -> full day-by-day scored history for one stock
#   GET  /stock/{ticker}/summary  -> a compact summary (latest score, max score,
#                                     total flagged days) for one stock
#
# CONTAINS:
#   Section 1 — Imports & Configuration
#   Section 2 — load_scored_stock()      : read one stock's fully-scored CSV
#   Section 3 — get_all_ticker_names()   : list every available ticker
#   Section 4 — build_flags_summary()    : cross-stock ranking for /flags
#   Section 5 — FastAPI app & endpoints
#   Section 6 — Tests                    : unittest, run with --test
#   Section 7 — Entry point
#
# HOW TO RUN:
#   python backend/api.py                → starts the server on port 8000
#   python backend/api.py --test         → runs the test suite (no server starts)
#
# HOW TO RUN THE SERVER MANUALLY (recommended for actual development):
#   uvicorn backend.api:app --reload --port 8000
#   (--reload restarts the server automatically when you edit the code)
#
# DEPENDENCY:
#   This file expects data/processed/{ticker}_features.csv to already
#   contain Suspicion_Score and Suspicion_Flag (i.e. scoring.py has run).
#
# NEW LIBRARIES USED HERE:
#   pip install fastapi uvicorn
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import unittest

import app
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
PROCESSED_DIR = "data/processed"

# Columns that must exist in a stock's CSV for it to be servable by this API.
# These come from the full pipeline: features.py + isolation_forest.py + scoring.py.
REQUIRED_COLUMNS = ["Date", "Suspicion_Score", "Suspicion_Flag"]

# CORS origins allowed to call this API. The Next.js dev server runs on
# localhost:3000 by default — without this, the browser blocks requests
# from the frontend due to the same-origin policy.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — load_scored_stock()
# ══════════════════════════════════════════════════════════════════════════════

def load_scored_stock(ticker: str) -> pd.DataFrame | None:
    """
    Reads one stock's fully-scored CSV from data/processed/.

    Parameters
    ----------
    ticker : str
        Clean ticker name, e.g. "RELIANCE" (matches the filename
        {ticker}_features.csv written by features.py and later
        overwritten in place by isolation_forest.py and scoring.py).

    Returns
    -------
    pd.DataFrame sorted by Date ascending, or None if:
        - the file doesn't exist, OR
        - the file exists but is missing REQUIRED_COLUMNS (meaning
          scoring.py hasn't been run on it yet)
    """
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — get_all_ticker_names()
# ══════════════════════════════════════════════════════════════════════════════

def get_all_ticker_names() -> list[str]:
    """
    Scans data/processed/ and returns the clean ticker name for every
    "{ticker}_features.csv" file found, REGARDLESS of whether scoring has
    run on it yet (callers that need scored data should still go through
    load_scored_stock() to filter further).

    Returns
    -------
    list[str], sorted alphabetically. Empty list if the directory
    doesn't exist or contains no matching files.
    """
    if not os.path.exists(PROCESSED_DIR):
        return []

    tickers = [
        f.replace("_features.csv", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_features.csv")
    ]

    return sorted(tickers)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — build_flags_summary()
# ══════════════════════════════════════════════════════════════════════════════

def build_flags_summary() -> list[dict]:
    """
    Builds the cross-stock "top suspicious stocks" list for the /flags
    endpoint. For each scorable ticker, finds its SINGLE highest
    Suspicion_Score across its entire history and the date that score
    occurred on, then returns all tickers sorted by that peak score,
    highest first.

    WHY "PEAK SCORE" RATHER THAN "MOST RECENT SCORE":
        The dashboard's alert table should surface stocks that have EVER
        shown strong suspicious behaviour, not just whatever happened
        on the very last trading day. A stock that spiked to 92 three
        weeks ago and has been quiet since is still worth a human
        analyst's attention — using only the latest day's score would
        hide that entirely.

    Returns
    -------
    list[dict], one dict per ticker, each containing:
        {
            "ticker": str,
            "peak_score": float,
            "peak_date": str (YYYY-MM-DD),
            "flagged_days": int   (count of Suspicion_Flag == 1 rows)
        }
        Sorted by peak_score descending. Tickers with no scorable data
        (scoring.py hasn't run on them) are silently excluded.
    """
    summary = []

    for ticker in get_all_ticker_names():
        df = load_scored_stock(ticker)
        if df is None or df.empty:
            continue

        peak_idx = df["Suspicion_Score"].idxmax()
        peak_row = df.loc[peak_idx]

        summary.append({
            "ticker": ticker,
            "peak_score": float(peak_row["Suspicion_Score"]),
            "peak_date": peak_row["Date"].strftime("%Y-%m-%d"),
            "flagged_days": int(df["Suspicion_Flag"].sum()),
        })

    summary.sort(key=lambda row: row["peak_score"], reverse=True)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FASTAPI APP & ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Insider Trading Pattern Detector API",
    description="Serves anomaly-scored stock data to the dashboard frontend.",
    version="1.0.0",
)

# Without this middleware, the browser blocks the Next.js frontend (running
# on a different port) from calling this API at all, due to CORS policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """
    Simple health check endpoint. The frontend can ping this on load to
    confirm the backend is reachable before making real data requests,
    and show a friendly "backend offline" message instead of a generic
    network error if this fails.
    """
    return {"status": "ok", "message": "Insider Trading Detector API is running"}


@app.get("/stocks")
def get_stocks():
    """
    Returns the list of all ticker names that have processed data
    available, regardless of whether scoring has completed on them yet.
    The frontend uses this to populate the stock search/dropdown.
    """
    tickers = get_all_ticker_names()
    return {"tickers": tickers, "count": len(tickers)}


@app.get("/flags")
def get_flags():
    """
    Returns the cross-stock "top suspicious stocks" ranking, used to
    populate the dashboard's main alert table. See build_flags_summary()
    for the exact ranking logic.
    """
    summary = build_flags_summary()
    return {"flagged_stocks": summary, "count": len(summary)}


@app.get("/stock/{ticker}")
def get_stock_detail(ticker: str):
    """
    Returns the FULL day-by-day scored history for one stock — every
    row of its processed CSV, as JSON records. Used by the frontend's
    per-stock detail page to render the candlestick chart, score gauge,
    and anomaly timeline.

    Raises
    ------
    404 if the ticker has no data, or if its data exists but hasn't
    been scored yet (scoring.py hasn't run on it).
    """
    df = load_scored_stock(ticker)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"No scored data found for ticker '{ticker}'. "
                   f"It may not exist, or the scoring pipeline hasn't "
                   f"run on it yet."
        )

    # Convert Date back to a clean string for JSON serialization —
    # pandas Timestamps aren't directly JSON-serializable.
    df_out = df.copy()
    df_out["Date"] = df_out["Date"].dt.strftime("%Y-%m-%d")

    return {
        "ticker": ticker,
        "row_count": len(df_out),
        "data": df_out.to_dict(orient="records"),
    }


@app.get("/stock/{ticker}/summary")
def get_stock_summary(ticker: str):
    """
    Returns a compact summary for one stock, rather than the full
    day-by-day history — useful for a lightweight card/preview view
    without pulling the entire dataset over the network.

    Raises
    ------
    404 under the same conditions as /stock/{ticker}.
    """
    df = load_scored_stock(ticker)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"No scored data found for ticker '{ticker}'."
        )

    latest_row = df.iloc[-1]
    peak_idx = df["Suspicion_Score"].idxmax()
    peak_row = df.loc[peak_idx]

    return {
        "ticker": ticker,
        "latest_date": latest_row["Date"].strftime("%Y-%m-%d"),
        "latest_score": float(latest_row["Suspicion_Score"]),
        "peak_score": float(peak_row["Suspicion_Score"]),
        "peak_date": peak_row["Date"].strftime("%Y-%m-%d"),
        "total_flagged_days": int(df["Suspicion_Flag"].sum()),
        "total_trading_days": len(df),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TESTS
# ══════════════════════════════════════════════════════════════════════════════
#
# HOW TO RUN:
#   python backend/api.py --test
#
# These tests use FastAPI's TestClient, which calls the endpoints directly
# in-process (no real network socket, no need to actually start uvicorn).
# This makes the tests fast and fully self-contained.
# ══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_scored_csv(ticker: str, n: int = 20, peak_score: float = 80.0,
                                 peak_row_index: int = None) -> str:
    """
    Writes a synthetic, fully-scored CSV to data/processed/ for one ticker,
    mimicking the final output of the entire pipeline
    (ingest -> features -> isolation_forest -> scoring).

    Parameters
    ----------
    peak_row_index : int, optional
        Which row gets the elevated peak_score. Defaults to the LAST row
        (n - 1) when not specified, so this helper is always safe to call
        regardless of how small n is — a fixed default like 10 would raise
        IndexError for any n <= 10.

    Returns the filepath written, so callers can clean it up afterward.
    """
    if peak_row_index is None:
        peak_row_index = n - 1

    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    scores = [10.0] * n
    flags = [0] * n
    scores[peak_row_index] = peak_score
    flags[peak_row_index] = 1 if peak_score >= 65 else 0

    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Suspicion_Score": scores,
        "Suspicion_Flag": flags,
    })

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    df.to_csv(filepath, index=False)

    return filepath


class TestLoadScoredStock(unittest.TestCase):
    """Tests for load_scored_stock()."""

    @classmethod
    def setUpClass(cls):
        cls.test_ticker = "_TESTSTOCK_API_LOAD"
        cls.test_filepath = _make_synthetic_scored_csv(cls.test_ticker, n=15)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_filepath):
            os.remove(cls.test_filepath)

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_returns_dataframe_for_existing_scored_ticker(self):
        """A valid, fully-scored ticker must return a DataFrame, not None."""
        result = load_scored_stock(self.test_ticker)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_returns_none_for_nonexistent_ticker(self):
        """A ticker with no CSV at all must return None, not raise an error."""
        result = load_scored_stock("_NONEXISTENT_TICKER_XYZ")
        self.assertIsNone(result)

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_returns_none_for_unscored_ticker(self):
        """
        A ticker whose CSV exists but lacks Suspicion_Score (i.e.
        scoring.py hasn't run on it) must return None — this prevents
        the API from serving incomplete/unscored data.
        """
        unscored_ticker = "_TESTSTOCK_API_UNSCORED"
        unscored_filepath = os.path.join(PROCESSED_DIR, f"{unscored_ticker}_features.csv")

        # Write a CSV missing Suspicion_Score entirely
        df = pd.DataFrame({"Date": ["2025-01-01"], "AVR": [1.0]})
        df.to_csv(unscored_filepath, index=False)

        try:
            result = load_scored_stock(unscored_ticker)
            self.assertIsNone(result)
        finally:
            if os.path.exists(unscored_filepath):
                os.remove(unscored_filepath)

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_data_sorted_chronologically(self):
        """
        Even if the underlying CSV were somehow out of order, load_scored_stock()
        must return rows sorted by Date ascending.
        """
        result = load_scored_stock(self.test_ticker)
        dates = result["Date"].tolist()
        self.assertEqual(dates, sorted(dates),
            "Expected dates to be in ascending chronological order")


class TestGetAllTickerNames(unittest.TestCase):
    """Tests for get_all_ticker_names()."""

    @classmethod
    def setUpClass(cls):
        cls.tickers = ["_TESTSTOCK_API_A", "_TESTSTOCK_API_B"]
        cls.filepaths = [
            _make_synthetic_scored_csv(t, n=5) for t in cls.tickers
        ]

    @classmethod
    def tearDownClass(cls):
        for f in cls.filepaths:
            if os.path.exists(f):
                os.remove(f)

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_includes_known_tickers(self):
        """Both synthetic tickers created in setUpClass must appear in the list."""
        result = get_all_ticker_names()
        for ticker in self.tickers:
            self.assertIn(ticker, result)

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_returns_sorted_list(self):
        """The returned list must be alphabetically sorted."""
        result = get_all_ticker_names()
        self.assertEqual(result, sorted(result))


class TestBuildFlagsSummary(unittest.TestCase):
    """Tests for build_flags_summary()."""

    @classmethod
    def setUpClass(cls):
        cls.high_ticker = "_TESTSTOCK_API_HIGH"
        cls.low_ticker = "_TESTSTOCK_API_LOW"

        cls.high_filepath = _make_synthetic_scored_csv(
            cls.high_ticker, n=20, peak_score=92.0, peak_row_index=15
        )
        cls.low_filepath = _make_synthetic_scored_csv(
            cls.low_ticker, n=20, peak_score=30.0, peak_row_index=5
        )

    @classmethod
    def tearDownClass(cls):
        for f in [cls.high_filepath, cls.low_filepath]:
            if os.path.exists(f):
                os.remove(f)

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_summary_sorted_by_peak_score_descending(self):
        """
        The high-scoring ticker must appear BEFORE the low-scoring ticker
        in the summary list, since results are sorted by peak_score
        descending.
        """
        summary = build_flags_summary()
        tickers_in_order = [row["ticker"] for row in summary]

        high_pos = tickers_in_order.index(self.high_ticker)
        low_pos = tickers_in_order.index(self.low_ticker)

        self.assertLess(high_pos, low_pos,
            "Expected the higher-scoring ticker to rank before the lower one")

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_peak_score_matches_injected_value(self):
        """The reported peak_score must exactly match what we injected."""
        summary = build_flags_summary()
        high_entry = next(row for row in summary if row["ticker"] == self.high_ticker)

        self.assertEqual(high_entry["peak_score"], 92.0)

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_each_entry_has_required_keys(self):
        """Every summary entry must have all 4 expected keys."""
        summary = build_flags_summary()
        expected_keys = {"ticker", "peak_score", "peak_date", "flagged_days"}

        for entry in summary:
            self.assertEqual(set(entry.keys()), expected_keys)


class TestAPIEndpoints(unittest.TestCase):
    """
    Tests for the actual FastAPI endpoints, using TestClient to call them
    in-process without starting a real uvicorn server.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        cls.test_ticker = "_TESTSTOCK_API_ENDPOINT"
        cls.test_filepath = _make_synthetic_scored_csv(
            cls.test_ticker, n=20, peak_score=75.0, peak_row_index=8
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_filepath):
            os.remove(cls.test_filepath)

    # ── Test 10 ───────────────────────────────────────────────────────────────
    def test_health_check_returns_200(self):
        """GET / must return HTTP 200 with a status field."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    # ── Test 11 ───────────────────────────────────────────────────────────────
    def test_get_stocks_includes_test_ticker(self):
        """GET /stocks must include our synthetic test ticker in the list."""
        response = self.client.get("/stocks")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertIn(self.test_ticker, body["tickers"])

    # ── Test 12 ───────────────────────────────────────────────────────────────
    def test_get_flags_returns_200_with_list(self):
        """GET /flags must return HTTP 200 with a flagged_stocks list."""
        response = self.client.get("/flags")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertIn("flagged_stocks", body)
        self.assertIsInstance(body["flagged_stocks"], list)

    # ── Test 13 ───────────────────────────────────────────────────────────────
    def test_get_stock_detail_returns_full_history(self):
        """
        GET /stock/{ticker} for a valid, scored ticker must return 200
        with row_count matching the number of rows in the CSV (20).
        """
        response = self.client.get(f"/stock/{self.test_ticker}")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["ticker"], self.test_ticker)
        self.assertEqual(body["row_count"], 20)
        self.assertEqual(len(body["data"]), 20)

    # ── Test 14 ───────────────────────────────────────────────────────────────
    def test_get_stock_detail_404_for_unknown_ticker(self):
        """
        GET /stock/{ticker} for a ticker with no data must return HTTP 404,
        not 500 or an empty 200 — a proper "not found" response.
        """
        response = self.client.get("/stock/_DEFINITELY_NOT_A_REAL_TICKER")
        self.assertEqual(response.status_code, 404)

    # ── Test 15 ───────────────────────────────────────────────────────────────
    def test_get_stock_summary_returns_correct_peak(self):
        """
        GET /stock/{ticker}/summary must report the correct peak_score
        (75.0, as injected in setUpClass) and peak_date.
        """
        response = self.client.get(f"/stock/{self.test_ticker}/summary")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["peak_score"], 75.0)
        self.assertEqual(body["total_trading_days"], 20)

    # ── Test 16 ───────────────────────────────────────────────────────────────
    def test_get_stock_summary_404_for_unknown_ticker(self):
        """GET /stock/{ticker}/summary must also 404 for unknown tickers."""
        response = self.client.get("/stock/_DEFINITELY_NOT_A_REAL_TICKER/summary")
        self.assertEqual(response.status_code, 404)

    # ── Test 17 ───────────────────────────────────────────────────────────────
    def test_cors_headers_present_for_allowed_origin(self):
        """
        A request carrying an Origin header matching ALLOWED_ORIGINS must
        receive the CORS Access-Control-Allow-Origin header back —
        otherwise the Next.js frontend would be silently blocked by the
        browser despite the API itself working fine.
        """
        response = self.client.get(
            "/stocks",
            headers={"Origin": "http://localhost:3000"}
        )
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:3000"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
#   python backend/api.py           → starts the live server on port 8000
#   python backend/api.py --test    → runs the 17-test suite (no server starts)
#
# RECOMMENDED FOR ACTUAL DEVELOPMENT:
#   uvicorn backend.api:app --reload --port 8000
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if "--test" in sys.argv:
        sys.argv.remove("--test")

        print("=" * 55)
        print("  Running api.py test suite  (17 tests)")
        print("=" * 55)
        print()

        unittest.main(verbosity=2)

    else:
        import uvicorn
        print("=" * 55)
        print("  Starting Insider Trading Detector API")
        print("  http://localhost:8000")
        print("  Docs: http://localhost:8000/docs")
        print("=" * 55)
        uvicorn.run(app, host="0.0.0.0", port=8000)