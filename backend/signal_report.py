# backend/signal_report.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Generates a decisive BUY or SELL signal with STRONG EVIDENCE
#           for each signal — not just a score, but a clear explanation of
#           exactly what triggered it and why it is high confidence.
#
# WHAT "STRONG EVIDENCE" MEANS HERE:
#   Every signal printed shows:
#   1. DIRECTION     — BUY or SELL (decisive, no ambiguity)
#   2. CONFIDENCE    — HIGH / MEDIUM based on RF quality score
#   3. CANDLE PROOF  — exact Open, High, Low, Close showing the pattern
#   4. VOLUME PROOF  — how many times normal volume today was
#   5. TREND PROOF   — what AVR baseline says (stock in accumulation zone?)
#   6. MOMENTUM      — is price momentum aligned with the signal?
#   7. HISTORICAL    — how similar past signals performed (forward returns)
#   8. RISK          — what would invalidate this signal (stop-loss level)
#
# HOW TO RUN:
#   python backend/signal_report.py              → today's signals, all stocks
#   python backend/signal_report.py CGPOWER      → one stock, full evidence
#   python backend/signal_report.py --history    → all historical signals
#   python backend/signal_report.py --test       → test suite
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

# Import detection logic from same_day_signal.py
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))
from backend.same_day_signal import (
    compute_baseline_features,
    compute_trigger_features,
    generate_same_day_signals,
    validate_signal_quality,
    run_second_stage,
    FORWARD_6M_DAYS,
    BUY_RETURN_THRESHOLD,
    SELL_RETURN_THRESHOLD,
)

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "data/results"

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD   = 0.70   # RF quality score >= 70% = HIGH confidence
MEDIUM_CONFIDENCE_THRESHOLD = 0.45   # RF quality score >= 45% = MEDIUM confidence

# Stop loss buffer
STOP_LOSS_PCT = 0.05   # 5% below entry for BUY, 5% above entry for SELL


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EVIDENCE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_signal_evidence(row: pd.Series, df: pd.DataFrame, ticker: str) -> dict:
    """
    Takes a single signal row and builds a complete evidence dict explaining
    exactly WHY this is a BUY or SELL signal with specific numbers.

    Parameters
    ----------
    row    : the single signal row from the signals DataFrame
    df     : the full stock history (for historical context)
    ticker : stock name for display

    Returns
    -------
    dict with all evidence fields for printing
    """
    direction    = str(row["Signal_Label"])
    rf_signal    = str(row.get("RF_Final_Signal", "UNSCORED"))
    quality_score = float(row.get("RF_Quality_Score", 0.0))
    signal_date  = str(row["Date"])[:10]
    close_price  = float(row["Close"])
    open_price   = float(row.get("Open", close_price))
    high_price   = float(row.get("High", close_price))
    low_price    = float(row.get("Low", close_price))
    volume       = int(row.get("Volume", 0))
    vol_surge    = float(row.get("Trigger_Volume_Surge", 0))
    body_pct     = float(row.get("Trigger_Body_Pct", 0)) * 100
    candle_score = float(row.get("Trigger_Candle_Score", 0))
    baseline_avr = float(row.get("Baseline_AVR_EMA", 0))
    car_10       = float(row.get("CAR_10", 0)) * 100
    return_z     = float(row.get("Return_Z", 0))
    upper_shadow = float(row.get("Trigger_Upper_Shadow", 0)) * 100
    lower_shadow = float(row.get("Trigger_Lower_Shadow", 0)) * 100
    momentum     = float(row.get("Price_Momentum_5", 0)) * 100 if "Price_Momentum_5" in row.index else 0.0

    # Confidence level
    if quality_score >= HIGH_CONFIDENCE_THRESHOLD:
        confidence = "HIGH"
    elif quality_score >= MEDIUM_CONFIDENCE_THRESHOLD:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Stop loss
    if direction == "BUY":
        stop_loss = round(close_price * (1 - STOP_LOSS_PCT), 2)
        target_1  = round(close_price * 1.15, 2)   # +15% target
        target_2  = round(close_price * 1.30, 2)   # +30% target
    else:
        stop_loss = round(close_price * (1 + STOP_LOSS_PCT), 2)
        target_1  = round(close_price * 0.85, 2)   # -15% target
        target_2  = round(close_price * 0.70, 2)   # -30% target

    # Historical similar signals — how did past signals in this stock perform?
    historical_returns = []
    if "Signal_Label" in df.columns and "Close" in df.columns:
        past_signals = df[
            (df["Signal_Label"] == direction) &
            (df["Date"] < signal_date)
        ].copy()
        close_arr = df["Close"].values
        n = len(df)
        for _, past_row in past_signals.iterrows():
            past_idx = df.index[df["Date"] == past_row["Date"]]
            if len(past_idx) == 0:
                continue
            idx = past_idx[0]
            fwd_idx = idx + FORWARD_6M_DAYS
            if fwd_idx < n:
                fwd_ret = (close_arr[fwd_idx] - close_arr[idx]) / close_arr[idx] * 100
                historical_returns.append(round(fwd_ret, 1))

    avg_hist_return = round(np.mean(historical_returns), 1) if historical_returns else None
    win_rate = None
    if historical_returns:
        if direction == "BUY":
            wins = sum(1 for r in historical_returns if r > 0)
        else:
            wins = sum(1 for r in historical_returns if r < 0)
        win_rate = round(wins / len(historical_returns) * 100, 1)

    # Evidence strength summary
    evidence_points = []
    if direction == "BUY":
        if close_price > open_price:
            evidence_points.append(f"Green candle confirmed (Close {close_price} > Open {open_price})")
        if vol_surge >= 1.6:
            evidence_points.append(f"Volume surge {vol_surge:.1f}x above 5-day average")
        if baseline_avr >= 2.0:
            evidence_points.append(f"AVR baseline elevated at {baseline_avr:.2f} (top 10% historically)")
        if car_10 > 2:
            evidence_points.append(f"CAR_10 positive at +{car_10:.2f}% (outperforming market)")
        if upper_shadow < 1.5:
            evidence_points.append(f"Minimal upper shadow ({upper_shadow:.1f}%) — no selling resistance")
        if return_z > 1.5:
            evidence_points.append(f"Return Z-score {return_z:.2f} — statistically elevated return today")
    else:
        if close_price < open_price:
            evidence_points.append(f"Red candle confirmed (Close {close_price} < Open {open_price})")
        if vol_surge >= 1.6:
            evidence_points.append(f"Volume surge {vol_surge:.1f}x above 5-day average")
        if baseline_avr >= 2.0:
            evidence_points.append(f"AVR baseline elevated at {baseline_avr:.2f} (top 10% historically)")
        if car_10 < -2:
            evidence_points.append(f"CAR_10 negative at {car_10:.2f}% (underperforming market)")
        if lower_shadow < 1.5:
            evidence_points.append(f"Minimal lower shadow ({lower_shadow:.1f}%) — no buying support")
        if return_z < -1.5:
            evidence_points.append(f"Return Z-score {return_z:.2f} — statistically unusual negative return")

    return {
        "ticker"           : ticker,
        "date"             : signal_date,
        "direction"        : direction,
        "confidence"       : confidence,
        "quality_score"    : quality_score,
        "rf_signal"        : rf_signal,
        "close"            : close_price,
        "open"             : open_price,
        "high"             : high_price,
        "low"              : low_price,
        "volume"           : volume,
        "vol_surge"        : vol_surge,
        "body_pct"         : body_pct,
        "candle_score"     : candle_score,
        "baseline_avr"     : baseline_avr,
        "car_10_pct"       : car_10,
        "return_z"         : return_z,
        "upper_shadow_pct" : upper_shadow,
        "lower_shadow_pct" : lower_shadow,
        "stop_loss"        : stop_loss,
        "target_1"         : target_1,
        "target_2"         : target_2,
        "evidence_points"  : evidence_points,
        "historical_count" : len(historical_returns),
        "avg_hist_return"  : avg_hist_return,
        "win_rate_pct"     : win_rate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIGNAL PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_signal(ev: dict):
    """
    Prints one signal with full evidence in a clean, readable format.
    This is what appears in your terminal or daily report.
    """
    direction  = ev["direction"]
    confidence = ev["confidence"]

    # Direction emoji and colour indicator
    arrow   = "🟢 BUY" if direction == "BUY" else "🔴 SELL"
    conf_str = f"[{confidence} CONFIDENCE — {ev['quality_score']:.0%}]"

    print()
    print("═" * 65)
    print(f"  {arrow}  ←  {ev['ticker']}            {conf_str}")
    print(f"  Signal Date : {ev['date']}")
    print("═" * 65)

    # Candle structure
    print(f"\n  📊 CANDLE STRUCTURE")
    print(f"     Open   : {ev['open']:.2f}")
    print(f"     High   : {ev['high']:.2f}")
    print(f"     Low    : {ev['low']:.2f}")
    print(f"     Close  : {ev['close']:.2f}   "
          f"({'▲ UP' if ev['close'] > ev['open'] else '▼ DOWN'} "
          f"{ev['body_pct']:.2f}% body)")
    print(f"     Candle Quality Score : {ev['candle_score']:.2f} / 1.0")

    # Volume
    print(f"\n  📈 VOLUME EVIDENCE")
    print(f"     Today's Volume  : {ev['volume']:,}")
    print(f"     Volume Surge    : {ev['vol_surge']:.2f}x the 5-day average")
    vol_comment = "VERY STRONG" if ev["vol_surge"] >= 3 else "STRONG" if ev["vol_surge"] >= 2 else "MODERATE"
    print(f"     Assessment      : {vol_comment} institutional participation")

    # Statistical signals
    print(f"\n  📉 STATISTICAL SIGNALS")
    print(f"     AVR Baseline EMA : {ev['baseline_avr']:.3f}  "
          f"({'⚠ ELEVATED' if ev['baseline_avr'] >= 2.0 else 'Normal'})")
    print(f"     CAR_10           : {ev['car_10_pct']:+.3f}%  "
          f"(10-day cumulative abnormal return vs Nifty)")
    print(f"     Return Z-Score   : {ev['return_z']:+.3f}  "
          f"({'Significant' if abs(ev['return_z']) > 2 else 'Moderate'})")

    # Trade levels
    print(f"\n  🎯 TRADE LEVELS")
    print(f"     Entry     : ₹{ev['close']:.2f}  (today's close)")
    if direction == "BUY":
        print(f"     Stop Loss : ₹{ev['stop_loss']:.2f}  (-5% from entry)")
        print(f"     Target 1  : ₹{ev['target_1']:.2f}  (+15% — 3M horizon)")
        print(f"     Target 2  : ₹{ev['target_2']:.2f}  (+30% — 6M horizon)")
    else:
        print(f"     Stop Loss : ₹{ev['stop_loss']:.2f}  (+5% from entry)")
        print(f"     Target 1  : ₹{ev['target_1']:.2f}  (-15% — 3M horizon)")
        print(f"     Target 2  : ₹{ev['target_2']:.2f}  (-30% — 6M horizon)")

    # Evidence checklist
    print(f"\n  ✅ EVIDENCE CHECKLIST ({len(ev['evidence_points'])} points confirmed)")
    for point in ev["evidence_points"]:
        print(f"     • {point}")

    # Historical performance
    if ev["historical_count"] > 0:
        print(f"\n  📜 HISTORICAL CONTEXT")
        print(f"     Past similar signals in {ev['ticker']}  : {ev['historical_count']}")
        print(f"     Average 6-month forward return         : {ev['avg_hist_return']:+.1f}%")
        print(f"     Win rate (signal direction confirmed)  : {ev['win_rate_pct']:.1f}%")
    else:
        print(f"\n  📜 HISTORICAL CONTEXT")
        print(f"     No prior {direction} signals in training period to compare against.")

    print(f"\n  ⚠  DISCLAIMER: Statistical pattern — not investment advice.")
    print("═" * 65)


def print_summary_header(signals: list[dict], run_date: str):
    """Prints the summary block before individual signals."""
    n_buy  = sum(1 for s in signals if s["direction"] == "BUY")
    n_sell = sum(1 for s in signals if s["direction"] == "SELL")
    n_high = sum(1 for s in signals if s["confidence"] == "HIGH")

    print()
    print("╔" + "═" * 63 + "╗")
    print(f"║  INSIDER TRADING PATTERN DETECTOR — SIGNAL REPORT")
    print(f"║  Date   : {run_date}")
    print(f"║  Stocks : 38 NSE sector stocks monitored")
    print(f"╠" + "═" * 63 + "╣")
    print(f"║  🟢 BUY signals today   : {n_buy}")
    print(f"║  🔴 SELL signals today  : {n_sell}")
    print(f"║  ⭐ HIGH confidence     : {n_high}")
    print("╚" + "═" * 63 + "╝")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def load_and_run_stock(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    Loads one stock's features CSV, runs the full same-day signal pipeline,
    and returns (full_df, signals_df). Returns (None, None) if data missing.
    """
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(filepath):
        return None, None

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing = [c for c in ["Open","High","Low","Close","Volume"] if c not in df.columns]
    if missing:
        return None, None

    df = compute_baseline_features(df)
    df = compute_trigger_features(df)
    df = generate_same_day_signals(df)
    df = validate_signal_quality(df)
    df, _ = run_second_stage(df, ticker)

    signals = df[df["Signal_Raw"] != 0].copy()
    return df, signals


def run_today(ticker_names: list = None, history_mode: bool = False) -> list[dict]:
    """
    Runs the signal report for all stocks.

    history_mode=False : returns only the MOST RECENT signal per stock
                          (what you'd see at 6 PM today)
    history_mode=True  : returns ALL historical signals for review
    """
    if ticker_names is None:
        if not os.path.exists(PROCESSED_DIR):
            print(f"  Error: {PROCESSED_DIR}/ not found. Run features.py first.")
            return []
        ticker_names = sorted([
            f.replace("_features.csv", "")
            for f in os.listdir(PROCESSED_DIR)
            if f.endswith("_features.csv")
        ])

    all_evidence = []

    for ticker in ticker_names:
        full_df, signals = load_and_run_stock(ticker)
        if full_df is None or signals is None or signals.empty:
            continue

        if not history_mode:
            # Only take the most recent signal
            recent = signals.tail(1)
        else:
            recent = signals

        for _, row in recent.iterrows():
            ev = build_signal_evidence(row, full_df, ticker)
            # Only include signals with at least some confidence
            if ev["quality_score"] >= MEDIUM_CONFIDENCE_THRESHOLD or \
               ev["rf_signal"] in ("BUY_UNSCORED", "SELL_UNSCORED"):
                all_evidence.append(ev)

    # Sort: HIGH confidence first, then MEDIUM, then by quality score
    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_evidence.sort(key=lambda e: (
        confidence_order.get(e["confidence"], 3),
        -e["quality_score"]
    ))

    return all_evidence


def save_signals_csv(signals: list[dict], filename: str = "signal_report.csv"):
    """Saves signal evidence to CSV for the API and dashboard."""
    if not signals:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for s in signals:
        rows.append({
            "Date"           : s["date"],
            "Ticker"         : s["ticker"],
            "Direction"      : s["direction"],
            "Confidence"     : s["confidence"],
            "Quality_Score"  : s["quality_score"],
            "Close"          : s["close"],
            "Stop_Loss"      : s["stop_loss"],
            "Target_1"       : s["target_1"],
            "Target_2"       : s["target_2"],
            "Volume_Surge"   : s["vol_surge"],
            "AVR_Baseline"   : s["baseline_avr"],
            "CAR_10_Pct"     : s["car_10_pct"],
            "Return_Z"       : s["return_z"],
            "Evidence_Count" : len(s["evidence_points"]),
            "Hist_Win_Rate"  : s["win_rate_pct"],
            "Avg_6M_Return"  : s["avg_hist_return"],
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, filename), index=False
    )
    print(f"\n  Saved to: {RESULTS_DIR}/{filename}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_signal_row(direction: str = "BUY") -> pd.Series:
    """Creates a synthetic signal row for testing."""
    close = 250.0
    open_ = 240.0 if direction == "BUY" else 260.0
    return pd.Series({
        "Date": "2024-03-15",
        "Signal_Label": direction,
        "RF_Final_Signal": f"HIGH_QUALITY_{direction}",
        "RF_Quality_Score": 0.78,
        "Signal_Raw": 1 if direction == "BUY" else -1,
        "Close": close, "Open": open_,
        "High": close + 5, "Low": open_ - 3,
        "Volume": 2_500_000,
        "Trigger_Volume_Surge": 2.8,
        "Trigger_Body_Pct": 0.042,
        "Trigger_Candle_Score": 0.72,
        "Trigger_Candle_Dir": 1 if direction == "BUY" else -1,
        "Trigger_Upper_Shadow": 0.008,
        "Trigger_Lower_Shadow": 0.005,
        "Baseline_AVR_EMA": 2.45,
        "CAR_10": 0.068 if direction == "BUY" else -0.068,
        "Return_Z": 2.1 if direction == "BUY" else -2.1,
        "Price_Momentum_5": 0.03,
    })


def _make_empty_df() -> pd.DataFrame:
    """Creates a minimal empty history DataFrame for testing."""
    return pd.DataFrame(columns=["Date", "Close", "Signal_Label", "Signal_Raw"])


class TestBuildSignalEvidence(unittest.TestCase):

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_buy_signal_evidence_structure(self):
        """BUY evidence must have all required keys."""
        row = _make_signal_row("BUY")
        ev  = build_signal_evidence(row, _make_empty_df(), "CGPOWER")
        required = ["ticker","date","direction","confidence","quality_score",
                    "close","stop_loss","target_1","target_2","evidence_points",
                    "vol_surge","baseline_avr","car_10_pct","return_z"]
        for key in required:
            self.assertIn(key, ev, f"Missing key: {key}")

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_sell_signal_evidence_structure(self):
        """SELL evidence must have all required keys."""
        row = _make_signal_row("SELL")
        ev  = build_signal_evidence(row, _make_empty_df(), "BALAMINES")
        self.assertEqual(ev["direction"], "SELL")
        self.assertIn("stop_loss", ev)

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_buy_stop_loss_is_below_entry(self):
        """BUY stop loss must ALWAYS be below the entry price."""
        row = _make_signal_row("BUY")
        ev  = build_signal_evidence(row, _make_empty_df(), "TEST")
        self.assertLess(ev["stop_loss"], ev["close"],
            "BUY stop loss must be below close price")

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_sell_stop_loss_is_above_entry(self):
        """SELL stop loss must ALWAYS be above the entry price."""
        row = _make_signal_row("SELL")
        ev  = build_signal_evidence(row, _make_empty_df(), "TEST")
        self.assertGreater(ev["stop_loss"], ev["close"],
            "SELL stop loss must be above close price")

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_buy_targets_are_above_entry(self):
        """BUY targets must both be above the entry close price."""
        row = _make_signal_row("BUY")
        ev  = build_signal_evidence(row, _make_empty_df(), "TEST")
        self.assertGreater(ev["target_1"], ev["close"])
        self.assertGreater(ev["target_2"], ev["target_1"])

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_sell_targets_are_below_entry(self):
        """SELL targets must both be below the entry close price."""
        row = _make_signal_row("SELL")
        ev  = build_signal_evidence(row, _make_empty_df(), "TEST")
        self.assertLess(ev["target_1"], ev["close"])
        self.assertLess(ev["target_2"], ev["target_1"])

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_high_confidence_when_score_above_threshold(self):
        """A quality score >= 0.70 must give HIGH confidence."""
        row = _make_signal_row("BUY")
        row["RF_Quality_Score"] = 0.85
        ev  = build_signal_evidence(row, _make_empty_df(), "TEST")
        self.assertEqual(ev["confidence"], "HIGH")

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_medium_confidence_when_score_between_thresholds(self):
        """A quality score between 0.45 and 0.70 must give MEDIUM confidence."""
        row = _make_signal_row("BUY")
        row["RF_Quality_Score"] = 0.55
        ev  = build_signal_evidence(row, _make_empty_df(), "TEST")
        self.assertEqual(ev["confidence"], "MEDIUM")

    # ── Test 9 ────────────────────────────────────────────────────────────────
    def test_evidence_points_not_empty_for_strong_signal(self):
        """A strong BUY signal (green candle, high volume) must produce evidence points."""
        row = _make_signal_row("BUY")
        ev  = build_signal_evidence(row, _make_empty_df(), "TEST")
        self.assertGreater(len(ev["evidence_points"]), 0,
            "Expected at least one evidence point for a strong signal")

    # ── Test 10 ───────────────────────────────────────────────────────────────
    def test_target_2_is_bigger_move_than_target_1(self):
        """Target 2 must be a larger move than Target 1 in the signal direction."""
        buy_row = _make_signal_row("BUY")
        buy_ev  = build_signal_evidence(buy_row, _make_empty_df(), "TEST")
        self.assertGreater(buy_ev["target_2"], buy_ev["target_1"])

        sell_row = _make_signal_row("SELL")
        sell_ev  = build_signal_evidence(sell_row, _make_empty_df(), "TEST")
        self.assertLess(sell_ev["target_2"], sell_ev["target_1"])


class TestSignalRunner(unittest.TestCase):
    """Integration tests for load_and_run_stock and run_today."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        import numpy as np
        rng = np.random.default_rng(seed=77)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        close = 200 + np.cumsum(rng.normal(0.05, 1.5, n))
        open_ = close - rng.normal(0, 2, n)
        high  = np.maximum(open_, close) + rng.uniform(0.5, 3, n)
        low   = np.minimum(open_, close) - rng.uniform(0.5, 3, n)
        vol   = rng.integers(500_000, 3_000_000, n).astype(float)
        avr   = rng.lognormal(-0.2, 0.55, n)
        for idx in [80, 150, 280, 360]:
            avr[idx]  = 4.5
            vol[idx] *= 3.5
            open_[idx] = close[idx] - abs(rng.normal(5, 2))
        pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": open_.round(2), "High": high.round(2),
            "Low": low.round(2), "Close": close.round(2),
            "Volume": vol.astype(int),
            "AVR": np.round(avr, 4),
            "CAR_10": np.round(rng.normal(0, 0.015, n), 6),
            "Vol_Spike": rng.integers(0, 2, n),
            "Return_Z": np.round(rng.normal(0, 0.9, n), 4),
            "IF_Flag": np.zeros(n, dtype=int),
        }).to_csv(os.path.join(PROCESSED_DIR, "_TEST_REPORT_features.csv"), index=False)
        cls.ticker = "_TEST_REPORT"

    @classmethod
    def tearDownClass(cls):
        for f in [f"{cls.ticker}_features.csv"]:
            p = os.path.join(PROCESSED_DIR, f)
            if os.path.exists(p): os.remove(p)
        for f in ["signal_report.csv"]:
            p = os.path.join(RESULTS_DIR, f)
            if os.path.exists(p): os.remove(p)

    # ── Test 11 ───────────────────────────────────────────────────────────────
    def test_load_and_run_returns_dataframes(self):
        """load_and_run_stock must return two DataFrames for a valid ticker."""
        full_df, signals = load_and_run_stock(self.ticker)
        self.assertIsNotNone(full_df)
        self.assertIsInstance(full_df, pd.DataFrame)
        self.assertIsInstance(signals, pd.DataFrame)

    # ── Test 12 ───────────────────────────────────────────────────────────────
    def test_missing_ticker_returns_none(self):
        """Missing ticker must return (None, None) gracefully."""
        full_df, signals = load_and_run_stock("_NONEXISTENT_TICKER_XYZ")
        self.assertIsNone(full_df)
        self.assertIsNone(signals)

    # ── Test 13 ───────────────────────────────────────────────────────────────
    def test_run_today_returns_list(self):
        """run_today must return a list even if no signals found."""
        result = run_today(ticker_names=[self.ticker])
        self.assertIsInstance(result, list)

    # ── Test 14 ───────────────────────────────────────────────────────────────
    def test_save_signals_csv_creates_file(self):
        """save_signals_csv must create the output CSV file."""
        fake_signals = [build_signal_evidence(
            _make_signal_row("BUY"), _make_empty_df(), "TEST_SAVE"
        )]
        save_signals_csv(fake_signals, "signal_report.csv")
        path = os.path.join(RESULTS_DIR, "signal_report.csv")
        self.assertTrue(os.path.exists(path))

    # ── Test 15 ───────────────────────────────────────────────────────────────
    def test_history_mode_returns_more_signals_than_today_mode(self):
        """History mode must return >= signals vs today mode (latest only)."""
        today_signals   = run_today(ticker_names=[self.ticker], history_mode=False)
        history_signals = run_today(ticker_names=[self.ticker], history_mode=True)
        self.assertGreaterEqual(len(history_signals), len(today_signals))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal report with full evidence for each BUY/SELL signal."
    )
    parser.add_argument("ticker", nargs="?", default=None,
        help="Optional: single ticker (e.g. CGPOWER). Omit for all stocks.")
    parser.add_argument("--history", action="store_true",
        help="Show ALL historical signals, not just the most recent.")
    parser.add_argument("--test", action="store_true",
        help="Run the test suite.")
    args = parser.parse_args()

    if args.test:
        sys.argv = [sys.argv[0]]
        print("=" * 60)
        print("  Running signal_report.py test suite  (15 tests)")
        print("=" * 60)
        print()
        unittest.main(verbosity=2)
        sys.exit(0)

    ticker_names = [args.ticker.upper()] if args.ticker else None
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n  Loading signals for {ticker_names or 'all stocks'}...")
    signals = run_today(ticker_names=ticker_names, history_mode=args.history)

    print_summary_header(signals, run_date)

    if not signals:
        print("\n  No signals meeting confidence threshold today.")
        print("  All stocks are below the elevated baseline threshold.")
    else:
        for ev in signals:
            print_signal(ev)

    save_signals_csv(signals)
    print(f"\n  Total signals: {len(signals)}")
    print("  Done.\n")