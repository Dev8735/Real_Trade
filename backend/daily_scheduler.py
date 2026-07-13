# backend/daily_scheduler.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Automatically runs the ENTIRE insider trading detection pipeline
#           every day at 6:00 PM IST and produces a dated daily report
#           showing each stock's current Suspicion Score and any new flags.
#
# WHAT RUNS AT 6 PM:
#   1. ingest.py        — download fresh price data for all 39 stocks
#   2. features.py      — recompute AVR, CAR_10, Vol_Spike, Return_Z
#   3. zscore_detector  — per-stock Z-score anomaly flags
#   4. isolation_forest — multivariate anomaly flags
#   5. scoring.py       — composite Suspicion Score 0-100
#   6. train_test_predict — model training on 8yr, testing on 2yr, predictions
#   7. Daily report     — printed to terminal AND saved to
#                          data/reports/daily_report_YYYY-MM-DD.txt
#
# HOW TO RUN:
#   python backend/daily_scheduler.py            → starts the scheduler (runs at 6pm daily)
#   python backend/daily_scheduler.py --now      → runs the pipeline RIGHT NOW (for testing)
#   python backend/daily_scheduler.py --report   → just regenerates today's report from
#                                                   existing data (no pipeline re-run)
#
# HOW TO KEEP IT RUNNING IN THE BACKGROUND (Windows):
#   Start-Process python -ArgumentList "backend/daily_scheduler.py" -WindowStyle Hidden
#   (Or add it to Windows Task Scheduler for production use — see bottom of file)
#
# DEPENDENCY:
#   pip install schedule
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, date

import numpy as np
import pandas as pd
import schedule

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROCESSED_DIR  = "data/processed"
REPORTS_DIR    = "data/reports"
RUN_TIME       = "18:00"          # 6:00 PM — change to "09:15" for market open, etc.

SUSPICION_THRESHOLD = 65          # score >= this → flagged in report
HIGH_ALERT_THRESHOLD = 80         # score >= this → HIGH ALERT in report

# Sector mapping — used to group stocks in the daily report
SECTOR_MAP = {
    "PARAS":     "Defence",    "DATAPATTNS": "Defence",  "ASTRAMICRO": "Defence",
    "SIKAINTERP":"Defence",    "AVANTEL":    "Defence",
    "CESC":      "Power",      "GENUSPOWER": "Power",    "SKIPPER":    "Power",
    "RTNPOWER":  "Power",      "PIGL":       "Power",
    "CGPOWER":   "CapGoods",   "APARINDS":   "CapGoods", "RITES":      "CapGoods",
    "VESUVIUS":  "CapGoods",   "TEXRAIL":    "CapGoods",
    "LAURUSLABS":"Pharma",     "AARTIDRUGS": "Pharma",   "INNOVACAP":  "Pharma",
    "SOLARA":    "Pharma",     "FDC":        "Pharma",
    "AARTIIND":  "Chemicals",  "DEEPAKNTR":  "Chemicals","NEONAMINES": "Chemicals",
    "BALAMINES": "Chemicals",  "PRIVISCL":   "Chemicals",
    "POONAWALLA":"NBFC",       "FIVESTAR":   "NBFC",     "MUTHOOTFIN": "NBFC",
    "CHOLAFIN":  "NBFC",       "FEDFINA":    "NBFC",
    "TITAGARH":  "Railways",   "RVNL":       "Railways", "IRCON":      "Railways",
    "SONACOMS":  "Auto",       "ENDURANCE":  "Auto",     "MSUMI":      "Auto",
    "TUBEINVEST":"Auto",       "BHARATFORG": "Auto",
}


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_step(script_path: str, step_name: str) -> bool:
    """
    Runs one Python script as a subprocess and streams its output to terminal.
    Returns True if it exited with code 0, False if it failed.
    """
    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Running {step_name}...")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # let output flow directly to terminal
    )
    if result.returncode != 0:
        print(f"  ❌ {step_name} failed with exit code {result.returncode}")
        return False
    print(f"  ✅ {step_name} complete")
    return True


def run_full_pipeline() -> bool:
    """
    Runs the entire detection pipeline in the correct order.
    Returns True if ALL steps succeeded, False if any step failed
    (subsequent steps are still attempted so partial results are available).
    """
    steps = [
        ("backend/ingest.py",                        "Data Ingestion"),
        ("backend/features.py",                      "Feature Engineering"),
        ("backend/models/zscore_detector.py",         "Z-score Detection"),
        ("backend/models/isolation_forest.py",        "Isolation Forest"),
        ("backend/scoring.py",                        "Suspicion Scoring"),
        ("backend/train_test_predict.py",             "Train / Test / Predict"),
    ]

    all_ok = True
    for script, name in steps:
        ok = run_pipeline_step(script, name)
        if not ok:
            all_ok = False

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# DAILY REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def load_latest_scores() -> list[dict]:
    """
    Reads the latest scored CSV for every stock in data/processed/ and
    returns a list of dicts with the most recent day's data per stock.
    """
    if not os.path.exists(PROCESSED_DIR):
        return []

    rows = []
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        if not fname.endswith("_features.csv"):
            continue

        ticker = fname.replace("_features.csv", "")
        # Skip any leftover large-cap files
        if ticker not in SECTOR_MAP:
            continue

        filepath = os.path.join(PROCESSED_DIR, fname)
        try:
            df = pd.read_csv(filepath, parse_dates=["Date"])
            if "Suspicion_Score" not in df.columns:
                continue
            df = df.sort_values("Date").reset_index(drop=True)

            latest = df.iloc[-1]
            flagged_total = int(df["Suspicion_Flag"].sum())
            max_score = float(df["Suspicion_Score"].max())
            max_date = df.loc[df["Suspicion_Score"].idxmax(), "Date"]

            rows.append({
                "Ticker":         ticker,
                "Sector":         SECTOR_MAP.get(ticker, "Unknown"),
                "Date":           latest["Date"].strftime("%Y-%m-%d"),
                "Close":          float(latest.get("Close", 0)),
                "AVR":            float(latest.get("AVR", 0)) if pd.notna(latest.get("AVR")) else 0,
                "CAR_10":         float(latest.get("CAR_10", 0)) if pd.notna(latest.get("CAR_10")) else 0,
                "IF_Flag":        int(latest.get("IF_Flag", 0)),
                "Today_Score":    float(latest["Suspicion_Score"]),
                "Today_Flag":     int(latest["Suspicion_Flag"]),
                "Max_Score":      max_score,
                "Max_Score_Date": max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date),
                "Total_Flagged":  flagged_total,
            })
        except Exception as e:
            print(f"  Warning: could not load {ticker}: {e}")

    return rows


def generate_report(rows: list[dict], report_date: str = None) -> str:
    """
    Builds the full daily report as a formatted string.

    Parameters
    ----------
    rows       : list of per-stock dicts from load_latest_scores()
    report_date: the date string to put in the report header.
                 Defaults to today.

    Returns
    -------
    str — the complete report text, ready to print or save to file.
    """
    if report_date is None:
        report_date = date.today().strftime("%Y-%m-%d")

    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append(f"  INSIDER TRADING PATTERN DETECTOR — DAILY REPORT")
    lines.append(f"  Date        : {report_date}")
    lines.append(f"  Generated at: {datetime.now().strftime('%H:%M:%S IST')}")
    lines.append(f"  Stocks monitored: {len(rows)}")
    lines.append(f"  Alert threshold : Suspicion Score >= {SUSPICION_THRESHOLD}")
    lines.append(f"  High alert      : Suspicion Score >= {HIGH_ALERT_THRESHOLD}")
    lines.append(sep)

    # ── Section 1: HIGH ALERTS ────────────────────────────────────────────────
    high_alerts = [r for r in rows if r["Today_Score"] >= HIGH_ALERT_THRESHOLD]
    high_alerts.sort(key=lambda r: r["Today_Score"], reverse=True)

    lines.append("")
    lines.append("  🚨 HIGH ALERTS — Immediate Attention Required")
    lines.append(f"  {'-'*68}")

    if high_alerts:
        for r in high_alerts:
            lines.append(
                f"  ⚠️  {r['Ticker']:<12} [{r['Sector']:<10}]  "
                f"Score: {r['Today_Score']:>6.2f}  "
                f"AVR: {r['AVR']:>5.2f}  "
                f"CAR: {r['CAR_10']:>7.4f}  "
                f"IF: {r['IF_Flag']}"
            )
    else:
        lines.append("  None today.")

    # ── Section 2: FLAGGED (>= 65, < 80) ─────────────────────────────────────
    flagged = [r for r in rows
               if r["Today_Score"] >= SUSPICION_THRESHOLD
               and r["Today_Score"] < HIGH_ALERT_THRESHOLD]
    flagged.sort(key=lambda r: r["Today_Score"], reverse=True)

    lines.append("")
    lines.append("  ⚡ FLAGGED TODAY — Worth Monitoring")
    lines.append(f"  {'-'*68}")

    if flagged:
        for r in flagged:
            lines.append(
                f"  →  {r['Ticker']:<12} [{r['Sector']:<10}]  "
                f"Score: {r['Today_Score']:>6.2f}  "
                f"AVR: {r['AVR']:>5.2f}  "
                f"CAR: {r['CAR_10']:>7.4f}  "
                f"IF: {r['IF_Flag']}"
            )
    else:
        lines.append("  None today.")

    # ── Section 3: ALL STOCKS BY SECTOR ──────────────────────────────────────
    lines.append("")
    lines.append("  📊 ALL STOCKS — Today's Scores by Sector")
    lines.append(f"  {'-'*68}")
    lines.append(
        f"  {'Ticker':<12} {'Sector':<10} {'Close':>8} "
        f"{'Today':>7} {'MaxEver':>8} {'MaxDate':<12} {'TotFlag':>8}"
    )
    lines.append(f"  {'-'*68}")

    sectors = sorted(set(r["Sector"] for r in rows))
    for sector in sectors:
        sector_rows = [r for r in rows if r["Sector"] == sector]
        sector_rows.sort(key=lambda r: r["Today_Score"], reverse=True)

        lines.append(f"")
        lines.append(f"  ── {sector} ──")
        for r in sector_rows:
            flag_marker = " ⚠️ " if r["Today_Score"] >= HIGH_ALERT_THRESHOLD \
                else " ⚡ " if r["Today_Score"] >= SUSPICION_THRESHOLD \
                else "   "
            lines.append(
                f"  {flag_marker}{r['Ticker']:<12} {r['Sector']:<10} "
                f"{r['Close']:>8.2f} "
                f"{r['Today_Score']:>7.2f} "
                f"{r['Max_Score']:>8.2f} "
                f"{r['Max_Score_Date']:<12} "
                f"{r['Total_Flagged']:>8}"
            )

    # ── Section 4: SUMMARY STATS ──────────────────────────────────────────────
    lines.append("")
    lines.append(f"  {'-'*68}")
    lines.append("  SUMMARY")
    lines.append(f"  {'-'*68}")
    lines.append(f"  Total stocks monitored  : {len(rows)}")
    lines.append(f"  High alerts today       : {len(high_alerts)}")
    lines.append(f"  Flagged today           : {len(flagged)}")
    lines.append(f"  Clean (score < 65)      : {len(rows) - len(high_alerts) - len(flagged)}")

    if rows:
        scores_today = [r["Today_Score"] for r in rows]
        lines.append(f"  Average score today     : {np.mean(scores_today):.2f}")
        lines.append(f"  Highest score today     : {max(scores_today):.2f}")
        top = max(rows, key=lambda r: r["Today_Score"])
        lines.append(f"  Most suspicious stock   : {top['Ticker']} "
                     f"({top['Today_Score']:.2f}) [{top['Sector']}]")

    lines.append("")
    lines.append(sep)
    lines.append(f"  Report saved to: data/reports/daily_report_{report_date}.txt")
    lines.append(sep)

    return "\n".join(lines)


def save_report(report_text: str, report_date: str = None) -> str:
    """Saves the report to data/reports/ and returns the filepath."""
    if report_date is None:
        report_date = date.today().strftime("%Y-%m-%d")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filepath = os.path.join(REPORTS_DIR, f"daily_report_{report_date}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    return filepath


# ══════════════════════════════════════════════════════════════════════════════
# MAIN JOB — runs at 6 PM
# ══════════════════════════════════════════════════════════════════════════════

def daily_job():
    """
    The full daily job that runs automatically at RUN_TIME (6:00 PM).
    Runs the entire pipeline, then generates and saves the daily report.
    """
    today = date.today().strftime("%Y-%m-%d")

    print("\n")
    print("=" * 72)
    print(f"  DAILY JOB STARTED — {today} at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 72)

    # Step 1: Run the pipeline
    print("\n  PHASE 1: Running detection pipeline...")
    pipeline_ok = run_full_pipeline()

    if not pipeline_ok:
        print("\n  ⚠️  Some pipeline steps failed. "
              "Generating report from available data.")

    # Step 2: Load results and generate report
    print("\n  PHASE 2: Generating daily report...")
    rows = load_latest_scores()

    if not rows:
        print("  ❌ No scored data available. Check that the pipeline ran correctly.")
        return

    report_text = generate_report(rows, report_date=today)
    filepath = save_report(report_text, report_date=today)

    # Step 3: Print report to terminal
    print("\n")
    print(report_text)
    print(f"\n  ✅ Report saved to: {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily scheduler for the Insider Trading Pattern Detector."
    )
    parser.add_argument(
        "--now", action="store_true",
        help="Run the full pipeline and generate report RIGHT NOW "
             "(for testing — doesn't wait for 6 PM)."
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Only regenerate today's report from existing data "
             "(skips pipeline re-run — faster for debugging)."
    )
    args = parser.parse_args()

    if args.now:
        # Immediately run everything — useful for testing
        print("  Running full pipeline and report NOW (--now flag detected)...")
        daily_job()

    elif args.report:
        # Just regenerate today's report without re-running the pipeline
        print("  Regenerating report from existing data (--report flag detected)...")
        rows = load_latest_scores()
        if not rows:
            print("  ❌ No scored data available. Run the pipeline first.")
            sys.exit(1)
        today = date.today().strftime("%Y-%m-%d")
        report_text = generate_report(rows, report_date=today)
        filepath = save_report(report_text, report_date=today)
        print(report_text)
        print(f"\n  ✅ Report saved to: {filepath}")

    else:
        # Normal mode: wait and run at 6 PM every day
        print("=" * 72)
        print("  Insider Trading Detector — Daily Scheduler")
        print("=" * 72)
        print(f"  Scheduled run time : {RUN_TIME} IST (every day)")
        print(f"  To run immediately : python backend/daily_scheduler.py --now")
        print(f"  To report only     : python backend/daily_scheduler.py --report")
        print(f"  Current time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Next run           : today at {RUN_TIME} "
              f"(or tomorrow if {RUN_TIME} has already passed today)")
        print("=" * 72)
        print("  Scheduler is running... Press Ctrl+C to stop.")
        print()

        schedule.every().day.at(RUN_TIME).do(daily_job)

        while True:
            schedule.run_pending()
            time.sleep(30)   # check every 30 seconds


# ══════════════════════════════════════════════════════════════════════════════
# WINDOWS TASK SCHEDULER INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# If you want Windows to auto-start this script at login (so you don't have
# to keep a terminal open), use Windows Task Scheduler:
#
# 1. Open Task Scheduler (search for it in Start)
# 2. Click "Create Basic Task"
# 3. Name: "InsiderTradingDetector"
# 4. Trigger: Daily, at 5:55 PM (5 min before 6 PM gives it time to start)
# 5. Action: "Start a program"
#    Program: C:\Users\HP\PycharmProjects\insider-trading-pattern-detector\.venv\Scripts\python.exe
#    Arguments: backend/daily_scheduler.py --now
#    Start in: C:\Users\HP\PycharmProjects\insider-trading-pattern-detector
# 6. Finish → the job runs automatically every day even without this terminal