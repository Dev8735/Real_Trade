"""
Merge Signals Log
------------------
Safely merges two conflicting versions of signals_log.csv by combining
all rows from both, removing exact duplicates. Used to resolve git
conflicts without losing data - critical since this file is written to
by multiple independent workflow runs throughout the day.
"""

import pandas as pd
import subprocess

def merge_conflicted_csv(path):
    # Extract both versions from git's conflict markers
    result = subprocess.run(["git", "show", f":2:{path}"], capture_output=True, text=True)
    ours_content = result.stdout

    result = subprocess.run(["git", "show", f":3:{path}"], capture_output=True, text=True)
    theirs_content = result.stdout

    with open("_ours_temp.csv", "w", encoding="utf-8") as f:
        f.write(ours_content)
    with open("_theirs_temp.csv", "w", encoding="utf-8") as f:
        f.write(theirs_content)

    ours_df = pd.read_csv("_ours_temp.csv")
    theirs_df = pd.read_csv("_theirs_temp.csv")

    combined = pd.concat([ours_df, theirs_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["signal_date", "ticker", "price_at_signal"], keep="first")

    combined.to_csv(path, index=False)
    print(f"Merged: {len(ours_df)} + {len(theirs_df)} rows -> {len(combined)} unique rows after dedup")

    import os
    os.remove("_ours_temp.csv")
    os.remove("_theirs_temp.csv")

if __name__ == "__main__":
    merge_conflicted_csv("data/signals_log.csv")
