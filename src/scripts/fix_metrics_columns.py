import pandas as pd
import glob
import os
import sys


def fix_metrics_files(metrics_dir="./data/metrics"):
    print(f"--- Repairing CSV Columns in {metrics_dir} ---")

    files = glob.glob(os.path.join(metrics_dir, "*.csv"))
    if not files:
        print("No files found.")
        return

    fixed_count = 0
    defaulted_count = 0
    already_good = 0

    # CRITICAL: This list includes the exact name found in your forensic test
    ls_candidates = [
        'count_toptrader_long_short_ratio',  # <--- The one found in your test
        'longShortRatio',
        'long_short_ratio',
        'top_long_short_account_ratio',
        'count_top_trader_long_short_ratio',
        'sum_toptrader_long_short_ratio',
        'sum_top_trader_long_short_ratio',
        'globalLongShortAccountRatio',
        'topLongShortAccountRatio'
    ]

    print(f"Scanning {len(files)} files...")

    for fpath in files:
        try:
            # Read CSV
            df = pd.read_csv(fpath)
            changed = False

            # 1. Standardize Timestamp
            if 'ts' not in df.columns:
                if 'create_time' in df.columns:
                    df.rename(columns={'create_time': 'ts'}, inplace=True)
                    changed = True
                elif 'timestamp' in df.columns:
                    df.rename(columns={'timestamp': 'ts'}, inplace=True)
                    changed = True

            # 2. Standardize Open Interest
            if 'open_interest' not in df.columns:
                if 'sum_open_interest' in df.columns:
                    df.rename(
                        columns={'sum_open_interest': 'open_interest'}, inplace=True)
                    changed = True
                elif 'openInterest' in df.columns:
                    df.rename(
                        columns={'openInterest': 'open_interest'}, inplace=True)
                    changed = True

            # 3. Standardize L/S Ratio
            if 'ls_ratio' not in df.columns:
                found_col = None
                for candidate in ls_candidates:
                    if candidate in df.columns:
                        found_col = candidate
                        break

                if found_col:
                    df.rename(columns={found_col: 'ls_ratio'}, inplace=True)
                    changed = True
                else:
                    # Only default to 1.0 if we really can't find it
                    # (This prevents crashing backtests)
                    df['ls_ratio'] = 1.0
                    changed = True
                    defaulted_count += 1
            else:
                already_good += 1

            # 4. Save if modified
            if changed:
                # Keep only clean columns to reduce file size
                cols_to_keep = ['ts', 'open_interest', 'ls_ratio']
                # Filter strictly
                df = df[cols_to_keep]
                df.to_csv(fpath, index=False)
                fixed_count += 1

        except Exception as e:
            print(f"Error fixing {os.path.basename(fpath)}: {e}")

    print("-" * 30)
    print(f"Files Already Good:     {already_good}")
    print(f"Files Repaired:         {fixed_count}")
    print(f"Files Defaulted (1.0):  {defaulted_count}")
    print("-" * 30)


if __name__ == "__main__":
    fix_metrics_files()
