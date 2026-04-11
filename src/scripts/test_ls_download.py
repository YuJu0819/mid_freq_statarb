import requests
import zipfile
import pandas as pd
import os
import shutil

# --- CONFIG ---
SYMBOL = "BTCUSDT"
YEAR = "2024"
MONTH = "01"
DAY = "01"

# CORRECTED URL: DAILY instead of MONTHLY
DAILY_URL = f"https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YEAR}-{MONTH}-{DAY}.zip"
TEMP_DIR = "./temp_debug_metrics"


def robust_extract_logic(df):
    # 1. Standardize Time
    if 'create_time' in df.columns:
        df['ts'] = pd.to_datetime(df['create_time'])
    elif 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        return "FAIL: No Timestamp Found"

    # 2. Rename Open Interest
    if 'sum_open_interest' in df.columns:
        df.rename(columns={'sum_open_interest': 'open_interest'}, inplace=True)
    elif 'openInterest' in df.columns:
        df.rename(columns={'openInterest': 'open_interest'}, inplace=True)

    if 'open_interest' not in df.columns:
        return "FAIL: No Open Interest Found"

    # 3. Rename Long/Short Ratio
    ls_candidates = [
        'longShortRatio',
        'long_short_ratio',
        'top_long_short_account_ratio',
        'count_top_trader_long_short_ratio',
        'sum_top_trader_long_short_ratio',
        'globalLongShortAccountRatio'
    ]

    found_ls = None
    for col in ls_candidates:
        if col in df.columns:
            df.rename(columns={col: 'ls_ratio'}, inplace=True)
            found_ls = col
            break

    return found_ls if found_ls else "MISSING"


def run_forensic_test():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    zip_path = os.path.join(TEMP_DIR, "test.zip")

    print(f"1. Downloading Test File: {DAILY_URL}")
    try:
        r = requests.get(DAILY_URL, stream=True)
        if r.status_code != 200:
            print(f"[ERROR] Download failed. HTTP {r.status_code}")
            return
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("   -> Download Complete.")
    except Exception as e:
        print(f"[ERROR] Network error: {e}")
        return

    print("2. Extracting CSV...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = z.namelist()[0]
            z.extractall(TEMP_DIR)
        print(f"   -> Extracted: {csv_name}")
    except Exception as e:
        print(f"[ERROR] Unzip failed: {e}")
        return

    print("3. INSPECTING HEADERS...")
    csv_path = os.path.join(TEMP_DIR, csv_name)
    df = pd.read_csv(csv_path)

    print("\n" + "="*40)
    print("RAW CSV COLUMNS FOUND:")
    print("="*40)
    for col in df.columns:
        print(f" • {col}")
    print("="*40 + "\n")

    print("4. Testing Extraction Logic...")
    result = robust_extract_logic(df)

    print(f"   -> L/S Column Detection: {result}")

    # Cleanup
    shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    run_forensic_test()
