import requests
import os
import zipfile
import pandas as pd
import time
import shutil
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from ..data.binance_futures_rest import fetch_open_interest
from ..scripts.get_top_futures_symbols import get_top_futures_symbols

# --- Configuration ---
# Update this range to cover your full backtest period
START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 11, 20)
OUTPUT_DIRECTORY = "./data/open_interest"
DAILY_BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
MONTHLY_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/metrics"
# ---------------------


def download_file(url, target_path):
    """Helper to download a file with stream."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"Download error: {e}")
        return False


def process_monthly_data(symbol, year, month):
    """
    Downloads monthly zip, extracts it, splits it into daily CSVs.
    Returns True if successful.
    """
    date_str = f"{year}-{month:02d}"
    file_name = f"{symbol}-metrics-{date_str}.zip"
    url = f"{MONTHLY_BASE_URL}/{symbol}/{file_name}"
    zip_path = os.path.join(OUTPUT_DIRECTORY, "temp_monthly.zip")

    print(f"   [Monthly] Attempting {date_str} archive...")

    if download_file(url, zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract the single CSV inside
                csv_name = zip_ref.namelist()[0]
                zip_ref.extractall(OUTPUT_DIRECTORY)

            extracted_csv_path = os.path.join(OUTPUT_DIRECTORY, csv_name)

            # Read the big monthly CSV
            # Columns usually: symbol, sum_open_interest, sum_open_interest_value, count_top_trader... , create_time
            df = pd.read_csv(extracted_csv_path)

            # Ensure create_time is datetime
            if 'create_time' in df.columns:
                df['dt'] = pd.to_datetime(df['create_time'])
            else:
                # Fallback if format differs
                print("   [Error] Could not find 'create_time' in monthly CSV.")
                os.remove(extracted_csv_path)
                return False

            # Split into days
            for group_date, group_df in df.groupby(df['dt'].dt.date):
                day_str = group_date.strftime("%Y-%m-%d")
                target_csv = os.path.join(
                    OUTPUT_DIRECTORY, f"{symbol}-metrics-{day_str}.csv")

                # Save only if we don't have it (or overwrite to ensure quality)
                group_df.drop(columns=['dt']).to_csv(target_csv, index=False)

            print(f"   [Success] Extracted {date_str} into daily files.")

            # Cleanup
            os.remove(extracted_csv_path)
            os.remove(zip_path)
            return True

        except Exception as e:
            print(f"   [Error] Failed processing monthly zip: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False

    return False


def download_oi_data():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print("Fetching top liquid symbols...")
    symbols = get_top_futures_symbols(top_n=200)

    # 1. Generate list of months to check
    current_date = START_DATE
    months_to_check = []
    while current_date <= END_DATE:
        months_to_check.append((current_date.year, current_date.month))
        current_date += relativedelta(months=1)

    # Deduplicate list of months
    months_to_check = sorted(list(set(months_to_check)))

    for symbol in symbols:
        print(f"\n--- Processing {symbol} ---")

        # --- PHASE 1: Try Monthly Archives (Best for > 1 month old) ---
        for year, month in months_to_check:
            # Skip current month (usually not archived yet)
            if date(year, month, 1) >= date(date.today().year, date.today().month, 1):
                continue

            # Check if we already have data for the middle of this month
            # (Heuristic: If we have the 15th, we probably have the month)
            test_day = f"{year}-{month:02d}-15"
            if os.path.exists(os.path.join(OUTPUT_DIRECTORY, f"{symbol}-metrics-{test_day}.csv")):
                continue

            # Attempt Download
            process_monthly_data(symbol, year, month)

        # --- PHASE 2: Fill Gaps with Daily Archives & API ---
        # We iterate through every single day to fill holes
        current_day = START_DATE
        while current_day <= END_DATE:
            date_str = current_day.strftime("%Y-%m-%d")
            csv_name = f"{symbol}-metrics-{date_str}.csv"
            csv_path = os.path.join(OUTPUT_DIRECTORY, csv_name)

            current_day += timedelta(days=1)

            if os.path.exists(csv_path):
                continue

            # Try Daily Zip
            file_name = f"{symbol}-metrics-{date_str}.zip"
            zip_path = os.path.join(OUTPUT_DIRECTORY, file_name)
            url = f"{DAILY_BASE_URL}/{symbol}/{file_name}"

            if download_file(url, zip_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(OUTPUT_DIRECTORY)
                    os.remove(zip_path)
                    print(f"[{date_str}] Downloaded Daily Archive.")
                    continue
                except:
                    pass

            # Try API (Only if within last 30 days)
            days_diff = (date.today() - (current_day - timedelta(days=1))).days
            if days_diff <= 30:
                print(f"[{date_str}] Fetching via API...")
                day_start = date_str
                day_end = (datetime.strptime(date_str, "%Y-%m-%d") +
                           timedelta(days=1)).strftime("%Y-%m-%d")

                df = fetch_open_interest(
                    symbol, interval="1d", start_date=day_start, end_date=day_end)
                if not df.empty:
                    df.to_csv(csv_path, index=False)
                    print(f"[{date_str}] API Success.")
                    time.sleep(0.1)  # Rate limit
            else:
                # If we are here, it means:
                # 1. No Monthly Archive
                # 2. No Daily Archive
                # 3. Too old for API
                # Data is permanently missing for this specific coin/day.
                print(
                    f"[{date_str}] Data unavailable (Too old for API, no archives).")


if __name__ == "__main__":
    download_oi_data()
