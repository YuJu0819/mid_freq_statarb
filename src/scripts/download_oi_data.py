import requests
import os
import zipfile
from datetime import date, timedelta

# --- Configuration ---
SYMBOLS = [
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "BNBUSDT",
    "ALPACAUSDT",
    "ASTERUSDT",
    "SUIUSDT",
    "1000PEPEUSDT",
    "ADAUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "ZECUSDT",
    "AVAXUSDT",
    "XPLUSDT",
    "ENAUSDT",
    "COAIUSDT",
    "HYPEUSDT",
    "PUMPUSDT",
    "WLFIUSDT",
    "DOTUSDT",
    "WLDUSDT",
    "FARTCOINUSDT",
    "TRUMPUSDT",
    "NEARUSDT",
    "AAVEUSDT",
    "GIGGLEUSDT",
    "PENGUUSDT",
    "WIFUSDT",
    "UNIUSDT",
    "ARBUSDT",
    "1000BONKUSDT",
    "TAOUSDT",
    "BCHUSDT",
    "FILUSDT",
    "APTUSDT",
    "ETCUSDT",
    "INUSDT",
    "ONDOUSDT",
    "XLMUSDT",
    "ZORAUSDT",
    "BNXUSDT",
    "HBARUSDT",
    "1000SHIBUSDT",
    "OPUSDT",
    "CRVUSDT",
    "TIAUSDT",
    "TONUSDT",
    "KGENUSDT",
    "ALPHAUSDT",
    "4USDT",
    "PAXGUSDT",
    "ZENUSDT",
    "TRXUSDT",
    "LDOUSDT",
    "IPUSDT",
    "SEIUSDT",
    "ETHFIUSDT",
    "CAKEUSDT",
    "ATOMUSDT",
    "INJUSDT",
    "AVNTUSDT",
    "FFUSDT",
    "FETUSDT",
    "EIGENUSDT",
    "FORMUSDT",
    "LINEAUSDT",
    "DASHUSDT",
    "YBUSDT",
    "GALAUSDT",
    "AIAUSDT",
    "RENDERUSDT",
    "1000FLOKIUSDT",
    "ALGOUSDT",
    "VIRTUALUSDT",
    "MYXUSDT",
    "USELESSUSDT",
    "SUSDT",
    "PENDLEUSDT",
    "STBLUSDT",
    "STRKUSDT",
    "2ZUSDT",
    "USDCUSDT",
    "BIOUSDT",
    "ORDIUSDT",
    "NEIROUSDT",
    "ENSUSDT",
    "HANAUSDT",
    "WALUSDT",
    "DYDXUSDT",
    "ICPUSDT",
    "POLUSDT",
    "SPXUSDT",
    "SNXUSDT",
    "0GUSDT",
    "PNUTUSDT",
    "BERAUSDT",
    "PYTHUSDT",
    "WUSDT",
]  # Add all the symbols you need
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 11, 1)
OUTPUT_DIRECTORY = "./data/open_interest"  # Directory to save the CSV files
# ---------------------

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def download_oi_data():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created directory: {OUTPUT_DIRECTORY}")

    for symbol in SYMBOLS:
        print(f"\n--- Downloading Open Interest for {symbol} ---")
        for single_date in daterange(START_DATE, END_DATE):
            date_str = single_date.strftime("%Y-%m-%d")
            file_name = f"{symbol}-metrics-{date_str}.zip"
            url = f"{BASE_URL}/{symbol}/{file_name}"

            zip_path = os.path.join(OUTPUT_DIRECTORY, file_name)
            csv_path = os.path.join(
                OUTPUT_DIRECTORY, f"{symbol}-metrics-{date_str}.csv")

            # Skip if CSV already exists
            if os.path.exists(csv_path):
                print(f"Skipping {date_str}, CSV already exists.")
                continue

            try:
                # Download the file
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)

                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Successfully downloaded {file_name}")

                # Unzip the file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(OUTPUT_DIRECTORY)
                print(f"Successfully unzipped to {csv_path}")

                # Clean up the zip file
                os.remove(zip_path)

            except requests.exceptions.HTTPError as e:
                print(
                    f"Failed to download for {date_str}. URL might not exist. Error: {e}")
            except Exception as e:
                print(f"An error occurred for {date_str}: {e}")


if __name__ == "__main__":
    download_oi_data()
