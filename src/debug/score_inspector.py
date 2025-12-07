import pandas as pd
import matplotlib.pyplot as plt
import os


def analyze_active_universe(file_path, score_col='trend_score'):
    """
    Counts how many symbols have a VALID (Non-Zero, Non-NaN) score at each timestamp.
    This reveals the true 'Active Universe' size.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    # Ensure timestamp is datetime
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])

    print(f"Analyzing active universe size for {score_col}...")

    # 1. Group by Timestamp and Count Active Assets
    # Condition: Score is NOT 0.0 AND Score is NOT NaN
    active_count = df.groupby('ts')[score_col].apply(
        lambda x: ((x != 0) & x.notna()).sum())

    # 2. Count Total Records (including 0.0s) per timestamp for comparison
    # .count() excludes NaNs but includes 0.0s
    total_records = df.groupby('ts')[score_col].count()

    # 3. Calculate Active Ratio
    active_ratio = (active_count / total_records) * 100

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Raw Counts
    ax1.plot(active_count.index, active_count,
             label=f'Active Count ({score_col} != 0)', color='blue', linewidth=1.5)
    ax1.plot(total_records.index, total_records,
             label='Total Records (Includes 0.0)', color='gray', linestyle='--', alpha=0.5)

    ax1.set_title(
        f'Universe Health: Number of Assets with Non-Zero {score_col}')
    ax1.set_ylabel('Number of Assets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Active Percentage
    ax2.plot(active_ratio.index, active_ratio,
             label=f'% Active', color='green')
    ax2.set_title(f'Active Ratio: % of Universe with Non-Zero Score')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()

    # Save or Show
    filename = f"universe_health_{score_col}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.show()


# --- Usage ---
if __name__ == "__main__":
    # Point this to your generated CSV
    csv_path = "reports/score_inspection.csv"

    # Analyze the Trend Score (Primary Signal)
    analyze_active_universe(csv_path, score_col='basis_momentum')

    # You can also check other factors
    # analyze_active_universe(csv_path, score_col='volatility')
