import pandas as pd
import matplotlib.pyplot as plt
import os

# Add this function to src/backtest/reporting.py
# Make sure you have 'import matplotlib.pyplot as plt' and 'import os' at the top


def plot_cross_sectional_analysis(score_df: pd.DataFrame, report_dir: str):
    """
    Analyzes and plots the average cross-sectional factor score by regime.
    """
    if score_df.empty or 'final_score' not in score_df.columns:
        print("\nNo score data available for cross-sectional regime analysis.")
        return

    if not isinstance(score_df.index, pd.DatetimeIndex):
        score_df = score_df.set_index(pd.to_datetime(score_df['ts']))

    regime_cols = ['volatility_regime', 'trend_regime', 'skew_regime']

    for col in regime_cols:
        if col not in score_df.columns:
            continue

        plt.figure(figsize=(12, 7))

        # Group by time and regime, then get the mean score for that group
        # unstack() pivots the regimes into columns for plotting
        try:
            avg_score_by_regime = score_df.groupby([score_df.index, col])[
                'final_score'].mean().unstack()
        except Exception as e:
            print(f"Could not analyze cross-sectional scores for {col}: {e}")
            continue

        if not avg_score_by_regime.empty:
            avg_score_by_regime.plot(ax=plt.gca())

        plt.title(f"Cross-Sectional Average 'final_score' by {col}")
        plt.xlabel("Date")
        plt.ylabel("Average 'final_score'")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(report_dir, f"score_cs_{col}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Cross-sectional score chart saved to: {save_path}")


def plot_daily_regime_pnl_ts(equity_curve: pd.DataFrame, report_dir: str):
    """
    Plots the cumulative PnL curve for each regime based on DAILY attribution.
    This aligns with the 'generate_daily_regime_analysis' report.
    """
    if equity_curve.empty:
        print("No equity curve data for plotting.")
        return

    df = equity_curve.copy()

    # Ensure datetime index for plotting
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts')

    # Calculate Daily PnL
    df['daily_pnl'] = df['equity'].diff().fillna(0.0)

    regime_cols = ['volatility_regime', 'trend_regime', 'skew_regime']

    for col in regime_cols:
        if col not in df.columns:
            continue

        plt.figure(figsize=(12, 7))

        # Get all unique regimes present in the data
        unique_regimes = df[col].unique()

        for regime in unique_regimes:
            if pd.isna(regime):
                continue

            # 1. Create a mask for days belonging to this regime
            mask = (df[col] == regime).astype(int)

            # 2. Attribute PnL only on those days (0 on other days)
            # This ensures the line goes flat when the regime is not active,
            # maintaining the correct time-series perspective.
            regime_daily_pnl = df['daily_pnl'] * mask

            # 3. Calculate Cumulative PnL
            cumulative_pnl = regime_daily_pnl.cumsum()

            plt.plot(cumulative_pnl.index, cumulative_pnl, label=str(regime))

        plt.title(f"Cumulative Daily PnL by {col} (Daily Attribution)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL (USDT)")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(report_dir, f"daily_pnl_ts_{col}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Daily PnL regime chart saved to: {save_path}")


def plot_equity_curve(equity_curve: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve["equity"])
    plt.title("Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USDT)")
    plt.grid(True)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Equity curve saved to: {save_path}")


def generate_regime_analysis_report(trades_df: pd.DataFrame):
    """
    Analyzes and prints strategy performance under different market regimes.
    """
    if trades_df.empty or 'volatility_regime' not in trades_df.columns:
        print("\nNo trades or regime data available for analysis.")
        return

    # Ensure PnL column exists
    if 'pnl' not in trades_df.columns:
        print("\nWarning: PnL column not found in trades. Cannot generate regime report.")
        return

    print("\n\n==== Market Regime Analysis (BTC Proxy) ====")

    # Analyze Volatility Regime
    print("\n--- Performance by Volatility Regime ---")
    vol_analysis = trades_df.groupby('volatility_regime')['pnl'].agg(
        ['sum', 'count', lambda x: (x > 0).mean()])
    vol_analysis.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    vol_analysis['Win Rate'] = vol_analysis['Win Rate'].map('{:.2%}'.format)
    print(vol_analysis)

    # Analyze Trend Regime
    print("\n--- Performance by Trend Regime ---")
    trend_analysis = trades_df.groupby('trend_regime')['pnl'].agg(
        ['sum', 'count', lambda x: (x > 0).mean()])
    trend_analysis.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    trend_analysis['Win Rate'] = trend_analysis['Win Rate'].map(
        '{:.2%}'.format)
    print(trend_analysis)


def generate_weekday_analysis_report(trades_df: pd.DataFrame):
    """
    Analyzes and prints strategy performance broken down by weekday.
    """
    if trades_df.empty or 'pnl' not in trades_df.columns:
        print("\nNo trades or PnL data available for weekday analysis.")
        return

    print("\n\n==== Weekday Performance Analysis ====")

    trades_df['weekday'] = trades_df['ts'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']

    weekday_analysis = trades_df.groupby('weekday')['pnl'].agg(
        ['sum', 'count', lambda x: (x > 0).mean()])
    weekday_analysis.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    weekday_analysis['Win Rate'] = weekday_analysis['Win Rate'].map(
        '{:.2%}'.format)

    # Reindex to ensure correct weekday order
    weekday_analysis = weekday_analysis.reindex(weekday_order).fillna(
        {'Total PnL': 0, 'Trade Count': 0, 'Win Rate': '0.00%'})
    # Ensure Trade Count is integer
    weekday_analysis['Trade Count'] = weekday_analysis['Trade Count'].astype(
        int)

    print(weekday_analysis)

# --- NEW FUNCTION ---


def generate_skew_analysis_report(trades_df: pd.DataFrame):
    """
    Analyzes and prints strategy performance broken down by asset return skewness.
    """
    if trades_df.empty or 'skew_regime' not in trades_df.columns:
        print("\nNo trades or skew data available for analysis.")
        return

    print("\n\n==== Per-Asset Skewness Performance Analysis ====")

    skew_analysis = trades_df.groupby('skew_regime')['pnl'].agg(
        ['sum', 'count', lambda x: (x > 0).mean()])
    skew_analysis.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    skew_analysis['Win Rate'] = skew_analysis['Win Rate'].map('{:.2%}'.format)

    # Reorder to logical sort
    skew_order = ['Positive Skew', 'Neutral Skew', 'Negative Skew', 'Unknown']
    skew_analysis = skew_analysis.reindex(skew_order).fillna(
        {'Total PnL': 0, 'Trade Count': 0, 'Win Rate': '0.00%'})
    skew_analysis['Trade Count'] = skew_analysis['Trade Count'].astype(int)

    print(skew_analysis)
# --- END NEW FUNCTION ---

# --- NEW FUNCTION FOR DAILY ANALYSIS ---


def generate_daily_regime_analysis(equity_curve: pd.DataFrame):
    """
    Analyzes performance based on DAILY PnL attribution to that day's regime.
    """
    if equity_curve.empty:
        return

    # Calculate daily stats
    df = equity_curve.copy()
    df['daily_pnl'] = df['equity'].diff()
    df['daily_ret'] = df['equity'].pct_change()

    regime_cols = ['volatility_regime', 'trend_regime', 'skew_regime']

    print("\n\n==== Daily PnL Regime Analysis (Attribution by Day) ====")

    for col in regime_cols:
        if col not in df.columns:
            continue

        print(f"\n--- Daily Performance by {col} ---")

        # Group by regime and calculate stats
        # We filter out the first row (NaN pnl)
        stats = df.dropna().groupby(col)['daily_pnl'].agg(
            ['sum', 'mean', 'std', 'count'])

        # Annualized Sharpe (assuming daily data)
        stats['Sharpe'] = (stats['mean'] / stats['std']) * (365**0.5)

        # Win Rate (percentage of days with positive PnL)
        win_rate = df.dropna().groupby(
            col)['daily_pnl'].apply(lambda x: (x > 0).mean())
        stats['Win Rate'] = win_rate.map('{:.2%}'.format)

        # Format for readability
        stats['sum'] = stats['sum'].map('${:,.2f}'.format)
        stats['mean'] = stats['mean'].map('${:,.2f}'.format)
        stats['std'] = stats['std'].map('${:,.2f}'.format)
        stats['Sharpe'] = stats['Sharpe'].map('{:.2f}'.format)

        print(stats)
