import pandas as pd
import matplotlib.pyplot as plt
import os


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

    print("\n\n==== Market Regime Analysis ====")

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

# --- NEW FUNCTION ---


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
# --- END NEW FUNCTION ---
