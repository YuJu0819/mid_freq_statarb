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
