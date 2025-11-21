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
    Analyzes and prints strategy performance under different market regimes (Trade-based).
    """
    if trades_df.empty or 'volatility_regime' not in trades_df.columns:
        print("\nNo trades or regime data available for analysis.")
        return

    if 'pnl' not in trades_df.columns:
        print("\nWarning: PnL column not found in trades. Cannot generate regime report.")
        return

    print("\n\n==== Trade-Based Market Regime Analysis (BTC Proxy) ====")

    print("\n--- Performance by Volatility Regime (Trade Entry) ---")
    vol_analysis = trades_df.groupby('volatility_regime')['pnl'].agg(
        ['sum', 'count', lambda x: (x > 0).mean()])
    vol_analysis.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    vol_analysis['Win Rate'] = vol_analysis['Win Rate'].map('{:.2%}'.format)
    print(vol_analysis)

    print("\n--- Performance by Trend Regime (Trade Entry) ---")
    trend_analysis = trades_df.groupby('trend_regime')['pnl'].agg(
        ['sum', 'count', lambda x: (x > 0).mean()])
    trend_analysis.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    trend_analysis['Win Rate'] = trend_analysis['Win Rate'].map(
        '{:.2%}'.format)
    print(trend_analysis)


def generate_weekday_analysis_report(equity_curve: pd.DataFrame):
    """
    Analyzes and prints strategy performance broken down by weekday based on DAILY PnL.
    """
    if equity_curve.empty:
        print("\nNo equity data available for daily weekday analysis.")
        return

    print("\n\n==== Weekday Performance Analysis (Daily PnL Attribution) ====")

    df = equity_curve.copy()

    # 1. Calculate Daily PnL
    df['daily_pnl'] = df['equity'].diff().fillna(0.0)

    # 2. Extract Weekday from the datetime index (assuming it's already a datetime index from engine.py)
    df['weekday'] = df.index.day_name()

    weekday_order = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']

    # 3. Aggregate Daily PnL by Weekday
    weekday_analysis = df.groupby('weekday')['daily_pnl'].agg(
        ['sum', 'mean', 'std', 'count', lambda x: (x > 0).mean()])

    weekday_analysis.columns = [
        'Total PnL', 'Mean Daily PnL', 'Daily PnL Std', 'Day Count', 'Win Rate']

    # Calculate Sharpe Ratio for context
    # Annualized Sharpe (assuming daily data)
    weekday_analysis['Sharpe'] = (
        weekday_analysis['Mean Daily PnL'] / weekday_analysis['Daily PnL Std']) * (365**0.5)

    # Format the output
    weekday_analysis['Win Rate'] = weekday_analysis['Win Rate'].map(
        '{:.2%}'.format)
    weekday_analysis['Total PnL'] = weekday_analysis['Total PnL'].map(
        '${:,.2f}'.format)
    weekday_analysis['Mean Daily PnL'] = weekday_analysis['Mean Daily PnL'].map(
        '${:,.2f}'.format)
    weekday_analysis['Daily PnL Std'] = weekday_analysis['Daily PnL Std'].map(
        '${:,.2f}'.format)
    weekday_analysis['Sharpe'] = weekday_analysis['Sharpe'].map(
        '{:.2f}'.format)

    # Reindex to ensure correct weekday order
    weekday_analysis = weekday_analysis.reindex(weekday_order).fillna(
        {'Total PnL': '$0.00', 'Mean Daily PnL': '$0.00', 'Daily PnL Std': '$0.00', 'Day Count': 0, 'Win Rate': '0.00%', 'Sharpe': '0.00'})

    weekday_analysis['Day Count'] = weekday_analysis['Day Count'].astype(int)

    print(weekday_analysis)


def generate_skew_analysis_report(trades_df: pd.DataFrame):
    """
    Analyzes and prints strategy performance broken down by asset return skewness.
    """
    if trades_df.empty or 'skew_regime' not in trades_df.columns:
        print("\nNo trades or skew data available for analysis.")
        return

    print("\n\n==== Per-Asset Skewness Performance Analysis (Trade Entry) ====")

    skew_analysis = trades_df.groupby('skew_regime')['pnl'].agg(
        ['sum', 'count', lambda x: (x > 0).mean()])
    skew_analysis.columns = ['Total PnL', 'Trade Count', 'Win Rate']
    skew_analysis['Win Rate'] = skew_analysis['Win Rate'].map('{:.2%}'.format)

    skew_order = ['Positive Skew', 'Neutral Skew', 'Negative Skew', 'Unknown']
    skew_analysis = skew_analysis.reindex(skew_order).fillna(
        {'Total PnL': 0, 'Trade Count': 0, 'Win Rate': '0.00%'})
    skew_analysis['Trade Count'] = skew_analysis['Trade Count'].astype(int)

    print(skew_analysis)


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
    # Assuming the index is already datetime from engine.py, but ensuring 'ts' isn't a column
    if 'ts' in df.columns:
        df = df.set_index(pd.to_datetime(df['ts']))

    # Calculate Daily PnL
    df['daily_pnl'] = df['equity'].diff().fillna(0.0)

    regime_cols = ['volatility_regime', 'trend_regime', 'skew_regime']

    for col in regime_cols:
        if col not in df.columns:
            continue

        plt.figure(figsize=(12, 7))

        unique_regimes = df[col].unique()

        for regime in unique_regimes:
            if pd.isna(regime):
                continue

            mask = (df[col] == regime).astype(int)
            regime_daily_pnl = df['daily_pnl'] * mask
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

        try:
            # Group by time and regime, then get the mean score for that group
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


def generate_daily_regime_analysis(equity_curve: pd.DataFrame):
    """
    Analyzes performance based on DAILY PnL attribution to that day's regime.
    """
    if equity_curve.empty:
        return

    df = equity_curve.copy()
    df['daily_pnl'] = df['equity'].diff()
    df['daily_ret'] = df['equity'].pct_change()

    regime_cols = ['volatility_regime', 'trend_regime', 'skew_regime']

    print("\n\n==== Daily PnL Regime Analysis (Attribution by Day) ====")

    for col in regime_cols:
        if col not in df.columns:
            continue

        print(f"\n--- Daily Performance by {col} ---")

        stats = df.dropna().groupby(col)['daily_pnl'].agg(
            ['sum', 'mean', 'std', 'count'])

        stats['Sharpe'] = (stats['mean'] / stats['std']) * (365**0.5)

        win_rate = df.dropna().groupby(
            col)['daily_pnl'].apply(lambda x: (x > 0).mean())
        stats['Win Rate'] = win_rate.map('{:.2%}'.format)

        stats['sum'] = stats['sum'].map('${:,.2f}'.format)
        stats['mean'] = stats['mean'].map('${:,.2f}'.format)
        stats['std'] = stats['std'].map('${:,.2f}'.format)
        stats['Sharpe'] = stats['Sharpe'].map('{:.2f}'.format)

        print(stats)
