import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


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


def analyze_factor_quantiles_pure_return(score_df: pd.DataFrame, data_dict: dict, factor_name: str, quantiles: int = 5, report_dir: str = "."):
    """
    Groups assets into quantiles based on 'factor_name' and plots their forward performance.
    """
    if score_df.empty or factor_name not in score_df.columns:
        print(
            f"\n[Quantile Analysis] Skipping {factor_name}: Data missing or column not found.")
        return

    print(
        f"\n--- Running Cross-Sectional Analysis for Factor: {factor_name} ---")

    # 1. Pivot Scores to Wide Format (Index=Time, Columns=Symbol, Values=Factor)
    try:
        factor_wide = score_df.pivot(
            index='ts', columns='symbol', values=factor_name)
    except ValueError as e:
        # Handle duplicate entries if they exist
        print(
            f"Warning: Duplicate entries found in score_df. Aggregating by mean. Error: {e}")
        factor_wide = score_df.pivot_table(
            index='ts', columns='symbol', values=factor_name, aggfunc='mean')

    # 2. Construct Price & Return Matrices (Wide)
    # We extract 'futures_close' from the data_dict to match the score timestamps
    prices_dict = {}
    for sym, df in data_dict.items():
        if not df.empty and 'futures_close' in df.columns:
            # Ensure index is datetime for alignment
            temp_df = df.set_index('ts') if 'ts' in df.columns else df
            # If index is not datetime, try to convert
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                # Assuming 'ts' was the column we just set, usually it's adequate.
                # But if ts is int (ms), we might need conversion if score_df uses int.
                # Ideally both use the same type. Let's assume alignment is possible.
                pass
            prices_dict[sym] = temp_df['futures_close']

    prices_wide = pd.DataFrame(prices_dict)

    # Align indices: factor_wide uses the backtest timestamps.
    # We need prices at those timestamps.
    # We use reindex(method='ffill') to get the latest price at each decision point
    prices_aligned = prices_wide.reindex(factor_wide.index, method='ffill')

    # 3. Calculate Forward Returns
    # We want the return from t to t+1 (the period AFTER the score was observed)
    # shift(-1) brings the return at t+1 back to row t.
    # We calculate returns based on the ALIGNED prices (the backtest steps)
    forward_returns = prices_aligned.pct_change().shift(-1)

    # 4. Quantile Bucket Analysis
    # Rank assets cross-sectionally (axis=1) at each timestamp
    ranks = factor_wide.rank(axis=1, pct=True)

    stats_list = []
    plt.figure(figsize=(12, 7))

    # Use a colormap
    colors = plt.cm.RdYlGn(np.linspace(0, 1, quantiles))

    for q in range(quantiles):
        lower_bound = q / quantiles
        upper_bound = (q + 1) / quantiles

        # Create mask for assets falling into this quantile
        # Use generic masking to handle floating point edges
        if q == 0:
            mask = (ranks >= lower_bound) & (ranks <= upper_bound)
        else:
            mask = (ranks > lower_bound) & (ranks <= upper_bound)

        # Select returns where the asset was in this quantile
        # We take the mean across all assets in the bucket for that day (Equal Weighted)
        bucket_daily_rets = forward_returns[mask].mean(axis=1).fillna(0.0)

        # Calculate Cumulative Return
        cum_ret = (1 + bucket_daily_rets).cumprod()

        # Plot
        label = f"Q{q+1} ({int(lower_bound*100)}%-{int(upper_bound*100)}%)"
        plt.plot(cum_ret.index, cum_ret, label=label,
                 color=colors[q], linewidth=2)

        # Calculate Stats
        ann_ret = bucket_daily_rets.mean() * 365
        ann_vol = bucket_daily_rets.std() * (365**0.5)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0.0

        stats_list.append({
            "Quantile": f"Q{q+1}",
            "Ann Return": f"{ann_ret:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Vol": f"{ann_vol:.2%}"
        })

    plt.title(
        f"Forward Performance by {factor_name} Quantile (Equal Weighted)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return ($1 Invested)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save Plot
    save_path = os.path.join(
        report_dir, f"quantile_analysis_{factor_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Chart saved to: {save_path}")

    # Print Stats Table
    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False))
    print("-" * 50)


def analyze_factor_quantiles(score_df: pd.DataFrame, factor_name: str, quantiles: int = 5, report_dir: str = "."):
    """
    Groups *TRADED* assets into quantiles based on 'factor_name' and plots their 
    FORWARD PnL CONTRIBUTION.
    """
    # Check required columns
    required = [factor_name, 'position_qty', 'close_price', 'symbol', 'ts']
    if score_df.empty or not all(col in score_df.columns for col in required):
        print(
            f"\n[Quantile Analysis] Skipping {factor_name}: Missing required columns.")
        return

    print(
        f"\n--- Running Traded-Only Quantile Analysis for: {factor_name} ---")

    # 1. Pivot Data to Wide Format
    def pivot_col(col):
        try:
            return score_df.pivot(index='ts', columns='symbol', values=col)
        except ValueError:
            return score_df.pivot_table(index='ts', columns='symbol', values=col, aggfunc='mean')

    factor_wide = pivot_col(factor_name).abs()
    qty_wide = pivot_col('position_qty').fillna(0.0)
    price_wide = pivot_col('close_price')
    result = (factor_wide.notna() & (factor_wide != 0)).sum(axis=1)

    # --- V-- NEW: FILTER MASK --V ---
    # Identify assets that were actually held (Long or Short)
    is_traded = qty_wide != 0

    # Mask the factor matrix.
    # Values where is_traded is False become NaN.
    # Pandas rank() ignores NaNs, so we effectively rank ONLY the traded subset.
    factor_wide_traded = factor_wide.where(is_traded)
    # --------------------------------

    # 2. Calculate Forward PnL per Asset
    price_diff = price_wide.diff().shift(-1)
    asset_forward_pnl = qty_wide * price_diff

    # 3. Quantile Bucket Analysis
    # Rank only the survivors
    ranks = factor_wide_traded.rank(axis=1, pct=True)

    stats_list = []
    plt.figure(figsize=(12, 7))

    colors = plt.cm.RdYlGn(np.linspace(0, 1, quantiles))

    for q in range(quantiles):
        lower_bound = q / quantiles
        upper_bound = (q + 1) / quantiles

        if q == 0:
            mask = (ranks >= lower_bound) & (ranks <= upper_bound)
        else:
            mask = (ranks > lower_bound) & (ranks <= upper_bound)

        # Select PnL of assets in this quantile
        bucket_daily_pnl = asset_forward_pnl[mask].sum(axis=1).fillna(0.0)

        # Cumulative PnL
        cum_pnl = bucket_daily_pnl.cumsum()

        label = f"Q{q+1} ({int(lower_bound*100)}%-{int(upper_bound*100)}%)"
        plt.plot(cum_pnl.index, cum_pnl, label=label,
                 color=colors[q], linewidth=2)

        total_pnl = bucket_daily_pnl.sum()
        mean_pnl = bucket_daily_pnl.mean()
        std_pnl = bucket_daily_pnl.std()
        sharpe = (mean_pnl / std_pnl * (365**0.5)) if std_pnl != 0 else 0.0

        stats_list.append({
            "Quantile": f"Q{q+1}",
            "Total PnL": f"${total_pnl:,.0f}",
            "Daily Mean": f"${mean_pnl:.2f}",
            "PnL Sharpe": f"{sharpe:.2f}"
        })

    plt.title(f"Cumulative PnL by {factor_name} Quantile (Traded Assets Only)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (USDT)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(
        report_dir, f"quantile_pnl_traded_only_{factor_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Chart saved to: {save_path}")

    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False))
    print("-" * 50)


def generate_predictive_regime_analysis(equity_curve: pd.DataFrame):
    """
    Analyzes performance based on FORWARD PnL attribution.
    It attributes Tomorrow's PnL (t+1) to Today's Regime (t).
    """
    if equity_curve.empty:
        return

    df = equity_curve.copy()

    # Calculate Forward PnL (Equity[t+1] - Equity[t])
    # We assign this value to row [t]
    df['forward_pnl'] = df['equity'].shift(-1) - df['equity']

    regime_cols = ['volatility_regime', 'trend_regime', 'skew_regime']

    print("\n\n==== Predictive Regime Analysis (Forward Return Attribution) ====")
    print("Attributes PnL of Day (t+1) to the Regime observed on Day (t)")

    for col in regime_cols:
        if col not in df.columns:
            continue

        print(f"\n--- Predictive Performance by {col} ---")

        # Group by current day's regime, aggregate forward PnL
        stats = df.dropna().groupby(col)['forward_pnl'].agg(
            ['sum', 'mean', 'std', 'count'])

        stats['Sharpe'] = (stats['mean'] / stats['std']) * (365**0.5)

        win_rate = df.dropna().groupby(
            col)['forward_pnl'].apply(lambda x: (x > 0).mean())
        stats['Win Rate'] = win_rate.map('{:.2%}'.format)

        stats['sum'] = stats['sum'].map('${:,.2f}'.format)
        stats['mean'] = stats['mean'].map('${:,.2f}'.format)
        stats['std'] = stats['std'].map('${:,.2f}'.format)
        stats['Sharpe'] = stats['Sharpe'].map('{:.2f}'.format)

        print(stats)
