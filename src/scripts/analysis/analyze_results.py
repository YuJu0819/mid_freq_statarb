import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from itertools import combinations
from tabulate import tabulate


def analyze_optimization_results(csv_path: str):
    """
    Comprehensive analysis of optimization results.
    - Correlations
    - 1D Parameter Sensitivity (Boxplots)
    - 2D Parameter Interactions (Heatmaps)
    - Parallel Coordinates for Top Performers
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"--- Loading Results from {csv_path} ---")
    df = pd.read_csv(csv_path)

    # 1. Separate Parameters vs Metrics
    # We assume 'sharpe', 'return_pct', 'final_equity', 'turnover' are metrics.
    # Everything else is a parameter.
    metric_cols = ['sharpe', 'return_pct', 'final_equity', 'turnover']
    # Filter only metrics that actually exist in the CSV
    metric_cols = [c for c in metric_cols if c in df.columns]

    param_cols = [c for c in df.columns if c not in metric_cols]

    print(f"Parameters found: {param_cols}")
    print(f"Metrics found:    {metric_cols}")
    print(f"Total Runs:       {len(df)}")

    # Setup Plotting Style
    sns.set_theme(style="whitegrid")

    # --- ANALYSIS 1: GLOBAL CORRELATION ---
    # Which parameters actually matter?
    print("\n[1/4] Generating Correlation Matrix...")
    plt.figure(figsize=(10, 8))

    # Calculate correlation between everything
    corr_matrix = df.corr()

    # Isolate just Params vs Metrics (easier to read)
    # We want to see: Does increasing 'half_life' increase 'sharpe'?
    param_metric_corr = corr_matrix.loc[param_cols, metric_cols]

    sns.heatmap(param_metric_corr, annot=True, cmap="coolwarm",
                center=0, fmt=".2f", linewidths=.5)
    plt.title("Correlation: Parameters vs Metrics (What drives performance?)")
    plt.tight_layout()
    plt.show()

    # --- ANALYSIS 2: 1D SENSITIVITY (BOXPLOTS) ---
    # How stable is each parameter?
    print(
        f"\n[2/4] Generating 1D Sensitivity Plots for {len(param_cols)} parameters...")

    # Create a grid of subplots
    n_params = len(param_cols)
    cols_plot = 2
    rows_plot = (n_params + 1) // 2

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(15, 5 * rows_plot))
    axes = axes.flatten()

    for i, param in enumerate(param_cols):
        # We plot distribution of Sharpe Ratio for each value of the parameter
        sns.boxplot(x=param, y='sharpe', data=df,
                    ax=axes[i], palette="viridis")
        axes[i].set_title(f"Impact of {param}")
        axes[i].set_ylabel("Sharpe Ratio")

        # Add a red line for the mean to see the trend clearly
        means = df.groupby(param)['sharpe'].mean()
        # sns.lineplot(x=range(len(means)), y=means.values, ax=axes[i], color='red', marker='o', label='Mean')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # --- ANALYSIS 3: 2D INTERACTIONS (HEATMAPS) ---
    # Do parameters work together? (e.g. Long Lookback needs Long Decay?)
    # We plot pairs that have high standard deviation (meaning they vary in the grid search)
    print("\n[3/4] Generating 2D Interaction Heatmaps...")

    # Filter for params that actually have >1 unique value
    active_params = [p for p in param_cols if df[p].nunique() > 1]

    if len(active_params) >= 2:
        param_pairs = list(combinations(active_params, 2))

        # Limit to first 6 pairs to avoid spamming 20 plots if grid is huge
        max_plots = 6
        if len(param_pairs) > max_plots:
            print(
                f"  > Too many combinations ({len(param_pairs)}). Showing top {max_plots} pairs.")
            param_pairs = param_pairs[:max_plots]

        cols_heat = 2
        rows_heat = (len(param_pairs) + 1) // 2

        fig, axes = plt.subplots(rows_heat, cols_heat,
                                 figsize=(16, 6 * rows_heat))
        axes = axes.flatten() if len(param_pairs) > 1 else [axes]

        for i, (p1, p2) in enumerate(param_pairs):
            # Pivot table: X=p1, Y=p2, Value=Mean Sharpe
            pivot = df.groupby([p1, p2])['sharpe'].mean().unstack()

            sns.heatmap(pivot, annot=True, cmap="RdYlGn",
                        fmt=".2f", ax=axes[i])
            axes[i].set_title(f"Sharpe: {p1} vs {p2}")

        # Hide empty
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # --- ANALYSIS 4: PARALLEL COORDINATES (TOP PERFORMERS) ---
    # Visualizing the "Path" of the best configurations
    print("\n[4/4] Generating Parallel Coordinates for Top 20%...")

    # Filter top 20% by Sharpe
    top_threshold = df['sharpe'].quantile(0.80)
    top_df = df[df['sharpe'] >= top_threshold].copy()

    # We need to normalize data for parallel coordinates to look good
    # Or just use the raw values if scales are comparable.
    # For now, let's just plot the raw params + sharpe using pd.plotting

    if len(top_df) > 1:
        plt.figure(figsize=(15, 6))

        # Select cols: Params + Sharpe
        cols_to_plot = active_params + ['sharpe']

        # Use pandas parallel_coordinates
        # We need a 'class' column for color. Let's bin Sharpe into quartiles of the top set
        top_df['Performance'] = pd.qcut(top_df['sharpe'], 4, labels=[
                                        "Good", "Better", "Great", "Best"])

        pd.plotting.parallel_coordinates(
            top_df[cols_to_plot + ['Performance']], 'Performance', colormap='viridis', alpha=0.8)
        plt.title(
            f"Parameter Paths of Top 20% Configs (Sharpe > {top_threshold:.2f})")
        plt.legend(loc='upper right')
        plt.show()

    # --- SUMMARY TABLE ---
    print("\n" + "="*60)
    print(f"                 TOP 10 CONFIGURATIONS ({csv_path})")
    print("="*60)

    # Sort and print
    top_10 = df.sort_values('sharpe', ascending=False).head(10)
    print(tabulate(top_10, headers='keys', tablefmt='grid',
          floatfmt=".2f", showindex=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", default="optimization_results.csv", help="Path to results CSV")
    args = parser.parse_args()

    analyze_optimization_results(args.file)
