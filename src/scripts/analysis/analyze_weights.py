"""
Weight Analysis for Momentum + Reversal Strategies.

Loads the weight parquets saved by run_vectorized_backtest and produces:
  1. Weight distribution   — histogram / KDE per strategy
  2. Coverage stability    — active asset count, long/short counts over time
  3. Exposure metrics      — gross/net exposure, turnover, concentration (HHI)
  4. Strategy overlap      — agreement, conflict, correlation between strategies
  5. Per-symbol breakdown  — average weight, activity rate, direction bias

Outputs:
  ./reports/strategies/{run_id}/weight_analysis/
      distribution.png
      coverage.png
      exposure.png
      overlap.png
      per_symbol.png
      summary.csv          (per-symbol table)
      weight_stats.txt     (scalar summary)

Usage:
    python -m src.scripts.analyze_weights --run_id batch_v1
"""
import argparse
import os
import textwrap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# ── helpers ──────────────────────────────────────────────────────────────────


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def _ensure(d): os.makedirs(d, exist_ok=True); return d


def _savefig(fig, path, title=None):
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Section 1: Weight Distribution ───────────────────────────────────────────

def plot_distribution(strategies: dict[str, pd.DataFrame], out_dir: str):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = {"momentum": "#2196F3", "reversal": "#FF5722"}
    default_colors = list(colors.values()) + ["#4CAF50", "#9C27B0"]

    # 1a. Histogram of all non-zero weights per strategy
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (name, df) in enumerate(strategies.items()):
        vals = df.values.flatten()
        vals = vals[vals != 0]
        color = colors.get(name, default_colors[i % len(default_colors)])
        ax1.hist(vals, bins=60, alpha=0.6,
                 color=color, label=name, density=True)
    ax1.axvline(0, color="black", lw=0.8, linestyle="--")
    ax1.set_xlabel("Weight")
    ax1.set_ylabel("Density")
    ax1.set_title("Weight Distribution (non-zero)")
    ax1.legend()

    # 1b. Long vs Short weight split
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(strategies))
    long_means, short_means = [], []
    for name, df in strategies.items():
        v = df.values.flatten()
        long_means.append(v[v > 0].mean() if (v > 0).any() else 0)
        short_means.append(v[v < 0].mean() if (v < 0).any() else 0)
    bars_l = ax2.bar(x_pos - 0.2, long_means,  0.35,
                     label="Long avg",  color="#4CAF50", alpha=0.8)
    bars_s = ax2.bar(x_pos + 0.2, short_means, 0.35,
                     label="Short avg", color="#F44336", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(list(strategies.keys()))
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Mean Weight")
    ax2.set_title("Avg Long / Short Weight")
    ax2.legend()

    # 1c. CDF of |weight|
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (name, df) in enumerate(strategies.items()):
        abs_vals = np.abs(df.values.flatten())
        abs_vals = np.sort(abs_vals[abs_vals > 0])
        color = colors.get(name, default_colors[i % len(default_colors)])
        ax3.plot(abs_vals, np.linspace(
            0, 1, len(abs_vals)), color=color, label=name)
    ax3.set_xlabel("|Weight|")
    ax3.set_ylabel("CDF")
    ax3.set_title("CDF of Absolute Weights")
    ax3.legend()
    ax3.yaxis.set_major_formatter(PercentFormatter(1))

    # 1d. Weight magnitude over time (rolling mean of abs weights)
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (name, df) in enumerate(strategies.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        mean_abs = df.abs().mean(axis=1).rolling(10).mean()
        ax4.plot(mean_abs.index, mean_abs.values,
                 color=color, label=name, lw=1.2)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Mean |Weight|")
    ax4.set_title("Rolling Mean Absolute Weight (10-day)")
    ax4.legend()
    fig.autofmt_xdate()

    _savefig(fig, os.path.join(out_dir, "distribution.png"),
             "Weight Distribution")


# ── Section 2: Coverage Stability ────────────────────────────────────────────

def plot_coverage(strategies: dict[str, pd.DataFrame], out_dir: str):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = {"momentum": "#2196F3", "reversal": "#FF5722"}
    default_colors = list(colors.values()) + ["#4CAF50", "#9C27B0"]

    # 2a. Active asset count over time
    ax1 = fig.add_subplot(gs[0, :])
    for i, (name, df) in enumerate(strategies.items()):
        active = (df != 0).sum(axis=1)
        color = colors.get(name, default_colors[i % len(default_colors)])
        ax1.plot(active.index, active.values, lw=1.2,
                 label=f"{name} (active)", color=color)
        ax1.fill_between(active.index, active.values, alpha=0.1, color=color)
    ax1.set_ylabel("# Active Assets")
    ax1.set_title("Active Asset Count Over Time")
    ax1.legend()
    fig.autofmt_xdate()

    # 2b. Long vs short count per strategy
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax, (name, df) in zip([ax2, ax3], strategies.items()):
        longs = (df > 0).sum(axis=1)
        shorts = (df < 0).sum(axis=1)
        color = colors.get(name, default_colors[0])
        ax.stackplot(df.index, longs.values, -shorts.values,
                     labels=["Long", "Short"],
                     colors=["#4CAF50", "#F44336"], alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{name}: Long / Short Count")
        ax.set_ylabel("# Assets")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()

    _savefig(fig, os.path.join(out_dir, "coverage.png"), "Coverage Stability")


# ── Section 3: Exposure Metrics ──────────────────────────────────────────────

def _hhi(row: pd.Series) -> float:
    """Herfindahl-Hirschman Index on absolute weights (0=flat, 1=fully concentrated)."""
    w = row.abs()
    s = w.sum()

    if s == 0:
        return 0.0
    w_norm = w / s
    return float((w_norm ** 2).sum())


def plot_exposure(strategies: dict[str, pd.DataFrame], out_dir: str):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.3)

    colors = {"momentum": "#2196F3", "reversal": "#FF5722"}
    default_colors = list(colors.values()) + ["#4CAF50", "#9C27B0"]

    # 3a. Gross exposure
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (name, df) in enumerate(strategies.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        gross = df.abs().sum(axis=1)
        ax1.plot(gross.index, gross.values, lw=1.2, label=name, color=color)
    ax1.set_ylabel("Sum |Weight|")
    ax1.set_title("Gross Exposure Over Time")
    ax1.legend()

    # 3b. Net exposure
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, df) in enumerate(strategies.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        net = df.sum(axis=1)
        ax2.plot(net.index, net.values, lw=1.2, label=name, color=color)
    ax2.axhline(0, color="black", lw=0.5, linestyle="--")
    ax2.set_ylabel("Sum Weight")
    ax2.set_title("Net Exposure Over Time")
    ax2.legend()

    # 3c. Daily turnover
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (name, df) in enumerate(strategies.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        turnover = df.diff().abs().sum(axis=1).rolling(10).mean()
        ax3.plot(turnover.index, turnover.values, lw=1.2,
                 label=f"{name} (10d avg)", color=color)
    ax3.set_ylabel("Sum |ΔWeight|")
    ax3.set_title("Turnover (10-day Rolling Avg)")
    ax3.legend()

    # 3d. HHI concentration
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (name, df) in enumerate(strategies.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        hhi = df.apply(_hhi, axis=1)
        ax4.plot(hhi.index, hhi.values, lw=1.2, label=name, color=color)
    ax4.set_ylabel("HHI (0=diverse, 1=concentrated)")
    ax4.set_title("Concentration (HHI) Over Time")
    ax4.legend()

    # 3e. Long/short gross ratio
    ax5 = fig.add_subplot(gs[2, :])
    for i, (name, df) in enumerate(strategies.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        long_gross = df.clip(lower=0).sum(axis=1)
        short_gross = df.clip(upper=0).abs().sum(axis=1)
        ratio = long_gross / (short_gross + 1e-12)
        ratio_smooth = ratio.rolling(10).median().clip(0, 5)
        ax5.plot(ratio_smooth.index, ratio_smooth.values,
                 lw=1.2, label=name, color=color)
    ax5.axhline(1.0, color="black", lw=0.5, linestyle="--", label="L/S = 1")
    ax5.set_ylabel("Long Gross / Short Gross")
    ax5.set_title("Long/Short Gross Ratio (10-day Rolling Median)")
    ax5.legend()
    fig.autofmt_xdate()

    _savefig(fig, os.path.join(out_dir, "exposure.png"), "Exposure Metrics")


# ── Section 4: Strategy Overlap ──────────────────────────────────────────────

def plot_overlap(strategies: dict[str, pd.DataFrame], out_dir: str):
    if len(strategies) < 2:
        print("  Skipping overlap (need >= 2 strategies).")
        return

    names = list(strategies.keys())
    dfs = list(strategies.values())

    # Align to common index/columns
    all_ts = dfs[0].index.union(dfs[1].index)
    all_syms = dfs[0].columns.union(dfs[1].columns)
    a = dfs[0].reindex(index=all_ts, columns=all_syms).fillna(0.0)
    b = dfs[1].reindex(index=all_ts, columns=all_syms).fillna(0.0)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.3)

    # 4a. Overlap categories over time
    ax1 = fig.add_subplot(gs[0, :])
    a_active = a != 0
    b_active = b != 0
    both_active = (a_active & b_active).sum(axis=1)
    only_a = (a_active & ~b_active).sum(axis=1)
    only_b = (~a_active & b_active).sum(axis=1)
    ax1.stackplot(all_ts,
                  both_active.values, only_a.values, only_b.values,
                  labels=[f"Both active",
                          f"{names[0]} only", f"{names[1]} only"],
                  colors=["#9C27B0", "#2196F3", "#FF5722"], alpha=0.75)
    ax1.set_ylabel("# Symbols")
    ax1.set_title("Universe Overlap Over Time")
    ax1.legend(loc="upper left")

    # 4b. Agreement vs conflict (when both active)
    ax2 = fig.add_subplot(gs[1, 0])
    same_dir = ((a > 0) & (b > 0) | (a < 0) & (b < 0)) & a_active & b_active
    opposite_dir = ((a > 0) & (b < 0) | (a < 0) &
                    (b > 0)) & a_active & b_active
    agree_count = same_dir.sum(axis=1)
    conflict_count = opposite_dir.sum(axis=1)
    ax2.plot(all_ts, agree_count.rolling(10).mean().values,
             lw=1.2, color="#4CAF50", label="Agree (10d avg)")
    ax2.plot(all_ts, conflict_count.rolling(10).mean().values,
             lw=1.2, color="#F44336", label="Conflict (10d avg)")
    ax2.set_ylabel("# Symbols")
    ax2.set_title("Agreement vs Conflict")
    ax2.legend()

    # 4c. Per-symbol agreement rate (bar — top 30 most active)
    ax3 = fig.add_subplot(gs[1, 1])
    sym_agree = same_dir.sum(axis=0)
    sym_total = (a_active & b_active).sum(axis=0)
    agree_rate = (sym_agree / sym_total.replace(0, np.nan)
                  ).dropna().sort_values(ascending=False)
    # Top 30 by activity
    top_syms = sym_total.sort_values(ascending=False).head(30).index
    plot_rate = agree_rate.reindex(top_syms).dropna()
    colors_bar = ["#4CAF50" if v >=
                  0.5 else "#F44336" for v in plot_rate.values]
    ax3.barh(plot_rate.index[::-1], plot_rate.values[::-1],
             color=colors_bar[::-1], alpha=0.8)
    ax3.axvline(0.5, color="black", lw=0.8, linestyle="--")
    ax3.set_xlabel("Agreement Rate")
    ax3.set_title("Per-Symbol Direction Agreement\n(top 30 by co-activity)")
    ax3.xaxis.set_major_formatter(PercentFormatter(1))

    fig.autofmt_xdate()
    _savefig(fig, os.path.join(out_dir, "overlap.png"),
             "Strategy Overlap Analysis")


# ── Section 5: Per-Symbol Breakdown ──────────────────────────────────────────

def plot_per_symbol(strategies: dict[str, pd.DataFrame], out_dir: str) -> pd.DataFrame:
    rows = []
    for name, df in strategies.items():
        active_days = (df != 0).sum(axis=0)
        total_days = len(df)
        activity = active_days / total_days
        mean_w = df.where(df != 0).mean(axis=0)
        mean_abs_w = df.abs().where(df != 0).mean(axis=0)
        long_rate = (df > 0).sum(axis=0) / (active_days + 1e-12)
        for sym in df.columns:
            rows.append({
                "strategy":      name,
                "symbol":        sym,
                "activity_rate": round(float(activity[sym]), 4),
                "mean_weight":   round(float(mean_w[sym]), 6) if not np.isnan(mean_w[sym]) else 0.0,
                "mean_abs_w":    round(float(mean_abs_w[sym]), 6) if not np.isnan(mean_abs_w[sym]) else 0.0,
                "long_rate":     round(float(long_rate[sym]), 4),
            })
    summary = pd.DataFrame(rows)

    # Plot top-30 symbols by activity per strategy
    n_strats = len(strategies)
    fig, axes = plt.subplots(n_strats, 2, figsize=(14, 5 * n_strats))
    if n_strats == 1:
        axes = [axes]

    for ax_row, (name, df) in zip(axes, strategies.items()):
        sub = summary[summary["strategy"] == name].sort_values(
            "activity_rate", ascending=False).head(30)

        # Activity rate
        ax_row[0].barh(sub["symbol"][::-1], sub["activity_rate"]
                       [::-1], color="#2196F3", alpha=0.8)
        ax_row[0].set_xlabel(
            "Activity Rate (fraction of days with non-zero weight)")
        ax_row[0].set_title(f"{name}: Top 30 by Activity")
        ax_row[0].xaxis.set_major_formatter(PercentFormatter(1))

        # Mean weight with color by direction
        colors_bar = ["#4CAF50" if v >=
                      0 else "#F44336" for v in sub["mean_weight"][::-1]]
        ax_row[1].barh(sub["symbol"][::-1], sub["mean_weight"]
                       [::-1], color=colors_bar, alpha=0.8)
        ax_row[1].axvline(0, color="black", lw=0.5)
        ax_row[1].set_xlabel("Mean Weight (when active)")
        ax_row[1].set_title(f"{name}: Direction Bias")

    _savefig(fig, os.path.join(out_dir, "per_symbol.png"),
             "Per-Symbol Breakdown")
    return summary


# ── Scalar summary text ───────────────────────────────────────────────────────

def write_text_summary(strategies: dict[str, pd.DataFrame], out_path: str):
    lines = []
    sep = "─" * 55

    for name, df in strategies.items():
        active = (df != 0)
        daily_active = active.sum(axis=1)
        gross = df.abs().sum(axis=1)
        net = df.sum(axis=1)
        turnover = df.diff().abs().sum(axis=1)
        hhi_series = df.apply(_hhi, axis=1)
        non_zero = df.values.flatten()
        non_zero = non_zero[non_zero != 0]

        lines += [
            sep,
            f"  Strategy : {name}",
            sep,
            f"  Dates          : {df.index[0].date()} → {df.index[-1].date()}",
            f"  Trading days   : {len(df)}",
            f"  Universe size  : {df.shape[1]} symbols",
            "",
            f"  Active assets  : {daily_active.mean():.1f} avg / day  "
            f"(min {daily_active.min()}, max {daily_active.max()})",
            f"  Long count     : {(df > 0).sum(axis=1).mean():.1f} avg",
            f"  Short count    : {(df < 0).sum(axis=1).mean():.1f} avg",
            "",
            f"  Gross exposure : {gross.mean():.3f} avg  (max {gross.max():.3f})",
            f"  Net exposure   : {net.mean():+.3f} avg  (std {net.std():.3f})",
            "",
            f"  Avg daily turnover : {turnover.mean():.4f}",
            f"  Avg HHI            : {hhi_series.mean():.4f}  "
            f"(1=full concentration)",
            "",
            f"  Weight stats (non-zero):",
            f"    min  : {non_zero.min():.4f}",
            f"    max  : {non_zero.max():.4f}",
            f"    mean : {non_zero.mean():.4f}",
            f"    std  : {non_zero.std():.4f}",
            f"    p5   : {np.percentile(non_zero, 5):.4f}",
            f"    p95  : {np.percentile(non_zero, 95):.4f}",
            "",
        ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyse strategy weight files from a backtest run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Expects weight parquets at:
              ./reports/strategies/{run_id}/momentum.parquet
              ./reports/strategies/{run_id}/reversal.parquet
              (any *.parquet files in the directory are loaded automatically)
        """),
    )
    ap.add_argument("--run_id",   required=True,
                    help="Run ID (sub-folder under ./reports/strategies/)")
    ap.add_argument("--strategies", nargs="*",
                    help="Names to load (default: momentum reversal). Must match parquet filenames without extension.")
    args = ap.parse_args()

    base_dir = f"./reports/strategies/{args.run_id}"
    out_dir = _ensure(os.path.join(base_dir, "weight_analysis"))

    # Discover weight files
    strategy_names = args.strategies or ["momentum", "reversal"]
    strategies: dict[str, pd.DataFrame] = {}

    for name in strategy_names:
        path = os.path.join(base_dir, f"{name}.parquet")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found.")
            continue
        strategies[name] = _load(path)
        print(f"  Loaded {name}: {strategies[name].shape}")

    if not strategies:
        print("No weight files found. Run the backtest scripts first.")
        return

    print(f"\nWriting analysis to: {out_dir}\n")

    print("1/5  Weight distribution ...")
    plot_distribution(strategies, out_dir)

    print("2/5  Coverage stability ...")
    plot_coverage(strategies, out_dir)

    print("3/5  Exposure metrics ...")
    plot_exposure(strategies, out_dir)

    print("4/5  Strategy overlap ...")
    plot_overlap(strategies, out_dir)

    print("5/5  Per-symbol breakdown ...")
    symbol_summary = plot_per_symbol(strategies, out_dir)
    csv_path = os.path.join(out_dir, "summary.csv")
    symbol_summary.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")

    write_text_summary(strategies, os.path.join(out_dir, "weight_stats.txt"))

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
