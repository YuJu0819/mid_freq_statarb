"""
MoE Expert Usage Analysis.

Reconstructs the RegimeSelector from a panel and a prediction file, then
produces a full breakdown of how often each expert was the active gating
model during the OOS prediction window.

Outputs (all in ./reports/strategies/{run_id}/)
-----------------------------------------------
  moe_expert_usage.csv      date-level log: raw_regime, active_expert, switched
  moe_expert_summary.csv    per-expert aggregate stats
  moe_expert_usage.png      4-panel figure

Usage
-----
  python -m src.scripts.analyze_moe_expert_usage \\
      --panel_path ./data/ml/factor_panel_2024-01-01_2025-01-01.parquet \\
      --run_id moe_v1 \\
      --regime_col volatility_regime_enc \\
      --moe_hysteresis 3
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from ...core.utils import ensure_dir
from ...alpha.moe import RegimeSelector


# ── palette: up to 8 distinct regimes ────────────────────────────────────────
_PALETTE = [
    "#2196F3", "#FF9800", "#4CAF50", "#F44336",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
]


def _regime_color(regime_str: str, regime_order: list[str]) -> str:
    try:
        idx = regime_order.index(regime_str)
    except ValueError:
        idx = len(regime_order)
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------

def build_usage_table(
    panel: pd.DataFrame,
    pred_dates: pd.DatetimeIndex,
    regime_col: str,
    moe_hysteresis: int,
) -> pd.DataFrame:
    """
    For every date in `pred_dates`, record:
      raw_regime      — actual regime value at that date (no lag)
      active_expert   — lagged + hysteresis-smoothed regime used for gating
      switched        — True if active_expert changed vs previous date
      run_length      — consecutive days the current active_expert has been active
    """
    selector = RegimeSelector(panel, regime_col, hysteresis=moe_hysteresis)

    rows = []
    prev_expert = None
    run_len = 0

    for ts in sorted(pred_dates):
        raw = selector.get_raw_regime(ts)
        active = selector.get_regime(ts)

        switched = (active != prev_expert) and (prev_expert is not None)
        if active != prev_expert:
            run_len = 1
        else:
            run_len += 1

        rows.append({
            "date":          ts,
            "raw_regime":    raw,
            "active_expert": active,
            "switched":      switched,
            "run_length":    run_len,
        })
        prev_expert = active

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def compute_summary(usage: pd.DataFrame) -> pd.DataFrame:
    """
    Per-expert aggregate statistics.

    Columns
    -------
    days_active        total prediction days the expert was gating
    pct_active         fraction of total prediction days
    n_activations      how many times the expert became the active gating model
    avg_run_days       average consecutive days per activation
    median_run_days    median consecutive days per activation
    max_run_days       longest single activation streak
    """
    records = []
    experts = [e for e in usage["active_expert"].unique() if e is not None]

    for expert in sorted(experts):
        mask = usage["active_expert"] == expert
        sub = usage[mask]

        # Count activations = number of times run_length resets to 1
        n_activations = int((sub["run_length"] == 1).sum())

        # Collect run lengths for each activation
        run_lengths = []
        current = 0
        for rl in sub["run_length"].values:
            if rl == 1 and current > 0:
                run_lengths.append(current)
            current = rl
        if current > 0:
            run_lengths.append(current)

        records.append({
            "expert":           expert,
            "days_active":      int(mask.sum()),
            "pct_active":       mask.mean(),
            "n_activations":    n_activations,
            "avg_run_days":     float(np.mean(run_lengths)) if run_lengths else 0.0,
            "median_run_days":  float(np.median(run_lengths)) if run_lengths else 0.0,
            "max_run_days":     int(max(run_lengths)) if run_lengths else 0,
        })

    return pd.DataFrame(records).set_index("expert").sort_index()


def compute_expert_ic(
    usage: pd.DataFrame,
    predictions_wide: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
    target_horizon: int,
) -> pd.DataFrame:
    """
    For each active expert, compute the mean / std / IR of daily OOS IC.
    Returns a DataFrame indexed by expert.
    """
    wide_ret = panel.pivot(index="ts", columns="symbol", values=target_col)
    fwd_ret = wide_ret.shift(-target_horizon)

    daily_ic = {}
    for ts in predictions_wide.index:
        if ts not in fwd_ret.index or ts not in usage.index:
            continue
        pred = predictions_wide.loc[ts].dropna()
        real = fwd_ret.loc[ts].reindex(pred.index).dropna()
        common = pred.index.intersection(real.index)
        if len(common) < 5:
            continue
        ic, _ = stats.spearmanr(pred[common], real[common])
        daily_ic[ts] = {"ic": ic, "expert": usage.loc[ts, "active_expert"]}

    ic_df = pd.DataFrame(daily_ic).T
    ic_df["ic"] = ic_df["ic"].astype(float)

    records = []
    for expert, grp in ic_df.groupby("expert"):
        ic_vals = grp["ic"].dropna()
        records.append({
            "expert":    expert,
            "mean_ic":   ic_vals.mean(),
            "std_ic":    ic_vals.std(),
            "ir":        ic_vals.mean() / (ic_vals.std() + 1e-12),
            "ic_pos":    (ic_vals > 0).mean(),
            "n_days":    len(ic_vals),
        })

    return pd.DataFrame(records).set_index("expert").sort_index()


# ---------------------------------------------------------------------------

def plot_usage(
    usage: pd.DataFrame,
    summary: pd.DataFrame,
    ic_by_expert: pd.DataFrame | None,
    out_path: str,
    regime_col: str,
    moe_hysteresis: int,
):
    experts = sorted(e for e in usage["active_expert"].unique() if e is not None)
    colors = {e: _regime_color(e, experts) for e in experts}

    n_rows = 4 if ic_by_expert is not None else 3
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. Active-expert timeline (full width) ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for expert in experts:
        mask = usage["active_expert"] == expert
        ax1.scatter(
            usage.index[mask],
            [expert] * mask.sum(),
            color=colors[expert],
            s=4, linewidths=0, alpha=0.85,
        )
    # Mark transitions
    switch_dates = usage.index[usage["switched"]]
    for sd in switch_dates:
        ax1.axvline(sd, color="gray", lw=0.4, alpha=0.4)

    ax1.set_yticks(experts)
    ax1.set_ylabel(f"Active Expert\n(regime={regime_col})")
    ax1.set_title(
        f"Active Expert Timeline  |  hysteresis={moe_hysteresis}d  |  "
        f"{len(switch_dates)} switches over {len(usage)} days"
    )
    ax1.tick_params(axis="x", labelrotation=30, labelsize=8)

    # ── 2. Usage pie / bar (left) ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    days = summary["days_active"]
    bar_colors = [colors.get(e, "#999999") for e in days.index]
    bars = ax2.bar(days.index, days.values, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, days.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center", va="bottom", fontsize=8,
        )
    ax2.set_xlabel("Expert (Regime)")
    ax2.set_ylabel("Days Active")
    ax2.set_title("Days Active per Expert")

    # ── 3. Activation count + avg run (right) ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(summary))
    w = 0.35
    ax3.bar(x - w / 2, summary["n_activations"].values,
            width=w, label="# Activations",
            color=[colors.get(e, "#999") for e in summary.index],
            alpha=0.85)
    ax3b = ax3.twinx()
    ax3b.bar(x + w / 2, summary["avg_run_days"].values,
             width=w, label="Avg Run (days)",
             color=[colors.get(e, "#999") for e in summary.index],
             alpha=0.45, hatch="//")
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary.index)
    ax3.set_ylabel("# Activations", color="#333")
    ax3b.set_ylabel("Avg Run Length (days)", color="#666")
    ax3.set_title("Expert Activations & Average Run Length")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # ── 4. Run-length distribution per expert (full width) ───────────────
    ax4 = fig.add_subplot(gs[2, :])
    all_runs = []  # (expert, run_length) pairs for violin / box
    for expert in experts:
        sub = usage[usage["active_expert"] == expert]["run_length"]
        # Extract individual run lengths
        runs = []
        current = 0
        for rl in sub.values:
            if rl == 1 and current > 0:
                runs.append(current)
            current = rl
        if current > 0:
            runs.append(current)
        for r in runs:
            all_runs.append({"expert": expert, "run_length": r})

    run_df = pd.DataFrame(all_runs)
    if not run_df.empty:
        positions = list(range(len(experts)))
        data_per_expert = [
            run_df.loc[run_df["expert"] == e, "run_length"].values
            for e in experts
        ]
        # Only plot violin if enough data points
        can_violin = [len(d) >= 3 for d in data_per_expert]
        for i, (expert, data) in enumerate(zip(experts, data_per_expert)):
            if can_violin[i]:
                vp = ax4.violinplot([data], positions=[i],
                                    widths=0.6, showmedians=True)
                for pc in vp["bodies"]:
                    pc.set_facecolor(colors[expert])
                    pc.set_alpha(0.5)
                vp["cmedians"].set_color(colors[expert])
            else:
                ax4.scatter([i] * len(data), data,
                            color=colors[expert], s=30, zorder=3)

        ax4.set_xticks(positions)
        ax4.set_xticklabels(experts)
        ax4.set_ylabel("Run Length (days)")
        ax4.set_title("Expert Run-Length Distribution (each violin = one activation streak)")
        ax4.axhline(moe_hysteresis, color="gray", lw=0.8,
                    linestyle="--", label=f"hysteresis={moe_hysteresis}d")
        ax4.legend(fontsize=8)

    # ── 5. IC by expert (optional, full width) ───────────────────────────
    if ic_by_expert is not None and n_rows == 4:
        ax5 = fig.add_subplot(gs[3, :])
        ic_experts = [e for e in experts if e in ic_by_expert.index]
        mean_ics = [ic_by_expert.loc[e, "mean_ic"] for e in ic_experts]
        err = [ic_by_expert.loc[e, "std_ic"] for e in ic_experts]
        bar_c = ["#4CAF50" if v > 0 else "#F44336" for v in mean_ics]
        ax5.bar(ic_experts, mean_ics, yerr=err,
                color=bar_c, alpha=0.8, capsize=4)
        ax5.axhline(0, color="black", lw=0.6, linestyle="--")
        ax5.set_ylabel("Mean OOS Spearman IC ± 1σ")
        ax5.set_title("OOS IC When Each Expert Is Active")
        for i, (e, val) in enumerate(zip(ic_experts, mean_ics)):
            ir = ic_by_expert.loc[e, "ir"]
            ax5.text(i, val + (0.001 if val >= 0 else -0.003),
                     f"IR={ir:.2f}", ha="center", va="bottom", fontsize=8)

    # Legend patches
    patches = [
        mpatches.Patch(color=colors[e], label=f"Regime {e}")
        for e in experts
    ]
    fig.legend(handles=patches, loc="upper right",
               fontsize=8, title="Expert", framealpha=0.85)

    fig.suptitle(
        f"MoE Expert Usage Analysis  |  regime_col={regime_col}  "
        f"hysteresis={moe_hysteresis}d",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Analyze MoE expert usage from a walk-forward prediction run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel_path",    required=True,
                    help="Same panel parquet used during training")
    ap.add_argument("--run_id",        required=True,
                    help="Same run_id used during training — predictions loaded from "
                         "./reports/strategies/{run_id}/ebm_predictions.parquet")
    ap.add_argument("--regime_col",    default="volatility_regime_enc",
                    help="Regime column (must match --regime_col used in training)")
    ap.add_argument("--moe_hysteresis", type=int, default=3,
                    help="Hysteresis days (must match --moe_hysteresis used in training)")
    ap.add_argument("--target_col",    default="ret_1d",
                    choices=["ret_1d", "ret_5d", "ret_20d"],
                    help="Target column for IC calculation")
    ap.add_argument("--target_horizon", type=int, default=1,
                    help="Forward horizon for IC calculation")
    ap.add_argument("--skip_ic", action="store_true",
                    help="Skip per-expert IC calculation (faster)")
    args = ap.parse_args()

    out_dir = ensure_dir(f"./reports/strategies/{args.run_id}")
    pred_path = os.path.join(out_dir, "ebm_predictions.parquet")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading panel: {args.panel_path}")
    panel = pd.read_parquet(args.panel_path)
    panel["ts"] = pd.to_datetime(panel["ts"])
    panel = panel.sort_values(["ts", "symbol"]).reset_index(drop=True)

    if args.regime_col not in panel.columns:
        raise ValueError(
            f"--regime_col='{args.regime_col}' not found in panel. "
            f"Available columns: {panel.columns.tolist()}"
        )

    print(f"Loading predictions: {pred_path}")
    predictions_wide = pd.read_parquet(pred_path)
    predictions_wide.index = pd.to_datetime(predictions_wide.index)
    pred_dates = predictions_wide.index

    # ── Build usage table ─────────────────────────────────────────────────
    print(f"Building expert usage table "
          f"(regime_col={args.regime_col}, hysteresis={args.moe_hysteresis})...")
    usage = build_usage_table(
        panel, pred_dates, args.regime_col, args.moe_hysteresis)

    # ── Summary stats ─────────────────────────────────────────────────────
    summary = compute_summary(usage)

    # ── Per-expert IC ─────────────────────────────────────────────────────
    ic_by_expert = None
    if not args.skip_ic and args.target_col in panel.columns:
        print("Computing per-expert OOS IC...")
        ic_by_expert = compute_expert_ic(
            usage, predictions_wide, panel,
            args.target_col, args.target_horizon,
        )

    # ── Print report ──────────────────────────────────────────────────────
    total_days = len(usage)
    n_switches = int(usage["switched"].sum())

    print(f"\n{'═'*60}")
    print(f"  MoE Expert Usage Report")
    print(f"  regime_col = {args.regime_col}  |  hysteresis = {args.moe_hysteresis}d")
    print(f"  Prediction dates : {total_days}")
    print(f"  Expert switches  : {n_switches}  "
          f"(avg every {total_days / max(n_switches, 1):.1f} days)")
    print(f"{'═'*60}")

    sep = "─" * 60
    print(f"\n  {'Expert':<10} {'Days':>6} {'%Active':>8} "
          f"{'Activ.':>7} {'AvgRun':>7} {'MaxRun':>7}")
    print(f"  {sep}")
    for expert, row in summary.iterrows():
        print(f"  {str(expert):<10} "
              f"{int(row['days_active']):>6} "
              f"{row['pct_active']*100:>7.1f}% "
              f"{int(row['n_activations']):>7} "
              f"{row['avg_run_days']:>7.1f} "
              f"{int(row['max_run_days']):>7}")
    print(f"  {sep}")

    if ic_by_expert is not None and not ic_by_expert.empty:
        print(f"\n  {'Expert':<10} {'MeanIC':>8} {'StdIC':>7} "
              f"{'IR':>6} {'IC>0':>6} {'Days':>6}")
        print(f"  {sep}")
        for expert, row in ic_by_expert.iterrows():
            print(f"  {str(expert):<10} "
                  f"{row['mean_ic']:>8.4f} "
                  f"{row['std_ic']:>7.4f} "
                  f"{row['ir']:>6.2f} "
                  f"{row['ic_pos']*100:>5.0f}% "
                  f"{int(row['n_days']):>6}")
        print(f"  {sep}")

    # ── Save outputs ──────────────────────────────────────────────────────
    usage_path = os.path.join(out_dir, "moe_expert_usage.csv")
    usage.to_csv(usage_path)
    print(f"\nUsage log saved    → {usage_path}")

    summary_path = os.path.join(out_dir, "moe_expert_summary.csv")
    summary.to_csv(summary_path)
    print(f"Summary saved      → {summary_path}")

    if ic_by_expert is not None:
        ic_path = os.path.join(out_dir, "moe_expert_ic.csv")
        ic_by_expert.to_csv(ic_path)
        print(f"IC by expert saved → {ic_path}")

    plot_usage(
        usage, summary, ic_by_expert,
        out_path=os.path.join(out_dir, "moe_expert_usage.png"),
        regime_col=args.regime_col,
        moe_hysteresis=args.moe_hysteresis,
    )


if __name__ == "__main__":
    main()
