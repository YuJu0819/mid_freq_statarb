"""
EBM Raw Prediction Inspector.

Loads the raw OOS prediction scores saved by train_ebm_signal.py (before any
beta neutralization, quantile selection, or weight construction) and produces
a detailed diagnostic report.

Panels produced
---------------
1. Cross-sectional score distribution over time (box-plot per month)
2. Raw OOS IC (Spearman between raw scores and realized fwd return) time series
3. Score histogram + QQ-plot (are scores ~normal cross-sectionally?)
4. Score heatmap (ts × symbol) — top-50 symbols by coverage
5. Score vs weight scatter on a sampled date (raw score → final weight comparison)
6. Cumulative IC (running mean) and IC Information Ratio

Optionally compares raw predictions vs beta-neutralized predictions if
--neutralized flag is set (requires the panel parquet to reconstruct the
beta-neutralized version).

Usage
-----
  python -m src.scripts.analyze_ebm_predictions \\
      --run_id batch_v1 \\
      --panel_path ./data/ml/factor_panel_2024-01-01_2025-12-31.parquet \\
      --target_col ret_1d --target_horizon 1

  # also show beta-neutralized scores on each panel:
  python -m src.scripts.analyze_ebm_predictions \\
      --run_id batch_v1 \\
      --panel_path ./data/ml/factor_panel_2024-01-01_2025-12-31.parquet \\
      --target_col ret_1d --target_horizon 1 \\
      --neutralized
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

from ...alpha.ml_utils import compute_ic, neutralize_scores


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_predictions(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "ebm_predictions.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Predictions not found at {path}.\n"
            "Run train_ebm_signal.py with --include_signals first."
        )
    return pd.read_parquet(path)


def _load_weights(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "ebm.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _cs_stats(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional statistics of raw scores."""
    records = []
    for ts, row in predictions.iterrows():
        row = row.dropna()
        if row.empty:
            continue
        records.append({
            "ts":     ts,
            "mean":   float(row.mean()),
            "std":    float(row.std()),
            "min":    float(row.min()),
            "q25":    float(row.quantile(0.25)),
            "median": float(row.median()),
            "q75":    float(row.quantile(0.75)),
            "max":    float(row.max()),
            "n":      int(row.notna().sum()),
        })
    return pd.DataFrame(records).set_index("ts")


def _raw_ic_series(predictions: pd.DataFrame,
                   panel: pd.DataFrame,
                   target_col: str,
                   horizon: int) -> pd.Series:
    """
    Spearman IC between raw prediction scores and realized forward returns.
    Uses the same logic as ml_utils.compute_ic but operates directly on the
    wide predictions matrix to avoid re-pivoting inside the utility.
    """
    return compute_ic(predictions, panel, target_col, horizon)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_cs_distribution(ax, cs_stats: pd.DataFrame):
    """Box-plot of cross-sectional score distribution over time."""
    ax.fill_between(cs_stats.index, cs_stats["q25"], cs_stats["q75"],
                    alpha=0.35, color="#1565C0", label="IQR (Q25–Q75)")
    ax.fill_between(cs_stats.index, cs_stats["min"], cs_stats["max"],
                    alpha=0.12, color="#90CAF9", label="Min–Max")
    ax.plot(cs_stats.index, cs_stats["median"],
            color="#0D47A1", lw=1.4, label="Median")
    ax.plot(cs_stats.index, cs_stats["mean"],
            color="#FF5722", lw=1.0, linestyle="--", label="Mean")
    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Cross-Sectional Score Distribution Over Time", fontsize=9)
    ax.set_ylabel("Raw EBM Score")
    ax.legend(fontsize=7)


def _plot_ic_series(ax, ic_raw: pd.Series, ic_neutral: pd.Series | None):
    roll = ic_raw.rolling(21).mean()
    ax.bar(ic_raw.index, ic_raw.values,
           color="#BBDEFB", alpha=0.55, label="Daily IC (raw)")
    ax.plot(roll.index, roll.values, color="#1565C0",
            lw=1.5, label="21d rolling IC (raw)")
    if ic_neutral is not None:
        roll_n = ic_neutral.rolling(21).mean()
        ax.plot(roll_n.index, roll_n.values, color="#F57F17",
                lw=1.5, linestyle="--", label="21d rolling IC (β-neutral)")
    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.5)
    mean_ic = ic_raw.mean()
    ir = float(mean_ic / (ic_raw.std() + 1e-12))
    ax.set_title(f"OOS IC (Raw Scores)  |  Mean={mean_ic:.4f}  IR={ir:.2f}",
                 fontsize=9)
    ax.set_ylabel("Spearman IC")
    ax.legend(fontsize=7)


def _plot_score_histogram(ax, predictions: pd.DataFrame):
    """Histogram of all raw scores pooled across dates + QQ overlay."""
    scores = predictions.values.flatten()
    scores = scores[~np.isnan(scores)]
    ax.hist(scores, bins=80, density=True, color="#1565C0",
            alpha=0.70, label="Score distribution")

    # overlay fitted normal
    mu, sigma = scores.mean(), scores.std()
    x = np.linspace(scores.min(), scores.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma),
            color="#FF5722", lw=1.5, label=f"N({mu:.3f}, {sigma:.3f}²)")

    skew = float(stats.skew(scores))
    kurt = float(stats.kurtosis(scores))
    ax.set_title(
        f"Score Distribution  |  skew={skew:.2f}  kurt={kurt:.2f}", fontsize=9)
    ax.set_xlabel("Raw EBM Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)


def _plot_score_heatmap(ax, predictions: pd.DataFrame, top_n: int = 50):
    """Heatmap: rows = time, columns = top-N symbols by non-NaN coverage."""
    coverage = predictions.notna().sum()
    top_syms = coverage.nlargest(top_n).index
    sub = predictions[top_syms].copy()

    # row-wise CS z-score for colour contrast
    mu = sub.mean(axis=1)
    sd = sub.std(axis=1).replace(0, np.nan)
    sub_z = sub.sub(mu, axis=0).div(sd, axis=0).clip(-2.5, 2.5)

    data = sub_z.values
    # downsample rows if too many dates
    if data.shape[0] > 500:
        step = data.shape[0] // 500
        data = data[::step]
        row_idx = sub_z.index[::step]
    else:
        row_idx = sub_z.index

    im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_yticks(range(len(top_syms)))
    ax.set_yticklabels(top_syms, fontsize=5)

    # X tick: sample ~8 dates
    n = len(row_idx)
    xtick_idx = np.linspace(0, n - 1, min(8, n), dtype=int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels(
        [str(row_idx[i].date()) for i in xtick_idx],
        rotation=45, ha="right", fontsize=6)
    ax.set_title(
        f"Score Heatmap (CS z-score, top-{top_n} symbols by coverage)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)


def _plot_score_vs_weight(ax, predictions: pd.DataFrame,
                          weights: pd.DataFrame | None,
                          sample_date=None):
    """Scatter of raw score vs final portfolio weight on a single date."""
    if weights is None:
        ax.text(0.5, 0.5, "ebm.parquet not found\n(weights unavailable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Score vs Weight (unavailable)", fontsize=9)
        return

    # pick the date closest to median of overlap
    common_dates = predictions.index.intersection(weights.index)
    if common_dates.empty:
        ax.text(0.5, 0.5, "No overlapping dates between predictions and weights",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Score vs Weight", fontsize=9)
        return

    if sample_date is None:
        sample_date = common_dates[len(common_dates) // 2]

    scores = predictions.loc[sample_date].dropna()
    wts    = weights.loc[sample_date].reindex(scores.index).fillna(0.0)

    colors = np.where(wts > 0, "#4CAF50",
              np.where(wts < 0, "#F44336", "#9E9E9E"))
    ax.scatter(scores.values, wts.values, c=colors, s=18, alpha=0.75)
    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Raw EBM Score")
    ax.set_ylabel("Final Weight")
    ax.set_title(
        f"Score vs Weight  [{sample_date.date()}]\n"
        "  green=long  red=short  grey=not selected",
        fontsize=8)


def _plot_cumulative_ic(ax, ic_raw: pd.Series, ic_neutral: pd.Series | None):
    """Cumulative mean IC (running) and expanding IR."""
    cum_mean = ic_raw.expanding().mean()
    cum_ir = ic_raw.expanding().mean() / (
        ic_raw.expanding().std() + 1e-12)

    ax.plot(cum_mean.index, cum_mean.values,
            color="#1565C0", lw=1.5, label="Running mean IC (raw)")
    if ic_neutral is not None:
        ax.plot(ic_neutral.expanding().mean().index,
                ic_neutral.expanding().mean().values,
                color="#F57F17", lw=1.2, linestyle="--",
                label="Running mean IC (β-neutral)")
    ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.4)

    ax2 = ax.twinx()
    ax2.plot(cum_ir.index, cum_ir.values,
             color="#9C27B0", lw=0.9, alpha=0.6, label="Expanding IR")
    ax2.set_ylabel("Expanding IR", fontsize=8, color="#9C27B0")
    ax2.tick_params(axis="y", labelcolor="#9C27B0")

    ax.set_title("Cumulative IC & Expanding IR (raw predictions)", fontsize=9)
    ax.set_ylabel("Running Mean IC")
    # Merge both legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(cs_stats: pd.DataFrame, ic_raw: pd.Series,
                   predictions: pd.DataFrame,
                   ic_neutral: pd.Series | None):
    sep = "─" * 60
    sep2 = "═" * 60
    print(f"\n{sep2}")
    print("  EBM RAW PREDICTION INSPECTION")
    print(sep2)

    scores_flat = predictions.values.flatten()
    scores_flat = scores_flat[~np.isnan(scores_flat)]

    print(f"  Prediction dates : {len(predictions)}")
    print(f"  Symbols covered  : {predictions.notna().any().sum()}")
    print(f"  Total observations: {len(scores_flat):,}")
    print(sep)
    print("  CROSS-SECTIONAL SCORE STATS (pooled)")
    print(f"    Mean    : {scores_flat.mean():+.6f}")
    print(f"    Std     : {scores_flat.std():.6f}")
    print(f"    Min     : {scores_flat.min():+.6f}")
    print(f"    Max     : {scores_flat.max():+.6f}")
    print(f"    Skewness: {stats.skew(scores_flat):+.4f}")
    print(f"    Kurtosis: {stats.kurtosis(scores_flat):+.4f}")
    print(sep)

    if not ic_raw.empty:
        mean_ic = ic_raw.mean()
        ir = float(mean_ic / (ic_raw.std() + 1e-12))
        frac_pos = float((ic_raw > 0).mean())
        print("  IC SUMMARY (raw vs realized return)")
        print(f"    Mean IC    : {mean_ic:+.5f}")
        print(f"    Std IC     : {ic_raw.std():.5f}")
        print(f"    IC IR      : {ir:+.4f}")
        print(f"    Frac > 0   : {frac_pos:.1%}")
        if ic_neutral is not None:
            m2 = ic_neutral.mean()
            ir2 = float(m2 / (ic_neutral.std() + 1e-12))
            print(f"    Mean IC (β-neutral): {m2:+.5f}  IR={ir2:+.4f}")
    else:
        print("  IC: no panel provided — skipped")
    print(sep2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Inspect raw EBM OOS predictions before post-processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_id", required=True,
                    help="Run ID (sub-folder under ./reports/strategies/)")
    ap.add_argument("--panel_path", default=None,
                    help="Path to factor panel parquet (needed for IC calculation)")
    ap.add_argument("--target_col", default="ret_1d",
                    help="Return column in the panel used for IC")
    ap.add_argument("--target_horizon", type=int, default=1,
                    help="Forward-return horizon (must match training)")
    ap.add_argument("--neutralized", action="store_true",
                    help="Also compute IC for beta-neutralized scores")
    ap.add_argument("--beta_col", default="beta_60",
                    help="Beta column in panel used for neutralization")
    ap.add_argument("--top_heatmap", type=int, default=50,
                    help="Number of symbols to show in the heatmap")
    args = ap.parse_args()

    run_dir = f"./reports/strategies/{args.run_id}"

    # ── Load predictions ─────────────────────────────────────────────────────
    print(f"Loading predictions from: {run_dir}")
    predictions = _load_predictions(run_dir)
    print(f"  Shape: {predictions.shape}  "
          f"({predictions.shape[0]} dates × {predictions.shape[1]} symbols)")
    print(f"  Date range: {predictions.index.min().date()} → "
          f"{predictions.index.max().date()}")

    weights = _load_weights(run_dir)
    if weights is not None:
        print(f"  Weights loaded: {weights.shape}")
    else:
        print("  ebm.parquet not found — score vs weight panel will be skipped")

    # ── Load panel (optional) ─────────────────────────────────────────────────
    panel = None
    if args.panel_path and os.path.exists(args.panel_path):
        print(f"Loading panel: {args.panel_path}")
        panel = pd.read_parquet(args.panel_path)
        print(f"  Panel shape: {panel.shape}")
    else:
        if args.panel_path:
            print(f"  [warn] Panel not found at {args.panel_path} — IC skipped")

    # ── Beta-neutralized predictions ─────────────────────────────────────────
    predictions_neutral = None
    if args.neutralized and panel is not None:
        print("  Computing beta-neutralized predictions…")
        try:
            predictions_neutral = neutralize_scores(
                predictions, panel, beta_col=args.beta_col)
            print("  Beta neutralization done.")
        except ValueError as e:
            print(f"  [warn] Beta neutralization failed: {e}")

    # ── Compute statistics ────────────────────────────────────────────────────
    cs_stats = _cs_stats(predictions)

    ic_raw = pd.Series(dtype=float)
    ic_neutral = None
    if panel is not None:
        print("  Computing OOS IC (raw)…")
        ic_raw = _raw_ic_series(
            predictions, panel, args.target_col, args.target_horizon)
        print(f"  IC computed for {len(ic_raw)} dates.")
        if predictions_neutral is not None:
            print("  Computing OOS IC (β-neutral)…")
            ic_neutral = _raw_ic_series(
                predictions_neutral, panel,
                args.target_col, args.target_horizon)

    _print_summary(cs_stats, ic_raw, predictions, ic_neutral)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    _plot_cs_distribution(ax1, cs_stats)

    ax2 = fig.add_subplot(gs[1, 0])
    _plot_ic_series(ax2, ic_raw, ic_neutral)

    ax3 = fig.add_subplot(gs[1, 1])
    _plot_score_histogram(ax3, predictions)

    ax4 = fig.add_subplot(gs[2, 0])
    _plot_score_heatmap(ax4, predictions, top_n=args.top_heatmap)

    ax5 = fig.add_subplot(gs[2, 1])
    _plot_score_vs_weight(ax5, predictions, weights)

    fig.suptitle(
        f"EBM Raw Prediction Inspection  [run_id={args.run_id}]",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out_path = os.path.join(run_dir, "ebm_prediction_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {out_path}")

    # ── Cumulative IC figure ──────────────────────────────────────────────────
    if not ic_raw.empty:
        fig2, ax_c = plt.subplots(figsize=(12, 5))
        _plot_cumulative_ic(ax_c, ic_raw, ic_neutral)
        fig2.tight_layout()
        ic_path = os.path.join(run_dir, "ebm_cumulative_ic.png")
        fig2.savefig(ic_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  IC plot saved → {ic_path}")

        # Save raw IC series to CSV
        ic_df = pd.DataFrame({"ic_raw": ic_raw})
        if ic_neutral is not None:
            ic_df["ic_neutral"] = ic_neutral
        ic_csv = os.path.join(run_dir, "ebm_raw_ic.csv")
        ic_df.to_csv(ic_csv)
        print(f"  IC series saved → {ic_csv}")


if __name__ == "__main__":
    main()
