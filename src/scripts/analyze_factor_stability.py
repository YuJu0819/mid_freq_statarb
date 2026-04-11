"""
EBM Factor Importance Stability Analysis.

Measures how consistently each feature contributes across walk-forward folds.
A feature can have high mean importance but low stability (it dominates some
periods and disappears in others), or low mean importance but high stability
(a steady, reliable contributor).

Metrics computed per feature (main effects only):
  mean_imp    : mean importance across folds
  std_imp     : std of importance across folds
  cv          : coefficient of variation = std / mean  (lower → more stable)
  mean_rank   : mean rank across folds (1 = most important each fold)
  std_rank    : std of rank across folds
  pct_top_k   : fraction of folds where feature is in top-K  (default K=10)
  trend_slope : OLS slope of importance over time (positive → growing signal)
  consec_rank_corr : mean Spearman rank-correlation between consecutive folds
                     (measures whether the ranking *order* is stable, not just
                     whether individual features stay high)

Outputs (in ./reports/strategies/{run_id}/):
  factor_stability.csv     full per-feature stability table
  factor_stability.png     4-panel figure:
    1. Stability scatter  (mean importance vs CV, annotated)
    2. Importance heatmap (top-N features × folds, colour = normalised importance)
    3. Rank evolution     (top-K features rank over folds)
    4. Consecutive-fold rank correlation over time

Usage:
  python -m src.scripts.analyze_factor_stability --run_id batch_v1
  python -m src.scripts.analyze_factor_stability --run_id batch_v1 --top_n 15 --top_k 10
"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

INTERACTION_SEP = " & "


# ---------------------------------------------------------------------------
# Load + filter
# ---------------------------------------------------------------------------

def load_main_effects(run_dir: str, skip_warmup: int = 0) -> pd.DataFrame:
    """
    Load importance CSV and return only main-effect columns (no interaction terms).

    skip_warmup : discard the first N folds before computing any stability
                  metric.  With an expanding training window the earliest folds
                  train on very few dates, causing EBM to overfit onto a tiny set
                  of dominant features.  Those folds inflate early-period rank
                  stability and make the heatmap appear to "fade" over time.
                  Skipping them isolates the steady-state behaviour.

    Importances are normalised to sum-to-1 within each fold after filtering so
    that the scale difference between small early windows and large late windows
    does not dominate the stability metrics.  A feature's value then represents
    its *share* of total importance that fold rather than its absolute magnitude.
    """
    path = os.path.join(run_dir, "ebm_feature_importance.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No importance file at {path}.\n"
            "Run train_ebm_signal.py first."
        )
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    main_cols = [c for c in df.columns if INTERACTION_SEP not in c]
    df = df[main_cols].fillna(0.0)

    if skip_warmup > 0 and len(df) > skip_warmup:
        df = df.iloc[skip_warmup:]

    # Normalise to importance shares: row_sum may be 0 for empty folds → fill 0
    row_sums = df.sum(axis=1).replace(0.0, np.nan)
    df = df.div(row_sums, axis=0).fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def compute_stability(imp: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    imp : DataFrame (folds × features), main effects only, NaN→0 already filled.
    Returns a per-feature stability table sorted by mean_imp descending.
    """
    n_folds = len(imp)

    mean_imp  = imp.mean()
    std_imp   = imp.std()
    cv        = std_imp / mean_imp.replace(0, np.nan)   # avoid /0

    # Rank each fold (rank 1 = highest importance that fold)
    ranks = imp.rank(axis=1, ascending=False, method="min")
    mean_rank = ranks.mean()
    std_rank  = ranks.std()

    # Fraction of folds where feature lands in top-K
    pct_top_k = (ranks <= top_k).mean()

    # OLS trend slope of importance over fold index (normalised by n_folds)
    fold_idx = np.arange(n_folds)
    def _slope(col):
        y = imp[col].values
        if y.std() == 0:
            return 0.0
        slope, *_ = np.polyfit(fold_idx, y, 1)
        return float(slope)
    trend_slope = pd.Series({c: _slope(c) for c in imp.columns})

    # Consecutive-fold Spearman rank correlation per feature:
    # How much does this feature's importance rank *relative to the full feature
    # set* change from one fold to the next?
    # We compute this per-feature as the correlation of its rank-position
    # vector shifted by 1 (i.e., corr(ranks[:-1], ranks[1:]) per feature).
    def _consec_corr(col):
        r = ranks[col].values
        if len(r) < 3:
            return np.nan
        c, _ = stats.pearsonr(r[:-1], r[1:])
        return float(c)
    consec_rank_corr = pd.Series({c: _consec_corr(c) for c in imp.columns})

    result = pd.DataFrame({
        "mean_imp":         mean_imp,
        "std_imp":          std_imp,
        "cv":               cv,
        "mean_rank":        mean_rank,
        "std_rank":         std_rank,
        f"pct_top{top_k}":  pct_top_k,
        "trend_slope":      trend_slope,
        "consec_rank_corr": consec_rank_corr,
    })
    result = result.sort_values("mean_imp", ascending=False)
    return result, ranks


# ---------------------------------------------------------------------------
# Consecutive-fold global rank correlation (series over time)
# ---------------------------------------------------------------------------

def global_consec_rank_corr(
    imp: pd.DataFrame,
    min_share: float = 0.01,
) -> pd.Series:
    """
    Per-fold Spearman correlation between the *full feature ranking* of fold t
    and fold t-1.  Captures whether the overall importance ordering is stable.

    min_share : a feature must have importance share >= min_share in at least
                one of the two consecutive folds to be included in the
                correlation.  Features that are effectively zero in BOTH folds
                are excluded because their relative ordering is noise — including
                them dramatically deflates the Spearman when a spike fold
                (one feature = 100%) is followed by a spread fold.  The typical
                symptom is consecutive correlations of -0.2 or lower at folds
                where a single feature absorbs all model importance.
    """
    corrs = {}
    dates = imp.index.tolist()
    for i in range(1, len(dates)):
        a = imp.iloc[i - 1]
        b = imp.iloc[i]
        active = (a >= min_share) | (b >= min_share)
        if active.sum() < 3:
            corrs[dates[i]] = np.nan
            continue
        c, _ = stats.spearmanr(a[active], b[active])
        corrs[dates[i]] = c
    return pd.Series(corrs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stability(
    result: pd.DataFrame,
    ranks: pd.DataFrame,
    imp: pd.DataFrame,
    consec_corr_ts: pd.Series,
    top_n: int,
    top_k: int,
    run_id: str,
    out_path: str,
):
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    top_features = result.head(top_n).index.tolist()
    pct_col = f"pct_top{top_k}"

    # ── 1. Stability scatter: mean importance vs CV ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    cv_vals   = result["cv"].fillna(result["cv"].max())   # treat NaN CV as max (constant 0)
    mean_vals = result["mean_imp"]
    size_vals = np.clip(result[pct_col] * 600, 20, 600)

    sc = ax1.scatter(mean_vals, cv_vals, s=size_vals, c=result[pct_col],
                     cmap="RdYlGn", vmin=0, vmax=1, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax1, label=f"Fraction of folds in top-{top_k}", pad=0.01)

    # Annotate top-N features
    label_set = set(top_features)
    for feat, row in result.iterrows():
        if feat not in label_set:
            continue
        ax1.text(
            row["mean_imp"], cv_vals[feat], feat,
            fontsize=7, va="bottom", ha="left",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    med_imp = mean_vals.median()
    med_cv  = cv_vals.median()
    ax1.axvline(med_imp, color="black", lw=0.8, ls="--", alpha=0.4)
    ax1.axhline(med_cv,  color="black", lw=0.8, ls="--", alpha=0.4)

    ax1.set_xlabel("Mean Importance", fontsize=10)
    ax1.set_ylabel("Coefficient of Variation  (std / mean)", fontsize=10)
    ax1.set_title(
        f"Stability Scatter  —  run_id: {run_id}\n"
        f"Bottom-left = stable+important  |  bubble size ∝ top-{top_k} frequency",
        fontsize=9
    )

    # Quadrant corner text
    xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
    px = (xlim[1] - xlim[0]) * 0.02
    py = (ylim[1] - ylim[0]) * 0.02
    ax1.text(xlim[0]+px, ylim[0]+py, "Stable+Important",
             fontsize=7.5, color="#2E7D32", alpha=0.6, fontweight="bold")
    ax1.text(xlim[1]-px, ylim[0]+py, "Important but Volatile",
             fontsize=7.5, color="#F57F17", alpha=0.6, fontweight="bold", ha="right")
    ax1.text(xlim[0]+px, ylim[1]-py, "Stable but Weak",
             fontsize=7.5, color="#1565C0", alpha=0.6, fontweight="bold", va="top")
    ax1.text(xlim[1]-px, ylim[1]-py, "Weak+Volatile → Consider dropping",
             fontsize=7.5, color="#C62828", alpha=0.6, fontweight="bold",
             ha="right", va="top")

    # ── 2. Importance heatmap: top-N features × folds ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    heat = imp[top_features].T          # shape: (n_features, n_folds)
    # Row-normalise each feature so colours reflect within-feature variation
    row_max = heat.max(axis=1).replace(0, np.nan)
    heat_norm = heat.div(row_max, axis=0).fillna(0.0)

    im = ax2.imshow(heat_norm.values, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax2, label="Normalised importance (per feature)", pad=0.01)

    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features, fontsize=7.5)
    ax2.set_xlabel("Fold index (time →)", fontsize=9)
    ax2.set_title(f"Importance Heatmap — Top {top_n} Features\n"
                  "(row-normalised; bright = high importance that fold)", fontsize=9)

    # Sparse x-axis tick labels (fold dates)
    n_folds = len(imp)
    tick_step = max(1, n_folds // 8)
    tick_idx  = list(range(0, n_folds, tick_step))
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(
        [imp.index[i].strftime("%Y-%m") for i in tick_idx],
        rotation=30, fontsize=7, ha="right"
    )

    # ── 3. Rank evolution of top-K features ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    cmap3 = plt.cm.get_cmap("tab20", top_k)
    top_k_features = result.head(top_k).index.tolist()
    fold_dates = ranks.index

    for i, feat in enumerate(top_k_features):
        ax3.plot(fold_dates, ranks[feat].values,
                 color=cmap3(i), lw=1.4, alpha=0.85, label=feat)

    ax3.set_ylim(top_k + 0.5, 0.5)          # rank 1 at top, rank K at bottom
    ax3.set_ylabel("Rank (1 = most important)", fontsize=9)
    ax3.set_xlabel("Fold date", fontsize=9)
    ax3.set_title(
        f"Rank Evolution — Top {top_k} Features by Mean Importance\n"
        "(a flat line = perfectly stable rank)", fontsize=9
    )
    ax3.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax3.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(8))

    # ── 4. Global consecutive-fold rank correlation over time ─────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(consec_corr_ts.index, consec_corr_ts.values,
             color="#1565C0", lw=1.5, label="Consecutive-fold rank corr.")
    ax4.fill_between(consec_corr_ts.index, consec_corr_ts.values,
                     alpha=0.15, color="#1565C0")

    roll = consec_corr_ts.rolling(5, min_periods=2).mean()
    ax4.plot(roll.index, roll.values, color="#C62828", lw=1.5,
             linestyle="--", label="5-fold rolling mean")

    ax4.axhline(consec_corr_ts.mean(), color="black", lw=0.8, ls=":",
                label=f"Overall mean = {consec_corr_ts.mean():.2f}")
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_ylabel("Spearman ρ  (importance ranking)", fontsize=9)
    ax4.set_xlabel("Fold date", fontsize=9)
    ax4.set_title(
        "Consecutive-Fold Rank Stability\n"
        "(ρ→1 = ranking barely changed; ρ→0 = different factors dominate each period)",
        fontsize=9
    )
    ax4.legend(fontsize=8)
    ax4.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(8))

    fig.suptitle("EBM Factor Importance Stability", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(result: pd.DataFrame, top_k: int, consec_corr_ts: pd.Series):
    pct_col = f"pct_top{top_k}"
    sep  = "─" * 96
    sep2 = "═" * 96

    print(f"\n{sep2}")
    print("  EBM FACTOR IMPORTANCE STABILITY REPORT")
    print(sep2)
    print(f"  Folds analysed             : {len(consec_corr_ts) + 1}")
    print(f"  Overall consecutive corr.  : {consec_corr_ts.mean():.3f}  "
          f"(min={consec_corr_ts.min():.3f}  max={consec_corr_ts.max():.3f})")
    print(f"  Interpretation: ρ ≥ 0.8 → very stable rankings across periods")
    print(sep2)
    hdr = (f"  {'Feature':<28s}  {'Mean':>7s}  {'Std':>7s}  "
           f"{'CV':>6s}  {'MnRank':>7s}  {'StdRank':>8s}  "
           f"{f'Top{top_k}%':>7s}  {'Trend':>8s}  {'ConsecCorr':>10s}")
    print(hdr)
    print(sep)

    for feat, row in result.iterrows():
        trend_sym = ("↑" if row["trend_slope"] > 0 else "↓")
        cv_str    = f"{row['cv']:.3f}" if not np.isnan(row["cv"]) else "  N/A"
        print(
            f"  {feat:<28s}  {row['mean_imp']:>7.4f}  {row['std_imp']:>7.4f}  "
            f"{cv_str:>6s}  {row['mean_rank']:>7.1f}  {row['std_rank']:>8.1f}  "
            f"{row[pct_col]*100:>6.0f}%  "
            f"{row['trend_slope']:>+8.5f}{trend_sym}  "
            f"{row['consec_rank_corr']:>10.3f}"
        )
    print(sep2)

    # Stability quadrants
    cv_thresh   = result["cv"].median()
    imp_thresh  = result["mean_imp"].median()

    def _stab_quad(row):
        hi = row["mean_imp"] >= imp_thresh
        lo_cv = row["cv"] <= cv_thresh if not np.isnan(row["cv"]) else False
        if hi  and lo_cv:  return "STABLE+IMPORTANT"
        if hi  and not lo_cv: return "IMPORTANT_VOLATILE"
        if not hi and lo_cv:  return "STABLE_WEAK"
        return "WEAK_VOLATILE"

    result = result.copy()
    result["stab_quad"] = result.apply(_stab_quad, axis=1)

    print("\n  Stability quadrant breakdown:")
    for quad, grp in result.groupby("stab_quad"):
        names = ", ".join(grp.index.tolist())
        print(f"    {quad:<22s} ({len(grp):2d}): {names}")
    print(sep2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Analyse EBM feature importance stability across walk-forward folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_id",  required=True,
                    help="Run ID (sub-folder under ./reports/strategies/)")
    ap.add_argument("--top_n",   type=int, default=20,
                    help="Top-N features shown in heatmap and stability scatter labels")
    ap.add_argument("--top_k",   type=int, default=10,
                    help="K for pct_top_k metric and rank-evolution plot")
    ap.add_argument("--min_share", type=float, default=0.01,
                    help="Minimum importance share for a feature to be included in the "
                         "consecutive-fold Spearman correlation.  Features below this "
                         "threshold in BOTH consecutive folds are excluded — they are "
                         "noise when a spike fold assigns ~100%% to one feature. "
                         "Default 0.01 (1%% share).")
    ap.add_argument("--skip_warmup", type=int, default=0,
                    help="Discard the first N folds from the stability analysis. "
                         "Useful with --train_window 0 (expanding): early folds train "
                         "on very little data and overfit onto a few dominant features, "
                         "inflating early-period rank stability and making the heatmap "
                         "appear to fade. Rule of thumb: skip the folds until the "
                         "training window reaches ~252 dates (e.g. --skip_warmup 12 "
                         "with retrain_freq=21 skips ~252 training dates).")
    args = ap.parse_args()

    run_dir = f"./reports/strategies/{args.run_id}"

    print(f"Loading importance data from: {run_dir}")
    imp = load_main_effects(run_dir, skip_warmup=args.skip_warmup)
    n_skipped = args.skip_warmup
    print(f"  {len(imp)} folds  ×  {len(imp.columns)} main-effect features"
          + (f"  (first {n_skipped} warmup folds skipped)" if n_skipped else ""))

    result, ranks = compute_stability(imp, args.top_k)
    consec_corr_ts = global_consec_rank_corr(imp, min_share=args.min_share)

    print_report(result, args.top_k, consec_corr_ts)

    csv_path = os.path.join(run_dir, "factor_stability.csv")
    result.to_csv(csv_path)
    print(f"\n  Full table saved → {csv_path}")

    png_path = os.path.join(run_dir, "factor_stability.png")
    plot_stability(
        result, ranks, imp, consec_corr_ts,
        top_n=args.top_n, top_k=args.top_k,
        run_id=args.run_id, out_path=png_path,
    )


if __name__ == "__main__":
    main()
