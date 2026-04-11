"""
EBM Factor Importance Analysis — main effects vs. interaction terms.

For each feature the EBM was trained on, computes:
  - mean_main     : mean importance of the feature's main-effect term across folds
  - mean_interact : mean total importance from all pairwise interaction terms
                    that involve this feature (sum across folds, then mean)
  - combined      : mean_main + mean_interact  (overall contribution)

Quadrant classification (thresholds = per-metric medians):
  HIGH-HIGH  : high main + high interaction  → most valuable, keep
  HIGH-LOW   : high main, low  interaction   → strong standalone signal
  LOW-HIGH   : low  main, high interaction   → only valuable in combination
  LOW-LOW    : low  main, low  interaction   → candidates for removal

Outputs (in ./reports/strategies/{run_id}/):
  ebm_importance_analysis.csv   full per-feature table
  ebm_importance_analysis.png   scatter (main vs. interact) + ranked bars

Usage:
  python -m src.scripts.analyze_ebm_importance --run_id batch_v1
  python -m src.scripts.analyze_ebm_importance --run_id batch_v1 --top_n 15
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

INTERACTION_SEP = " & "   # EBM uses this in term_names_ for pairwise terms

_QUAD_COLORS = {
    "HIGH-HIGH": "#2E7D32",   # dark green
    "HIGH-LOW":  "#1565C0",   # dark blue
    "LOW-HIGH":  "#F57F17",   # amber
    "LOW-LOW":   "#C62828",   # dark red
}
_QUAD_LABELS = {
    "HIGH-HIGH": "HIGH-HIGH  (strong standalone + synergy)  → keep",
    "HIGH-LOW":  "HIGH-LOW   (strong standalone, low synergy)",
    "LOW-HIGH":  "LOW-HIGH   (weak standalone, valuable in combos)",
    "LOW-LOW":   "LOW-LOW    (weak standalone + low synergy)  → consider dropping",
}


def load_importance(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "ebm_feature_importance.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No importance file at {path}.\n"
            "Run train_ebm_signal.py first."
        )
    # rows = fold dates, cols = term names
    df = pd.read_csv(path, index_col=0)
    return df


def parse_importances(imp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split term importances into main-effect and interaction rows,
    then aggregate per feature:
      mean_main     : mean across folds of the main-effect importance
      mean_interact : mean across folds of (sum of all interaction importances
                      that include this feature)
      combined      : mean_main + mean_interact
    """
    all_terms = imp_df.columns.tolist()

    interaction_terms = [t for t in all_terms if INTERACTION_SEP in t]
    main_terms = [t for t in all_terms if INTERACTION_SEP not in t]

    # ── main effects ─────────────────────────────────────────────────────────
    main_df = imp_df[main_terms]          # shape: (n_folds, n_main_features)

    # ── interactions: per fold, sum interaction importance per feature ────────
    # For each fold (row) and each feature, sum importance of all interaction
    # terms that mention that feature.
    features = main_terms                 # same set of features

    # Fill NaN with 0: a term absent from a fold means zero importance,
    # not missing data.  NaN propagation would corrupt the per-fold sum.
    interact_imp = imp_df[interaction_terms].fillna(0.0)

    interact_records = []
    for fold_date, row in interact_imp.iterrows():
        feat_interact = {f: 0.0 for f in features}
        for term, val in row.items():
            if val == 0.0:
                continue          # fast-path: skip zero-importance terms
            fa, fb = term.split(INTERACTION_SEP)
            fa, fb = fa.strip(), fb.strip()
            if fa in feat_interact:
                feat_interact[fa] += val
            if fb in feat_interact:
                feat_interact[fb] += val
        interact_records.append(feat_interact)

    interact_df = pd.DataFrame(interact_records, index=imp_df.index)

    # ── aggregate across folds ────────────────────────────────────────────────
    result = pd.DataFrame({
        "mean_main":     main_df.mean(axis=0),
        "std_main":      main_df.std(axis=0),
        "mean_interact": interact_df[features].mean(axis=0),
        "std_interact":  interact_df[features].std(axis=0),
    })
    result["combined"] = result["mean_main"] + result["mean_interact"]
    result = result.sort_values("combined", ascending=False)
    return result, interact_df, main_df


def classify_quadrants(result: pd.DataFrame) -> pd.DataFrame:
    """Assign quadrant labels using median thresholds."""
    med_main = result["mean_main"].median()
    med_interact = result["mean_interact"].median()

    def _quad(row):
        hi_main = row["mean_main"] >= med_main
        hi_interact = row["mean_interact"] >= med_interact
        if hi_main and hi_interact:
            return "HIGH-HIGH"
        if hi_main and not hi_interact:
            return "HIGH-LOW"
        if not hi_main and hi_interact:
            return "LOW-HIGH"
        return "LOW-LOW"

    result = result.copy()
    result["quadrant"] = result.apply(_quad, axis=1)
    result["thresh_main"] = med_main
    result["thresh_interact"] = med_interact
    return result


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_analysis(result: pd.DataFrame, top_n: int, out_path: str):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Scatter: main vs interaction importance (full feature set) ─────────
    ax1 = fig.add_subplot(gs[:, 0])   # left column, full height

    med_main = result["thresh_main"].iloc[0]
    med_interact = result["thresh_interact"].iloc[0]

    # Only label the top-N features by combined importance to avoid clutter
    label_set = set(result.head(top_n).index)

    for quad, grp in result.groupby("quadrant"):
        if grp.empty:          # skip — empty scatter causes legend crash
            continue
        color = _QUAD_COLORS[quad]
        size = np.clip(grp["combined"] /
                       result["combined"].max() * 600, 30, 600)
        ax1.scatter(grp["mean_main"], grp["mean_interact"],
                    c=color, s=size, alpha=0.82, zorder=3)
        for feat, row in grp.iterrows():
            if feat not in label_set:
                continue        # skip labels for non-top features
            # Offset proportional to axis range to avoid fixed-offset overlap
            ax1.text(
                row["mean_main"], row["mean_interact"],
                feat, fontsize=7.5, color=color, fontweight="bold",
                va="bottom", ha="left",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )

    # Threshold lines
    ax1.axvline(med_main,     color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax1.axhline(med_interact, color="black", lw=0.8, linestyle="--", alpha=0.5)

    # Quadrant corner labels (text-only, no legend handle needed)
    ax1.autoscale_view()
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    pad_x = (xlim[1] - xlim[0]) * 0.02
    pad_y = (ylim[1] - ylim[0]) * 0.02
    for txt, x, y, ha, va in [
        ("HIGH-HIGH", xlim[1]-pad_x, ylim[1]-pad_y, "right", "top"),
        ("HIGH-LOW",  xlim[1]-pad_x, ylim[0]+pad_y, "right", "bottom"),
        ("LOW-HIGH",  xlim[0]+pad_x, ylim[1]-pad_y, "left",  "top"),
        ("LOW-LOW",   xlim[0]+pad_x, ylim[0]+pad_y, "left",  "bottom"),
    ]:
        ax1.text(x, y, txt, fontsize=8, color=_QUAD_COLORS[txt],
                 ha=ha, va=va, alpha=0.4, fontweight="bold")

    ax1.set_xlabel("Mean Main-Effect Importance", fontsize=10)
    ax1.set_ylabel(
        "Mean Interaction Importance (sum across all pairs)", fontsize=10)
    ax1.set_title("Feature Importance: Main vs. Interaction\n"
                  "(bubble size ∝ combined importance)", fontsize=10)

    # Use Patch handles so the legend is always renderable even when a quadrant
    # is empty (Patch never has the empty-PathCollection crash).
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_QUAD_COLORS[q], label=_QUAD_LABELS[q])
        for q in ["HIGH-HIGH", "HIGH-LOW", "LOW-HIGH", "LOW-LOW"]
    ]
    ax1.legend(handles=legend_handles, fontsize=7, loc="upper left")

    # ── 2. Top-N combined importance bar ──────────────────────────────────────
    # Sort ascending so the longest bar (rank-1) appears at the TOP of barh
    ax2 = fig.add_subplot(gs[0, 1])
    # reverse: rank-1 at last row → top of barh
    top = result.head(top_n).iloc[::-1]
    x = np.arange(len(top))
    ax2.barh(x, top["mean_main"],     height=0.6, label="Main",
             color="#1565C0", alpha=0.85)
    ax2.barh(x, top["mean_interact"], height=0.6, left=top["mean_main"],
             label="Interaction", color="#F57F17", alpha=0.85)
    ax2.set_yticks(x)
    ax2.set_yticklabels(top.index, fontsize=8)
    ax2.set_xlabel("Mean Importance")
    ax2.set_title(f"Top {top_n} Features by Combined Importance\n"
                  "(stacked: main + interaction)", fontsize=9)
    ax2.legend(fontsize=8)

    # ── 3. Bottom-N combined importance bar ───────────────────────────────────
    # Worst feature (rank-last) at bottom; slightly better ones higher up.
    ax3 = fig.add_subplot(gs[1, 1])
    bot = result.tail(top_n)              # lowest combined, ascending order
    x = np.arange(len(bot))
    ax3.barh(x, bot["mean_main"],     height=0.6, label="Main",
             color="#1565C0", alpha=0.85)
    ax3.barh(x, bot["mean_interact"], height=0.6, left=bot["mean_main"],
             label="Interaction", color="#F57F17", alpha=0.85)
    ax3.set_yticks(x)
    ax3.set_yticklabels(bot.index, fontsize=8)
    ax3.set_xlabel("Mean Importance")
    ax3.set_title(f"Bottom {top_n} Features by Combined Importance\n"
                  "(candidates for removal)", fontsize=9)
    ax3.legend(fontsize=8)

    fig.suptitle("EBM Factor Importance Analysis", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ── console report ────────────────────────────────────────────────────────────

def print_report(result: pd.DataFrame, top_n: int):
    sep = "─" * 72
    sep2 = "═" * 72
    hdr = f"  {'Feature':<30s} {'Main':>8s} {'Interact':>10s} {'Combined':>10s}  Quad"

    print(f"\n{sep2}")
    print("  EBM FACTOR IMPORTANCE ANALYSIS")
    print(sep2)
    print(f"  Median main-effect threshold   : "
          f"{result['thresh_main'].iloc[0]:.4f}")
    print(f"  Median interaction threshold   : "
          f"{result['thresh_interact'].iloc[0]:.4f}")
    print(sep2)

    for label, subset in [
        (f"TOP {top_n}  — highest combined importance  (keep these)",
         result.head(top_n)),
        (f"BOTTOM {top_n} — lowest combined importance  (review for removal)",
         result.tail(top_n).iloc[::-1]),
    ]:
        print(f"\n  {label}")
        print(sep)
        print(hdr)
        print(sep)
        for feat, row in subset.iterrows():
            q = row["quadrant"]
            color = {"HIGH-HIGH": "★★", "HIGH-LOW": "★ ",
                     "LOW-HIGH": " ★", "LOW-LOW": "  "}[q]
            print(f"  {feat:<30s} {row['mean_main']:>8.4f} "
                  f"{row['mean_interact']:>10.4f} {row['combined']:>10.4f}  "
                  f"{color} {q}")
        print(sep)

    print(f"\n  Quadrant breakdown:")
    for quad, grp in result.groupby("quadrant"):
        names = ", ".join(grp.index.tolist())
        print(f"    {quad:<12s} ({len(grp):2d} features): {names}")
    print(sep2)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyse EBM main-effect vs. interaction importances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_id", required=True,
                    help="Run ID (sub-folder under ./reports/strategies/)")
    ap.add_argument("--top_n", type=int, default=15,
                    help="Number of top/bottom features to show in bars and console")
    args = ap.parse_args()

    run_dir = f"./reports/strategies/{args.run_id}"

    print(f"Loading importance data from: {run_dir}")
    imp_df = load_importance(run_dir)
    print(f"  {len(imp_df)} folds  ×  {len(imp_df.columns)} terms")

    n_interactions = sum(1 for c in imp_df.columns if INTERACTION_SEP in c)
    n_main = len(imp_df.columns) - n_interactions
    print(f"  Main-effect terms : {n_main}")
    print(f"  Interaction terms : {n_interactions}")

    result, _, _ = parse_importances(imp_df)
    result = classify_quadrants(result)

    print_report(result, args.top_n)

    csv_path = os.path.join(run_dir, "ebm_importance_analysis.csv")
    result.to_csv(csv_path)
    print(f"\n  Full table saved → {csv_path}")

    plot_path = os.path.join(run_dir, "ebm_importance_analysis.png")
    plot_analysis(result, args.top_n, plot_path)


if __name__ == "__main__":
    main()
