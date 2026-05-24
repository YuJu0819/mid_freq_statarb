"""
Factor profile search engine — pull EVERY piece of evidence about a single
factor across a training run (global model + every expert), in one place.

Surfaces:
  • Main-effect importance per model: mean, std across folds, rank, coverage.
  • Interaction importance: every pair containing the factor, per model.
  • Per-fold timeline of the main-effect importance (gives a visual on
    whether the factor's usefulness is stable or regime-dependent).
  • Shape-health routing (when ebm_factor_health_compare.csv is on disk):
    routing decision per model, monotonicity, tail/core ratio, cross-bag
    variance — the exact verdicts emitted by analyze_factor_health.

Usage
-----
    # Exact name
    python -m src.scripts.search_factor --run_id production_v2 --factor volatility_30

    # Substring search — when multiple match, prints candidates and exits
    python -m src.scripts.search_factor --run_id production_v2 --factor vol

    # Top-K interactions to show (default 15)
    python -m src.scripts.search_factor --run_id production_v2 \\
        --factor liquidation_shock --top_k_interactions 20

    # Persist artefacts under reports/strategies/<run>/factor_<name>/
    python -m src.scripts.search_factor --run_id production_v2 \\
        --factor regime_score --save
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


INTERACTION_SEP = " & "
_EXPERT_RE = re.compile(r"ebm_expert_importance_regime_(.+)\.csv$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_importance_csvs(run_dir: str) -> dict[str, pd.DataFrame]:
    """
    Return {model_label: per-fold importance DataFrame (rows=folds,
    cols=terms)}. Always includes 'global' if present; one entry per
    discovered expert CSV.
    """
    out: dict[str, pd.DataFrame] = {}
    g = os.path.join(run_dir, "ebm_feature_importance.csv")
    if os.path.exists(g):
        df = pd.read_csv(g, index_col=0)
        df.index = pd.to_datetime(df.index)
        out["global"] = df
    for p in sorted(glob.glob(
            os.path.join(run_dir, "ebm_expert_importance_regime_*.csv"))):
        m = _EXPERT_RE.search(os.path.basename(p))
        if not m:
            continue
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index)
        out[f"expert_{m.group(1)}"] = df
    return out


def _resolve_factor(query: str, all_terms: set[str]) -> str:
    """
    Exact match preferred; otherwise substring match (case-insensitive).
    When multiple terms match, prints candidates and raises SystemExit.
    """
    if query in all_terms:
        return query
    lo = query.lower()
    # Restrict to main-effect names (no INTERACTION_SEP) so the matcher
    # returns the canonical feature, not an interaction term.
    mains = sorted({t for t in all_terms if INTERACTION_SEP not in t})
    matches = [t for t in mains if lo in t.lower()]
    if not matches:
        raise SystemExit(
            f"\nNo factor matched '{query}'. Try one of:\n  "
            + ", ".join(mains[:30])
            + (" …" if len(mains) > 30 else ""))
    if len(matches) > 1:
        bullets = "\n  ".join(matches)
        raise SystemExit(
            f"\n'{query}' matched {len(matches)} factors — be more specific:\n  "
            f"{bullets}")
    return matches[0]


# ---------------------------------------------------------------------------
# Per-section extractors
# ---------------------------------------------------------------------------

def main_effect_summary(
    importance: dict[str, pd.DataFrame],
    factor: str,
) -> pd.DataFrame:
    """One row per model: mean / std / rank / folds_active for `factor`."""
    rows = []
    for label, df in importance.items():
        main_cols = [c for c in df.columns if INTERACTION_SEP not in c]
        if factor not in main_cols:
            rows.append({
                "model": label, "mean": np.nan, "std": np.nan,
                "rank": np.nan, "folds_active": 0, "n_folds": len(df),
                "pct_of_top": np.nan,
            })
            continue
        s = df[factor]
        mean = float(s.mean(skipna=True))
        std = float(s.std(skipna=True))
        # Rank vs other main-effect features (1 = highest).
        means = df[main_cols].mean(axis=0, skipna=True)
        rank = int(means.rank(ascending=False, method="min").loc[factor])
        # Coverage: how many folds carry a non-zero importance.
        active = int((s.fillna(0) > 0).sum())
        # Importance relative to the top-ranked feature in that model.
        pct = float(mean / means.max()) if means.max() > 0 else np.nan
        rows.append({
            "model": label, "mean": mean, "std": std, "rank": rank,
            "folds_active": active, "n_folds": len(df),
            "pct_of_top": pct,
        })
    return pd.DataFrame(rows).set_index("model")


def interaction_partners(
    importance: dict[str, pd.DataFrame],
    factor: str,
) -> pd.DataFrame:
    """
    Returns long DataFrame: (partner, model) → mean interaction importance,
    rank among all interaction terms in that model, n_folds_active.

    A "partner" is the other side of a `factor & partner` (or
    `partner & factor`) term.
    """
    records = []
    for label, df in importance.items():
        inter_cols = [c for c in df.columns if INTERACTION_SEP in c]
        # Restrict to terms involving our factor.
        relevant = []
        for c in inter_cols:
            a, b = [x.strip() for x in c.split(INTERACTION_SEP)]
            if a == factor:
                relevant.append((c, b))
            elif b == factor:
                relevant.append((c, a))
        if not relevant:
            continue
        # Rank among ALL interaction terms in this model.
        means_all = df[inter_cols].mean(axis=0, skipna=True)
        ranks_all = means_all.rank(ascending=False, method="min")
        for term, partner in relevant:
            m = float(means_all[term])
            sd = float(df[term].std(skipna=True))
            active = int((df[term].fillna(0) > 0).sum())
            records.append({
                "model": label,
                "partner": partner,
                "interaction_term": term,
                "mean": m,
                "std": sd,
                "rank_in_interactions": int(ranks_all[term]),
                "folds_active": active,
                "n_folds": len(df),
            })
    return pd.DataFrame(records)


def shape_health_lookup(
    run_dir: str,
    factor: str,
) -> pd.DataFrame | None:
    """
    Pull per-model routing / shape metrics from a previously-generated
    `ebm_factor_health_compare.csv`. Returns None when the file isn't
    present (user hasn't run analyze_factor_health --compare_all yet).
    """
    path = os.path.join(run_dir, "ebm_factor_health_compare.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    if factor not in df.index:
        return pd.DataFrame()
    row = df.loc[factor]
    routing_cols = [c for c in df.columns if c.endswith("_routing")]
    models = [c.replace("_routing", "") for c in routing_cols]
    out = []
    for m in models:
        out.append({
            "model": m,
            "routing": row.get(f"{m}_routing"),
            "importance_rank": row.get(f"{m}_importance_rank"),
            "monotonicity": row.get(f"{m}_monotonicity"),
            "tail_core_ratio": row.get(f"{m}_tail_core_ratio"),
            "cross_bag_var": row.get(f"{m}_cross_bag_var"),
        })
    return pd.DataFrame(out).set_index("model")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_profile(
    factor: str,
    main_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    health_df: pd.DataFrame | None,
    importance: dict[str, pd.DataFrame],
    top_k: int,
) -> None:
    sep = "═" * 92
    print(f"\n{sep}")
    print(f"  FACTOR PROFILE — {factor}")
    print(sep)

    # ── Main effect ─────────────────────────────────────────────────────
    print("\n  MAIN-EFFECT IMPORTANCE")
    print("  " + "─" * 80)
    print(f"  {'model':<14s}{'mean':>10s}{'std':>10s}{'rank':>6s}"
          f"{'%of_top':>10s}{'folds_active':>14s}")
    for m, row in main_df.iterrows():
        if np.isnan(row["mean"]):
            print(f"  {m:<14s}{'(absent)':>10s}")
            continue
        cov = f"{int(row['folds_active'])}/{int(row['n_folds'])}"
        rank = int(row["rank"])
        pct = f"{row['pct_of_top']*100:.1f}%" if not np.isnan(row["pct_of_top"]) else "—"
        print(f"  {m:<14s}{row['mean']:>10.5f}{row['std']:>10.5f}"
              f"{rank:>6d}{pct:>10s}{cov:>14s}")

    # ── Interactions ───────────────────────────────────────────────────
    print(f"\n  INTERACTION PARTNERS  (top {top_k} by mean importance "
          f"across models)")
    print("  " + "─" * 80)
    if inter_df.empty:
        print("    (no interaction terms involving this factor in any model)")
    else:
        # Pivot: rows=partner, cols=model
        pivot = inter_df.pivot_table(
            index="partner", columns="model", values="mean",
            aggfunc="first")
        # Order partners by max importance across models, then by mean.
        pivot["_max"] = pivot.max(axis=1, skipna=True)
        pivot["_mean"] = pivot.mean(axis=1, skipna=True)
        pivot = pivot.sort_values(["_max", "_mean"],
                                  ascending=False).head(top_k)
        models = sorted([c for c in pivot.columns if not c.startswith("_")])
        header = f"  {'partner':<28s}" + "".join(
            f"{m:>14s}" for m in models)
        print(header)
        for partner, row in pivot.iterrows():
            cells = "".join(
                (f"{row[m]:>14.5f}" if pd.notna(row[m]) else f"{'—':>14s}")
                for m in models)
            print(f"  {partner:<28s}{cells}")
        # Also surface per-model top-3 separately so a partner that's
        # huge in one model but absent elsewhere isn't hidden by the
        # max-sort tiebreak.
        print(f"\n  PER-MODEL TOP-3 INTERACTIONS")
        for label in sorted(inter_df["model"].unique()):
            sub = (inter_df[inter_df["model"] == label]
                   .sort_values("mean", ascending=False).head(3))
            if sub.empty:
                continue
            line = "  ".join(
                f"{r.partner}={r['mean']:.5f}(#{int(r.rank_in_interactions)})"
                for _, r in sub.iterrows())
            print(f"    {label:<12s}  {line}")

    # ── Shape health ───────────────────────────────────────────────────
    print("\n  SHAPE HEALTH  (from ebm_factor_health_compare.csv)")
    print("  " + "─" * 80)
    if health_df is None:
        print("    [n/a] run `analyze_factor_health --compare_all` first to "
              "populate this section.")
    elif health_df.empty:
        print(f"    [n/a] factor '{factor}' not present in the health-compare "
              "table.")
    else:
        print(f"  {'model':<14s}{'routing':<26s}{'imp_rank':>10s}"
              f"{'monot':>10s}{'tail/core':>12s}{'cb_var':>12s}")
        for m, row in health_df.iterrows():
            def fmt(v, w, k):
                return f"{v:>{w}.{k}f}" if pd.notna(v) else f"{'—':>{w}s}"
            print(f"  {m:<14s}{str(row['routing'])[:25]:<26s}"
                  f"{fmt(row['importance_rank'], 10, 1)}"
                  f"{fmt(row['monotonicity'], 10, 3)}"
                  f"{fmt(row['tail_core_ratio'], 12, 3)}"
                  f"{fmt(row['cross_bag_var'], 12, 5)}")

    # ── Per-fold timeline ───────────────────────────────────────────────
    print("\n  PER-FOLD MAIN-EFFECT TIMELINE")
    print("  " + "─" * 80)
    series = {}
    for label, df in importance.items():
        if factor in df.columns:
            series[label] = df[factor]
    if series:
        timeline = pd.DataFrame(series).sort_index()
        # Cap rows to keep output readable.
        n_show = min(len(timeline), 12)
        idx_show = (timeline.index[::max(1, len(timeline) // n_show)]
                    .tolist()[:n_show])
        header = f"  {'fold':<14s}" + "".join(
            f"{m:>14s}" for m in timeline.columns)
        print(header)
        for ts in idx_show:
            row = timeline.loc[ts]
            cells = "".join(
                (f"{row[m]:>14.5f}" if pd.notna(row[m]) else f"{'—':>14s}")
                for m in timeline.columns)
            print(f"  {str(ts.date()):<14s}{cells}")
    print(sep)


def plot_profile(
    factor: str,
    main_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    importance: dict[str, pd.DataFrame],
    top_k: int,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30,
                           width_ratios=[1, 1.5])

    # Panel A: main-effect importance per model (bar)
    ax = fig.add_subplot(gs[0, 0])
    valid = main_df.dropna(subset=["mean"])
    if not valid.empty:
        ax.bar(range(len(valid)), valid["mean"].values,
               color="#1565C0", alpha=0.85,
               yerr=valid["std"].values, capsize=4, ecolor="#90A4AE")
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels(valid.index, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean main-effect importance")
        ax.set_title(f"Main-effect importance — {factor}", fontsize=10)
        for i, (m, r) in enumerate(valid.iterrows()):
            ax.text(i, r["mean"], f"#{int(r['rank'])}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "factor absent from every model",
                ha="center", va="center", transform=ax.transAxes)

    # Panel B: top interaction partners (grouped bar, one bar per model)
    ax = fig.add_subplot(gs[0, 1])
    if not inter_df.empty:
        pivot = inter_df.pivot_table(
            index="partner", columns="model", values="mean",
            aggfunc="first")
        pivot["_max"] = pivot.max(axis=1, skipna=True)
        pivot = pivot.sort_values("_max", ascending=False).head(top_k)
        pivot = pivot.drop(columns=["_max"])
        models = sorted(pivot.columns.tolist())
        x = np.arange(len(pivot))
        bar_w = 0.8 / max(1, len(models))
        palette = ["#1565C0", "#F57F17", "#2E7D32",
                   "#7B1FA2", "#C62828", "#00838F"]
        for j, m in enumerate(models):
            ax.bar(x + j * bar_w, pivot[m].fillna(0).values,
                   width=bar_w, label=m,
                   color=palette[j % len(palette)], alpha=0.85)
        ax.set_xticks(x + bar_w * (len(models) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean interaction importance")
        ax.set_title(f"Top {top_k} interaction partners — {factor}",
                     fontsize=10)
        ax.legend(fontsize=8, ncol=min(3, len(models)))
    else:
        ax.text(0.5, 0.5, "no interaction terms for this factor",
                ha="center", va="center", transform=ax.transAxes)

    # Panel C: per-fold timeline (full width, bottom row)
    ax = fig.add_subplot(gs[1, :])
    series = {}
    for label, df in importance.items():
        if factor in df.columns:
            series[label] = df[factor]
    if series:
        timeline = pd.DataFrame(series).sort_index()
        for col in timeline.columns:
            ax.plot(timeline.index, timeline[col].values,
                    marker="o", lw=1.4, ms=4, label=col)
        ax.set_ylabel("Main-effect importance")
        ax.set_title(f"Per-fold main-effect importance — {factor}",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    else:
        ax.text(0.5, 0.5, "no per-fold series available",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(f"Factor Profile — {factor}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--factor", required=True,
                    help="Exact factor name OR a substring (case-insensitive). "
                         "If a substring matches multiple factors the script "
                         "prints candidates and exits.")
    ap.add_argument("--top_k_interactions", type=int, default=15,
                    help="How many interaction partners to display in the "
                         "console table and the bar plot. Default 15.")
    ap.add_argument("--save", action="store_true",
                    help="Persist the report tables (main / interactions / "
                         "health / timeline) as CSVs and a 3-panel PNG under "
                         "reports/strategies/<run>/factor_<name>/.")
    args = ap.parse_args()

    run_dir = f"./reports/strategies/{args.run_id}"
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"{run_dir} missing.")

    importance = _load_importance_csvs(run_dir)
    if not importance:
        raise FileNotFoundError(
            f"No importance CSVs found in {run_dir}. Train with "
            f"`train_ebm_signal` first so ebm_feature_importance.csv "
            f"and ebm_expert_importance_regime_*.csv exist.")

    # Aggregate term universe for fuzzy matching.
    all_terms: set[str] = set()
    for df in importance.values():
        all_terms.update(df.columns)
    factor = _resolve_factor(args.factor, all_terms)
    if factor != args.factor:
        print(f"[match] '{args.factor}' → resolved to '{factor}'")

    main_df = main_effect_summary(importance, factor)
    inter_df = interaction_partners(importance, factor)
    health_df = shape_health_lookup(run_dir, factor)

    print_profile(factor, main_df, inter_df, health_df,
                  importance, args.top_k_interactions)

    if args.save:
        out_dir = os.path.join(run_dir, f"factor_{factor}")
        os.makedirs(out_dir, exist_ok=True)
        main_df.to_csv(os.path.join(out_dir, "main_effect_summary.csv"))
        if not inter_df.empty:
            inter_df.to_csv(os.path.join(out_dir, "interactions.csv"),
                            index=False)
        if health_df is not None and not health_df.empty:
            health_df.to_csv(os.path.join(out_dir, "shape_health.csv"))
        # Timeline
        series = {label: df[factor] for label, df in importance.items()
                  if factor in df.columns}
        if series:
            pd.DataFrame(series).sort_index().to_csv(
                os.path.join(out_dir, "per_fold_timeline.csv"))
        plot_profile(factor, main_df, inter_df, importance,
                     args.top_k_interactions,
                     os.path.join(out_dir, "factor_profile.png"))
        print(f"\n  Artefacts saved under {out_dir}/")


if __name__ == "__main__":
    main()
