"""
Analyse EBM factor shape health for a saved training run.

Reads pickled fold models from `./reports/strategies/{run_id}/ebm_models/`,
computes the four shape-geometry metrics from
`src.alpha.factor_health.FactorHealthEvaluator`, applies the routing rules,
and emits:

  ebm_factor_health.csv   per-feature aggregated metrics + routing decision
  ebm_factor_health.png   scatter of monotonicity × concentration, sized by
                          importance and coloured by routing

Usage
-----
    python -m src.scripts.analyze_factor_health --run_id production_v2
    python -m src.scripts.analyze_factor_health --run_id production_v2 --top_n 20
    python -m src.scripts.analyze_factor_health --run_id production_v2 --max_folds 5

The script auto-handles both vanilla `list[EBM]` pickles and `ResidualMoE`
pickles — when MoE, it analyses the global ensemble. Pass --expert REGIME to
analyse a specific expert ensemble instead.
"""
import argparse
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from ...alpha.factor_health import FactorHealthEvaluator


_FOLD_RE = re.compile(r"ebm_(?:moe|homoe|model)_(\d{8})\.pkl$")
_ROUTING_COLORS = {
    "GLOBAL":                  "#2E7D32",
    "EXPERT":                  "#F57F17",
    "DROP_tail":               "#C62828",
    "DROP_unstable":           "#7B1FA2",
    "DROP_tail_and_unstable":  "#3E2723",
}


def _load_fold_bags(path: str, expert: str | None) -> list:
    """
    Return the list of EBM bags for one fold.

    `ResidualMoE` pickles expose `.global_models` (list) and `.expert_dict`
    ({regime: list}). A plain pickle is already the list itself.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if hasattr(obj, "global_models") and hasattr(obj, "expert_dict"):
        if expert is None:
            return list(obj.global_models)
        if expert not in obj.expert_dict:
            raise KeyError(
                f"Expert regime {expert!r} not in fold "
                f"({list(obj.expert_dict.keys())})")
        return list(obj.expert_dict[expert])
    return list(obj)


def _load_folds(model_dir: str, max_folds: int | None,
                expert: str | None) -> dict[str, list]:
    """Scan model_dir, parse fold dates, return {fold_label: bags}."""
    paths = sorted(
        p for p in glob.glob(os.path.join(model_dir, "*.pkl"))
        if _FOLD_RE.search(os.path.basename(p))
    )
    if not paths:
        raise FileNotFoundError(
            f"No EBM fold pickles in {model_dir} — "
            "did you train with --save_models?")
    if max_folds and len(paths) > max_folds:
        # Evenly spaced subsample so we don't bias toward early/late training.
        idx = np.linspace(0, len(paths) - 1, max_folds).round().astype(int)
        paths = [paths[i] for i in idx]

    folds = {}
    for p in paths:
        m = _FOLD_RE.search(os.path.basename(p))
        fold_label = pd.to_datetime(m.group(1), format="%Y%m%d").strftime(
            "%Y-%m-%d")
        try:
            folds[fold_label] = _load_fold_bags(p, expert)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}")
    return folds


def _infer_features(folds: dict[str, list]) -> list[str]:
    """Pull feature names from the first available bag's term_names_."""
    for bags in folds.values():
        if bags:
            tn = list(bags[0].term_names_)
            return [t for t in tn if " & " not in t]
    raise RuntimeError("No bags available to infer feature names.")


def plot_health(agg: pd.DataFrame, out_path: str, top_n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: monotonicity × tail_core_ratio scatter --------------------
    ax = axes[0]
    has_imp = "importance_mean" in agg.columns
    if has_imp:
        s = (agg["importance_mean"].fillna(0)
             / max(agg["importance_mean"].max(), 1e-12) * 500 + 20)
    else:
        s = 80
    y_col = "tail_core_ratio_max"
    for routing, grp in agg.groupby("routing"):
        ax.scatter(grp["monotonicity_mean"], grp[y_col],
                   s=s.loc[grp.index] if has_imp else s,
                   c=_ROUTING_COLORS.get(routing, "#444"),
                   alpha=0.75, label=routing, edgecolor="white", linewidth=0.5)
    # Decision lines
    ax.axhline(3.0, color="red", lw=0.8, ls="--", alpha=0.5,
               label="tail/core=3.0 (DROP)")
    ax.axvline(0.3, color="orange", lw=0.6, ls=":", alpha=0.5)
    ax.axvline(-0.3, color="orange", lw=0.6, ls=":", alpha=0.5,
               label="|mono|=0.3 (EXPERT band)")

    # Label top-N by importance
    label_set = (agg["importance_mean"].nlargest(top_n).index
                 if has_imp else agg.head(top_n).index)
    for feat in label_set:
        if feat not in agg.index:
            continue
        x, y = agg.loc[feat, ["monotonicity_mean", y_col]]
        if pd.notna(x) and pd.notna(y):
            ax.text(x, y, feat, fontsize=7,
                    path_effects=[pe.withStroke(linewidth=1.5,
                                                foreground="white")])
    ax.set_xlabel("Monotonicity (density-weighted Spearman ρ)")
    ax.set_ylabel("Tail-to-Core Ratio  (max|score| in tail / max|score| in core)")
    ax.set_title(
        "Factor shape health\n"
        "(bubble size ∝ importance)" if has_imp else "Factor shape health",
        fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel B: routing breakdown bar -------------------------------------
    ax2 = axes[1]
    order = ["GLOBAL", "EXPERT", "DROP_tail", "DROP_unstable",
             "DROP_tail_and_unstable"]
    counts = agg["routing"].value_counts().reindex(order, fill_value=0)
    colors = [_ROUTING_COLORS[r] for r in order]
    ax2.bar(range(len(order)), counts.values, color=colors, alpha=0.85)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("# features")
    ax2.set_title("Routing decision breakdown")
    for i, v in enumerate(counts.values):
        ax2.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)

    fig.suptitle("EBM Factor Health Evaluator",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


_EXPERT_IMPORT_RE = re.compile(
    r"ebm_expert_importance_regime_(.+)\.csv$")


def _discover_experts(run_dir: str) -> list[str]:
    """Return the regime labels of all experts found in run_dir."""
    out = []
    for p in sorted(glob.glob(os.path.join(
            run_dir, "ebm_expert_importance_regime_*.csv"))):
        m = _EXPERT_IMPORT_RE.search(os.path.basename(p))
        if m:
            out.append(m.group(1))
    return out


def _evaluate_one_model(
    run_dir: str,
    model_dir: str,
    max_folds: int | None,
    expert: str | None,
    fhe_kwargs: dict,
    top_n_for_expert: int,
) -> pd.DataFrame:
    """
    Run the full FactorHealthEvaluator pipeline for one model
    (global or a specific expert) and return its routed `agg` DataFrame.
    Used both by single-model mode and by --compare_all.
    """
    folds = _load_folds(model_dir, max_folds, expert)
    if not folds:
        raise RuntimeError(
            f"No folds loaded for expert={expert!r}")
    feature_names = _infer_features(folds)

    if expert is not None:
        imp_csv = os.path.join(
            run_dir, f"ebm_expert_importance_regime_{expert}.csv")
    else:
        imp_csv = os.path.join(run_dir, "ebm_feature_importance.csv")
    importance_df = None
    if os.path.exists(imp_csv):
        importance_df = pd.read_csv(imp_csv, index_col=0)
        importance_df = importance_df[[
            c for c in importance_df.columns if " & " not in c]]

    fhe = FactorHealthEvaluator(feature_names=feature_names, **fhe_kwargs)
    agg = fhe.evaluate_folds(folds, importance_df=importance_df)
    agg = fhe.route(agg, top_n_for_expert=top_n_for_expert)
    return agg


def _compare_all_models(run_dir: str, model_dir: str, args) -> pd.DataFrame:
    """
    Run factor-health analysis on global + every discovered expert, and
    return a wide DataFrame (feature × {model}_routing / _importance_rank
    / _monotonicity / _tail_core_ratio / _cross_bag_var) plus consensus
    columns identifying universally-unhealthy factors.
    """
    fhe_kwargs = dict(
        drop_tail_core_ratio_thresh=args.drop_tail_core_ratio_thresh,
        drop_cross_bag_var_pctile=args.drop_cross_bag_var_pctile,
        drop_cross_bag_var_min_abs=args.drop_cross_bag_var_min_abs,
        expert_monotonicity_thresh=args.expert_monotonicity_thresh,
        core_density_thresh=args.core_density_thresh,
    )
    experts = _discover_experts(run_dir)
    models = [("global", None)] + [(f"expert_{e}", e) for e in experts]
    print(f"[compare_all] {len(models)} models: "
          f"{[m[0] for m in models]}")

    per_model = {}
    for label, expert in models:
        print(f"\n  ▸ Evaluating {label}...")
        try:
            per_model[label] = _evaluate_one_model(
                run_dir, model_dir, args.max_folds, expert,
                fhe_kwargs, args.top_n)
            print(f"    {len(per_model[label])} features evaluated.")
        except Exception as e:
            print(f"    [skip] {label}: {e}")

    if not per_model:
        raise RuntimeError("--compare_all: no models produced metrics.")

    # Build wide comparison table on the UNION of features.
    feats = sorted({f for r in per_model.values() for f in r.index})
    out = pd.DataFrame(index=feats)
    out.index.name = "feature"
    for label, agg in per_model.items():
        out[f"{label}_routing"] = agg["routing"].reindex(feats)
        if "importance_rank" in agg.columns:
            out[f"{label}_importance_rank"] = agg["importance_rank"].reindex(feats)
        out[f"{label}_monotonicity"] = agg["monotonicity_mean"].reindex(feats)
        out[f"{label}_tail_core_ratio"] = agg["tail_core_ratio_max"].reindex(feats)
        out[f"{label}_cross_bag_var"] = agg["cross_bag_variance_mean"].reindex(feats)

    routing_cols = [c for c in out.columns if c.endswith("_routing")]

    # Per-feature consensus flags.
    def _all_drop(row):
        vals = row[routing_cols].dropna().tolist()
        if not vals:
            return False
        return all(str(v).startswith("DROP") for v in vals)

    def _any_expert(row):
        vals = row[routing_cols].dropna().tolist()
        return any(v == "EXPERT" for v in vals)

    def _all_drop_or_global(row):
        vals = row[routing_cols].dropna().tolist()
        if not vals:
            return False
        return all(str(v).startswith("DROP") or v == "GLOBAL" for v in vals)

    def _n_drop(row):
        return int(sum(1 for v in row[routing_cols].dropna()
                       if str(v).startswith("DROP")))

    out["n_models_drop"] = out.apply(_n_drop, axis=1)
    out["n_models_total"] = out[routing_cols].notna().sum(axis=1)
    out["consensus_strict_drop"] = out.apply(_all_drop, axis=1)
    out["consensus_no_expert"] = ~out.apply(_any_expert, axis=1)
    out["consensus_useless_soft"] = (
        out.apply(_all_drop_or_global, axis=1)
        & out["consensus_no_expert"])
    # Mean importance rank across models — useful tiebreaker for "GLOBAL but
    # universally unimportant" candidates.
    imp_cols = [c for c in out.columns if c.endswith("_importance_rank")]
    if imp_cols:
        out["avg_importance_rank"] = out[imp_cols].mean(axis=1, skipna=True)

    return out


def _print_compare_report(cmp_df: pd.DataFrame) -> None:
    sep = "═" * 96
    routing_cols = [c for c in cmp_df.columns if c.endswith("_routing")]
    models = [c.replace("_routing", "") for c in routing_cols]

    print(f"\n{sep}")
    print(f"  CROSS-MODEL FACTOR HEALTH COMPARISON  "
          f"({len(cmp_df)} features × {len(models)} models)")
    print(sep)
    print(f"  Models: {', '.join(models)}")

    strict = cmp_df[cmp_df["consensus_strict_drop"]]
    soft = cmp_df[cmp_df["consensus_useless_soft"]
                  & ~cmp_df["consensus_strict_drop"]]

    print(f"\n  STRICT DROP (every model routed DROP_*): {len(strict)}")
    print("  " + "─" * 80)
    for feat, row in strict.iterrows():
        labels = "  ".join(
            f"{m}={str(row[m + '_routing']):>22s}" for m in models)
        print(f"    {feat:<28s}  {labels}")

    print(f"\n  SOFT USELESS (no model routed EXPERT, all DROP-or-GLOBAL): "
          f"{len(soft)}")
    print("  " + "─" * 80)
    soft_sorted = soft.sort_values(
        "avg_importance_rank" if "avg_importance_rank" in soft.columns
        else "n_models_drop", ascending=False)
    for feat, row in soft_sorted.iterrows():
        ar = (f"avg_rank={row['avg_importance_rank']:.1f}"
              if "avg_importance_rank" in row else "")
        labels = "  ".join(
            f"{m}={str(row[m + '_routing'])[:18]:>18s}" for m in models)
        print(f"    {feat:<28s}  {ar:<14s}  {labels}")

    print(f"\n  Routing-decision matrix (sample 20 features by avg rank):")
    head = (cmp_df.sort_values("avg_importance_rank",
                               ascending=False)
            if "avg_importance_rank" in cmp_df.columns
            else cmp_df.head(20))
    head = head.head(20)
    show_cols = routing_cols + ["n_models_drop"]
    print(head[show_cols].to_string())
    print(sep)


def _plot_compare(cmp_df: pd.DataFrame, out_path: str) -> None:
    """
    Categorical heatmap: feature × model coloured by routing decision.
    Features ordered so universally-DROPPED ones cluster at the top.
    """
    routing_cols = [c for c in cmp_df.columns if c.endswith("_routing")]
    models = [c.replace("_routing", "") for c in routing_cols]
    if not models:
        return

    color_idx = {
        "GLOBAL":                 0,
        "EXPERT":                 1,
        "DROP_tail":              2,
        "DROP_unstable":          3,
        "DROP_tail_and_unstable": 4,
        "nan":                    5,
    }
    palette = ["#2E7D32", "#F57F17", "#C62828",
               "#7B1FA2", "#3E2723", "#BDBDBD"]

    # Sort: most-DROPPED first, then by avg importance rank within each tier.
    sort_keys = ["n_models_drop"]
    if "avg_importance_rank" in cmp_df.columns:
        sort_keys.append("avg_importance_rank")
    df_sorted = cmp_df.sort_values(sort_keys, ascending=[False, False])

    mat = np.full((len(df_sorted), len(models)), len(palette) - 1,
                  dtype=int)
    for j, col in enumerate(routing_cols):
        for i, v in enumerate(df_sorted[col].values):
            key = str(v) if pd.notna(v) else "nan"
            mat[i, j] = color_idx.get(key, len(palette) - 1)

    cmap = plt.matplotlib.colors.ListedColormap(palette)
    fig, ax = plt.subplots(figsize=(2 + 1.4 * len(models),
                                    max(8, 0.20 * len(df_sorted))))
    ax.imshow(mat, aspect="auto", cmap=cmap,
              vmin=-0.5, vmax=len(palette) - 0.5, interpolation="nearest")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index, fontsize=7)
    ax.set_title(
        "Cross-Model Routing — features sorted by # models that DROP\n"
        "(top rows = consensus drop candidates)",
        fontsize=10)

    from matplotlib.patches import Patch
    handles = [Patch(color=palette[i], label=k)
               for k, i in color_idx.items()
               if k != "nan"]
    ax.legend(handles=handles, fontsize=8, bbox_to_anchor=(1.02, 1),
              loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison heatmap saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--top_n", type=int, default=20,
                    help="EXPERT-routing applies only to the top-N features "
                         "ranked by importance.")
    ap.add_argument("--max_folds", type=int, default=None,
                    help="Subsample at most N folds (evenly spaced) to keep "
                         "runtime manageable on long histories.")
    ap.add_argument("--expert", default=None,
                    help="When models are MoE, analyse this expert's regime "
                         "(string label, matching keys in expert_dict). "
                         "Default: analyse the global ensemble.")
    ap.add_argument("--compare_all", action="store_true",
                    help="Run the analysis on the global model AND every "
                         "discovered expert, then emit a feature×model "
                         "comparison table and a 'consensus useless' list "
                         "of factors that no model finds useful. Overrides "
                         "--expert.")
    ap.add_argument(
        "--drop_tail_core_ratio_thresh", type=float, default=3.0,
        help="DROP a factor when max|score| in low-density tail bins "
             "(density < 0.05) exceeds this multiple of max|score| in "
             "core bins. Default 3.0 per spec.")
    ap.add_argument(
        "--drop_cross_bag_var_pctile", type=float, default=0.80,
        help="Cross-bag-variance percentile threshold for DROP_unstable. "
             "Combined with --drop_cross_bag_var_min_abs (logical AND) so "
             "that on well-behaved datasets where every feature has "
             "noise-level variance, the rule does not fire on useful "
             "high-rank factors that just happen to be in the top 20% of "
             "a near-zero distribution.")
    ap.add_argument(
        "--drop_cross_bag_var_min_abs", type=float, default=1e-4,
        help="Absolute floor for the DROP_unstable rule. cbv must exceed "
             "BOTH the pctile threshold AND this floor. Set to 0.0 to "
             "recover the original purely-relative behaviour. Default 1e-4 "
             "is calibrated to typical EBM shape magnitudes O(0.01-1) — "
             "anything below is essentially numerical noise from bag "
             "bootstrap, not meaningful instability.")
    ap.add_argument(
        "--expert_monotonicity_thresh", type=float, default=0.30)
    ap.add_argument(
        "--core_density_thresh", type=float, default=0.05,
        help="PDF threshold (mass per unit x) above which a bin counts as "
             "'core'. The original spec value of 0.05 was authored for "
             "count-fraction density; we now compute true PDF density "
             "instead (count_fraction / bin_width) because interpret uses "
             "quantile-adaptive bins. In PDF units, 0.05 sits near the "
             "median PDF of a Gaussian-distributed feature, so it cleanly "
             "separates the central core from sparse outer-bin tails.")
    args = ap.parse_args()

    run_dir = f"./reports/strategies/{args.run_id}"
    model_dir = os.path.join(run_dir, "ebm_models")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"{model_dir} missing. Re-train with --save_models.")

    # ── Compare-all mode: feature × model matrix + consensus useless list ──
    if args.compare_all:
        cmp_df = _compare_all_models(run_dir, model_dir, args)
        _print_compare_report(cmp_df)

        csv_path = os.path.join(run_dir, "ebm_factor_health_compare.csv")
        cmp_df.to_csv(csv_path)
        print(f"\n  Comparison table saved → {csv_path}")

        # Persist the strict-drop and soft-useless lists as plain text for
        # easy ingestion into the feature-selection step.
        for name, df_sub in [
            ("strict_drop", cmp_df[cmp_df["consensus_strict_drop"]]),
            ("useless_soft",
             cmp_df[cmp_df["consensus_useless_soft"]
                    & ~cmp_df["consensus_strict_drop"]]),
        ]:
            txt_path = os.path.join(
                run_dir, f"ebm_factor_health_{name}.txt")
            with open(txt_path, "w") as fh:
                for feat in df_sub.index:
                    fh.write(feat + "\n")
            print(f"  {name} list saved → {txt_path}  ({len(df_sub)} features)")

        png_path = os.path.join(run_dir, "ebm_factor_health_compare.png")
        _plot_compare(cmp_df, png_path)
        return

    print(f"Loading fold pickles from {model_dir}")
    folds = _load_folds(model_dir, args.max_folds, args.expert)
    print(f"  Loaded {len(folds)} folds  "
          f"({sum(len(b) for b in folds.values())} total bags)")

    feature_names = _infer_features(folds)
    print(f"  Inferred {len(feature_names)} main-effect features.")

    # Best-effort importance series (per-fold).
    # When --expert REGIME is set we MUST read that expert's own
    # importance CSV — otherwise the EXPERT-routing rule
    # (top-N importance + low monotonicity) would rank expert features
    # by the *global* model's importance, which is meaningless because
    # the two models can disagree on which features carry weight.
    if args.expert is not None:
        imp_csv = os.path.join(
            run_dir, f"ebm_expert_importance_regime_{args.expert}.csv")
        imp_label = f"expert {args.expert}"
    else:
        imp_csv = os.path.join(run_dir, "ebm_feature_importance.csv")
        imp_label = "global"
    importance_df = None
    if os.path.exists(imp_csv):
        importance_df = pd.read_csv(imp_csv, index_col=0)
        # Restrict to main-effect cols only (drop pair-importance noise here).
        importance_df = importance_df[[
            c for c in importance_df.columns if " & " not in c]]
        print(f"  Loaded {imp_label} importance: {importance_df.shape}  "
              f"({os.path.basename(imp_csv)})")
    else:
        print(f"  [warn] {imp_csv} not found — EXPERT routing rule will "
              f"have no importance_rank to consult and every feature "
              f"defaults to GLOBAL.")

    fhe = FactorHealthEvaluator(
        feature_names=feature_names,
        drop_tail_core_ratio_thresh=args.drop_tail_core_ratio_thresh,
        drop_cross_bag_var_pctile=args.drop_cross_bag_var_pctile,
        drop_cross_bag_var_min_abs=args.drop_cross_bag_var_min_abs,
        expert_monotonicity_thresh=args.expert_monotonicity_thresh,
        core_density_thresh=args.core_density_thresh,
    )
    print("Evaluating folds...")
    agg = fhe.evaluate_folds(folds, importance_df=importance_df)
    print("Applying routing rules...")
    agg = fhe.route(agg, top_n_for_expert=args.top_n)

    # ── Console summary ────────────────────────────────────────────────────
    sep = "═" * 84
    print(f"\n{sep}")
    print(f"  FACTOR HEALTH — {len(agg)} features")
    if args.expert:
        print(f"  Analysed expert regime: {args.expert}")
    print(sep)
    cols_show = ["importance_rank", "monotonicity_mean",
                 "tail_core_ratio_max", "curvature_mean",
                 "cross_bag_variance_mean", "routing"]
    cols_show = [c for c in cols_show if c in agg.columns]
    with pd.option_context("display.width", 140,
                           "display.max_rows", 100,
                           "display.float_format",
                           lambda x: f"{x:.4f}"):
        print(agg[cols_show].to_string())

    print(f"\n  Routing breakdown:")
    for r, n in agg["routing"].value_counts().items():
        print(f"    {r:<22s}  {int(n):>3d}")
    print(sep)

    # ── Save artefacts ─────────────────────────────────────────────────────
    suffix = f"_expert_{args.expert}" if args.expert else ""
    csv_path = os.path.join(run_dir, f"ebm_factor_health{suffix}.csv")
    agg.to_csv(csv_path)
    print(f"\n  CSV saved → {csv_path}")

    png_path = os.path.join(run_dir, f"ebm_factor_health{suffix}.png")
    plot_health(agg, png_path, top_n=args.top_n)


if __name__ == "__main__":
    main()
