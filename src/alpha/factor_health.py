"""
Factor Health Evaluator for trained EBMs.

Translates EBM shape functions into four structural metrics so factors can be
routed (GLOBAL / EXPERT / DROP) in the MoE pipeline without manual inspection.

Metrics (per feature, per fold)
-------------------------------
  monotonicity        density-weighted Spearman ρ between bin centers and
                      bin scores. |ρ| → 1 means a clean monotone signal;
                      |ρ| → 0 means a non-linear / multi-modal shape.
  tail_core_ratio     max|score| in low-density "tail" bins divided by
                      max|score| in "core" bins. Density here is the true
                      PDF (count fraction / bin width), not the raw count
                      fraction — necessary because interpret uses quantile-
                      adaptive bin widths that flatten count-fraction
                      density. Threshold defaults to 0.05 PDF units. A
                      large value means the EBM's strongest learned effect
                      sits where very few training observations land — a
                      classic overfit fingerprint. Returns 0 when there
                      are no tail bins or no core signal.
  curvature           sum(|Δ²score|) across bins. Wiggly shapes are
                      structurally noisy.
  cross_bag_variance  density-weighted average of per-bin score variance
                      across the bag ensemble. High = the shape is unstable
                      between bootstrap bags → the model can't agree on what
                      the factor does.

Routing rules (defaults — overridable)
--------------------------------------
  DROP    : tail_core_ratio > 3.0  OR  cross_bag_variance > 80th-pct of all features
  EXPERT  : top-N by importance  AND  |monotonicity| < 0.30  (non-linear state)
  GLOBAL  : everything else (well-behaved linear-ish signal)

Usage
-----
    from src.alpha.factor_health import FactorHealthEvaluator

    fhe = FactorHealthEvaluator(feature_names=feature_cols)
    metrics_df = fhe.evaluate_ensemble(list_of_bag_ebms,
                                       importance=imp_series)
    routed = fhe.route(metrics_df, top_n_for_expert=20)
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy import stats

# Bin-name strings that interpret emits for missing/unseen categories. These
# carry no continuous information and must be stripped before any rank or
# numeric operation, or `float(edge)` raises.
_NON_NUMERIC_BIN = re.compile(r"^(missing|unseen|\$undef\$)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shape extraction
# ---------------------------------------------------------------------------

def _extract_univariate_shape(
    explanation,
    feat_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Pull (bin_centers, scores, density, keep_mask) for one univariate term.

    Returns None when the term is an interaction (pairwise), categorical, or
    has too few numeric bins to be meaningful. Interaction terms are skipped
    here — the PDF's framework is about per-feature shape, not pair surfaces.

    `keep_mask` is the boolean filter (length = raw bin count) that strips
    Missing/Unseen tail bins. Callers that need to slice the raw per-bag
    matrix from `m.bagged_scores_` apply the same mask to keep bin layouts
    consistent.

    Bin centers (not edges) are returned because the rank correlation
    against `scores` only makes sense at one-per-bin granularity.
    """
    data = explanation.data(feat_idx)
    if data is None:
        return None
    # Interaction terms have 2-D scores; skip them.
    scores = np.asarray(data.get("scores"))
    if scores.ndim != 1:
        return None
    names = data.get("names", [])
    # names is typically bin edges (len = len(scores) + 1) for continuous;
    # for categorical it is the category labels (len == len(scores)).
    names = list(names)

    # Strip trailing 'Missing'/'Unseen' bins from BOTH names and scores. They
    # always appear at the tail of the arrays in interpret's output.
    keep_mask = np.ones(len(scores), dtype=bool)
    for i, n in enumerate(names[-len(scores):]):
        if isinstance(n, str) and _NON_NUMERIC_BIN.match(n):
            keep_mask[i] = False

    # Build numeric edges. For continuous, names has len(scores)+1 floats.
    # For categorical, every entry is a string — fail out (not in scope).
    numeric_edges = []
    for n in names:
        if isinstance(n, str):
            if _NON_NUMERIC_BIN.match(n):
                continue
            try:
                numeric_edges.append(float(n))
            except ValueError:
                return None  # categorical feature — out of scope
        else:
            numeric_edges.append(float(n))
    numeric_edges = np.asarray(numeric_edges)
    if numeric_edges.size < 3:
        return None

    # Bin centers: midpoint of consecutive edges when continuous, else just
    # the edge values themselves when interpret already emitted centers.
    if numeric_edges.size == scores.size + 1:
        centers = 0.5 * (numeric_edges[:-1] + numeric_edges[1:])
    elif numeric_edges.size == scores.size:
        centers = numeric_edges
    else:
        # Edge count doesn't line up — bail rather than guess.
        return None

    # Density: interpret returns it on a SEPARATE uniform-width histogram
    # (typically ~14 bins) whose edges don't align with the data-adaptive
    # score bins (typically ~30). We remap mass onto the score bins via a
    # piecewise-linear CDF, then convert to a TRUE PDF (mass per unit x).
    #
    # Why PDF and not count-fraction:
    # Interpret's score bins are quantile-adaptive, so every bin holds
    # roughly 1/n_bins of the observations BY CONSTRUCTION — count-fraction
    # density is nearly flat across bins, which makes the tail/core split
    # collapse to "everything is tail" or "everything is core". The true
    # PDF (count fraction / bin width) puts outlier regions correctly
    # below the core regions because a sparse tail bin spans a much wider
    # x-range than a central bin.
    density = None
    den = data.get("density")
    if isinstance(den, dict):
        d_counts = np.asarray(den.get("scores", []), dtype=float)
        d_edges = np.asarray([
            float(v) for v in den.get("names", []) if not isinstance(v, str)
        ], dtype=float)
        if d_counts.size and d_edges.size == d_counts.size + 1:
            cum = np.concatenate([[0.0], np.cumsum(d_counts)])
            total = cum[-1] if cum[-1] > 0 else 1.0
            if numeric_edges.size == scores.size + 1:
                cdf_at_edges = np.interp(
                    numeric_edges, d_edges, cum,
                    left=0.0, right=cum[-1])
                mass = np.diff(cdf_at_edges) / total
                widths = np.diff(numeric_edges)
                # Guard against zero-width bins (degenerate edge case).
                widths = np.where(widths > 0, widths, np.nan)
                pdf = mass / widths
                # Replace NaN/inf from zero-width with the median PDF so
                # the bin doesn't poison max/min comparisons. NaN propagates
                # through the downstream metrics otherwise.
                if np.isfinite(pdf).any():
                    pdf = np.where(np.isfinite(pdf), pdf,
                                   np.nanmedian(pdf))
                else:
                    pdf = np.ones_like(scores, dtype=float)
                density = np.clip(pdf, 0.0, None)
    if density is None:
        density = np.ones_like(scores, dtype=float)

    scores = scores[keep_mask]
    centers = centers[keep_mask] if centers.size == keep_mask.size else centers
    density = density[keep_mask] if density.size == keep_mask.size else density

    if scores.size < 3:
        return None

    # NOTE: density is left in PDF units (mass per unit x), NOT normalised
    # to sum=1. The tail/core ratio needs the PDF directly to detect sparse
    # outlier regions; the weighted-Spearman and cross-bag-variance helpers
    # both renormalise to w/w.sum() internally before use.
    density = np.clip(density, 0.0, None)
    if not np.isfinite(density).any() or density.sum() <= 0:
        density = np.ones_like(density)

    return centers, scores, density, keep_mask


def _extract_per_bag_scores(
    model,
    feat_idx: int,
    keep_mask: np.ndarray,
) -> np.ndarray | None:
    """
    Read per-bag bin scores for term `feat_idx` from a single fitted EBM.

    interpret stores per-bag bin scores in `m.bagged_scores_` as a list
    (one entry per term), each entry an array of shape `(n_bags, n_bins)`.
    `m.term_scores_[feat_idx]` (and by extension `explain_global().data(i)
    ["scores"]`) is the bag-AVERAGE of that matrix.

    Layout alignment
    ----------------
    `bagged_scores_[i]` includes TWO sentinel columns that `explain_global`
    strips: column 0 is the "Missing" bin (always 0) and column -1 is the
    "Unseen" bin (always 0). Empirically verified:
      term_scores_[i][1:-1] == explain_global().data(i)["scores"]
    So we slice [:, 1:-1] before applying the analyzer's keep_mask, which
    addresses any remaining Missing/Unseen in the score-bin layout.

    Reading the bags directly lets us measure per-bag dispersion — which
    is what the cross-bag-variance metric needs but couldn't see when the
    analyzer only iterated over the outer EBM list (always length 1 in
    the trainer, whether or not block bagging is on).

    Returns
    -------
    (n_bags, n_kept_bins) array, or None when bagged_scores_ is missing
    or the bin layout can't be aligned.
    """
    bagged = getattr(model, "bagged_scores_", None)
    if bagged is None:
        return None
    try:
        mat = np.asarray(bagged[feat_idx], dtype=float)
    except (IndexError, TypeError, ValueError):
        return None
    if mat.ndim != 2:
        return None
    n_bags, n_bins_raw = mat.shape
    if n_bags < 2:
        return None
    # Strip the leading "Missing" + trailing "Unseen" sentinels when the
    # bag matrix is exactly 2 wider than the analyzer's bin count.
    if n_bins_raw == keep_mask.size + 2:
        mat = mat[:, 1:-1]
    elif n_bins_raw == keep_mask.size + 1:
        # Older interpret versions occasionally pad by 1.
        mat = mat[:, :keep_mask.size]
    elif n_bins_raw == keep_mask.size:
        pass
    elif n_bins_raw == keep_mask.size - 1:
        keep_mask = keep_mask[:n_bins_raw]
    else:
        return None
    return mat[:, keep_mask]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Density-weighted Spearman correlation: rank-correlate x and y, weighting
    each (rank_x, rank_y) pair by w. Implementation = weighted Pearson on
    rankdata(x), rankdata(y).
    """
    if x.size < 3:
        return np.nan
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return float(np.corrcoef(rx, ry)[0, 1])
    w = w / w.sum()
    mx = np.sum(w * rx)
    my = np.sum(w * ry)
    cov = np.sum(w * (rx - mx) * (ry - my))
    vx = np.sum(w * (rx - mx) ** 2)
    vy = np.sum(w * (ry - my) ** 2)
    denom = np.sqrt(vx * vy)
    if denom <= 0:
        return np.nan
    return float(cov / denom)


def _tail_core_ratio(
    scores: np.ndarray,
    density: np.ndarray,
    core_density_thresh: float = 0.05,
) -> float:
    """
    Tail-to-core outlier risk.

    Splits the bins by population mass: "core" bins carry at least
    `core_density_thresh` of the observations (default 5%), "tail" bins
    carry less. Returns max|score| in tail / max|score| in core.

    Large ratio  → the EBM's biggest learned effect sits where almost no
                   training data lives (textbook overfit on a sparse edge).
    Returns 0 when there are no tail bins, no core signal, or no scores —
    those degenerate cases shouldn't trigger a DROP.
    """
    scores = np.asarray(scores, dtype=float)
    density = np.asarray(density, dtype=float)
    if scores.size == 0:
        return 0.0
    core_mask = density >= core_density_thresh
    tail_mask = ~core_mask
    if not tail_mask.any() or not core_mask.any():
        return 0.0
    max_core = float(np.max(np.abs(scores[core_mask])))
    if max_core <= 0:
        return 0.0
    max_tail = float(np.max(np.abs(scores[tail_mask])))
    return max_tail / (max_core + 1e-6)


def _curvature(scores: np.ndarray) -> float:
    """Σ|second difference| — the wiggliness index from the PDF."""
    if scores.size < 3:
        return np.nan
    return float(np.sum(np.abs(np.diff(scores, n=2))))


def _cross_bag_variance(
    bag_scores: list[np.ndarray],
    density: np.ndarray,
) -> float:
    """
    Density-weighted mean of per-bin variance across bags.

    bag_scores : list of length n_bags, each a 1-D array of bin scores
                 with identical length. Bags that disagree on bin layout
                 are silently truncated to the shortest bag (interpret can
                 emit a different bin count when a bag's data range
                 differs).
    """
    if len(bag_scores) < 2:
        return 0.0
    min_len = min(len(s) for s in bag_scores)
    stack = np.vstack([s[:min_len] for s in bag_scores])
    if min_len != density.size:
        # Use truncated density and renormalise.
        density = density[:min_len]
        if density.sum() > 0:
            density = density / density.sum()
        else:
            density = np.ones_like(density) / len(density)
    per_bin_var = stack.var(axis=0, ddof=0)
    return float(np.sum(density * per_bin_var))


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FactorHealthEvaluator:
    """
    Translate EBM shape functions into per-feature health metrics.

    Parameters
    ----------
    feature_names : list[str]
        Names of the model's main-effect features, in the order EBM stores
        them. Used to map bag term indices back to factor names.
    drop_tail_core_ratio_thresh : float, default 3.0
        Tail-to-core ratio above this → DROP. A ratio of 3 means the largest
        effect in the tail bins is 3× larger than the largest effect in
        densely-populated bins, i.e. the model is leaning on sparse outlier
        bins. Defaults follow the spec.
    drop_cross_bag_var_pctile : float, default 0.80
        Cross-bag variance above this fold-level percentile → DROP.
    expert_monotonicity_thresh : float, default 0.30
        |monotonicity| below this AND top-N importance → EXPERT.
    """

    def __init__(
        self,
        feature_names: list[str],
        drop_tail_core_ratio_thresh: float = 3.0,
        drop_cross_bag_var_pctile: float = 0.80,
        expert_monotonicity_thresh: float = 0.30,
        core_density_thresh: float = 0.05,
    ) -> None:
        """
        core_density_thresh
            PDF threshold (mass per unit x) above which a bin counts as
            "core". The spec's 0.05 value was originally written for
            count-fraction density; we now feed true PDF density into the
            check (because interpret uses quantile-adaptive bin widths,
            which flatten count-fraction across bins). In PDF units, 0.05
            sits roughly at the median of a standard-normal feature's
            per-bin PDF and cleanly separates the central core from
            sparse outer-bin tails. Lower for heavier-tailed features.
        """
        self.feature_names = list(feature_names)
        self.drop_tail_core_ratio_thresh = drop_tail_core_ratio_thresh
        self.drop_cross_bag_var_pctile = drop_cross_bag_var_pctile
        self.expert_monotonicity_thresh = expert_monotonicity_thresh
        self.core_density_thresh = core_density_thresh

    # ------------------------------------------------------------------
    # Per-fold evaluation
    # ------------------------------------------------------------------

    def evaluate_ensemble(
        self,
        bag_models: list,
        importance: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute the four metrics for every univariate term in a single fold.

        bag_models : list of trained EBMs (one per bootstrap bag) — the
                     `models` list saved per fold by train_ebm_signal.
        importance : optional pd.Series indexed by term name carrying the
                     model's term_importances() for this fold. If provided,
                     attached as the `importance` column.

        Returns
        -------
        DataFrame indexed by feature name, columns:
          [monotonicity, tail_core_ratio, curvature, cross_bag_variance,
           n_bins, importance (optional)]
        """
        if not bag_models:
            raise ValueError("evaluate_ensemble: bag_models is empty.")

        # Use bag 0 as the "canonical" shape for centres/density/scores.
        # The bag-AVERAGED shape (returned by explain_global) is the right
        # input for monotonicity / tail-core / curvature — those describe
        # the model's effective learned function. Cross-bag variance, by
        # contrast, needs the INDIVIDUAL bags' bin scores, which interpret
        # exposes via `m.bagged_scores_` rather than as separate models.
        canon = bag_models[0]
        try:
            exp_canon = canon.explain_global()
        except Exception as e:
            raise RuntimeError(
                f"explain_global() failed on bag 0: {e}") from e

        # term_names_ lists EVERY term (mains + pairs). We only score mains
        # — pairwise interactions are reported via the existing importance
        # analyzer, not via shape geometry (their "shape" is 2-D).
        term_names = list(canon.term_names_)

        # Legacy fallback: if multiple EBMs are passed (e.g. the old
        # smoke-test pattern of training N independent EBMs and asking for
        # cross-model dispersion), retain the old per-EBM iteration so the
        # function stays useful when `bagged_scores_` isn't available.
        extra_explanations: list = []
        if len(bag_models) > 1:
            for m in bag_models[1:]:
                try:
                    extra_explanations.append(m.explain_global())
                except Exception:
                    extra_explanations.append(None)

        rows = []
        for idx, term in enumerate(term_names):
            if " & " in term:
                continue
            shape = _extract_univariate_shape(exp_canon, idx)
            if shape is None:
                continue
            centers, scores, density, keep_mask = shape

            mono = _weighted_spearman(centers, scores, density)
            tcr = _tail_core_ratio(
                scores, density,
                core_density_thresh=self.core_density_thresh)
            curv = _curvature(scores)

            # Cross-bag variance: prefer the TRUE per-bag scores from
            # canon.bagged_scores_; fall back to the (legacy) per-EBM
            # iteration when bagged_scores_ isn't exposed.
            cb_var = 0.0
            per_bag_mat = _extract_per_bag_scores(canon, idx, keep_mask)
            if per_bag_mat is not None and per_bag_mat.shape[0] >= 2:
                cb_var = _cross_bag_variance(
                    [per_bag_mat[b] for b in range(per_bag_mat.shape[0])],
                    density)
            elif extra_explanations:
                # Legacy multi-EBM path.
                other_scores = [scores]
                for be in extra_explanations:
                    if be is None:
                        continue
                    try:
                        bag_idx = list(be.feature_names).index(term)
                    except (AttributeError, ValueError):
                        bag_idx = idx
                    s = _extract_univariate_shape(be, bag_idx)
                    if s is None:
                        continue
                    other_scores.append(s[1])
                cb_var = _cross_bag_variance(other_scores, density)

            rows.append({
                "feature": term,
                "monotonicity": mono,
                "tail_core_ratio": tcr,
                "curvature": curv,
                "cross_bag_variance": cb_var,
                "n_bins": int(scores.size),
            })

        df = pd.DataFrame(rows).set_index("feature")
        if importance is not None:
            df["importance"] = importance.reindex(df.index)
        return df

    # ------------------------------------------------------------------
    # Multi-fold aggregation
    # ------------------------------------------------------------------

    def evaluate_folds(
        self,
        folds: dict[str, list],
        importance_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run evaluate_ensemble across multiple folds and aggregate.

        folds : {fold_label: list_of_bag_ebms}
                e.g. {'2024-05-06': [ebm0, ebm1, ...], ...}
        importance_df : optional (folds × terms) DataFrame from
                ebm_feature_importance.csv. The mean column-wise importance
                is attached to the output.

        Returns
        -------
        DataFrame indexed by feature, columns:
          mean & std of each metric across folds + n_folds + importance.
        """
        per_fold = []
        for fold_label, models in folds.items():
            try:
                imp = (importance_df.loc[fold_label]
                       if importance_df is not None
                       and fold_label in importance_df.index else None)
            except KeyError:
                imp = None
            df = self.evaluate_ensemble(models, importance=imp)
            df["fold"] = fold_label
            per_fold.append(df.reset_index())
        if not per_fold:
            raise ValueError("No fold produced any metrics.")
        long = pd.concat(per_fold, ignore_index=True)

        agg = long.groupby("feature").agg(
            monotonicity_mean=("monotonicity", "mean"),
            monotonicity_std=("monotonicity", "std"),
            tail_core_ratio_mean=("tail_core_ratio", "mean"),
            tail_core_ratio_max=("tail_core_ratio", "max"),
            curvature_mean=("curvature", "mean"),
            cross_bag_variance_mean=("cross_bag_variance", "mean"),
            n_bins_median=("n_bins", "median"),
            n_folds=("fold", "nunique"),
        )
        if importance_df is not None:
            mean_imp = importance_df.mean(axis=0)
            agg["importance_mean"] = mean_imp.reindex(agg.index)
            agg["importance_rank"] = agg["importance_mean"].rank(
                ascending=False, method="min")
        return agg.sort_values(
            "importance_mean" if "importance_mean" in agg.columns
            else "monotonicity_mean",
            ascending=False)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        agg_df: pd.DataFrame,
        top_n_for_expert: int = 20,
    ) -> pd.DataFrame:
        """
        Apply the routing rules from §3 of the spec.

        Order (first match wins):
          1. tail_core_ratio_max > drop_tail_core_ratio_thresh
               OR cross_bag_variance_mean > pctile(drop_cross_bag_var_pctile)
                                                                     → DROP
          2. importance_rank ≤ top_n  AND |monotonicity_mean| < thr  → EXPERT
          3. otherwise                                               → GLOBAL

        Note: the merged DROP rule (tail-or-unstable) follows the updated
        spec — both triggers report a distinct sub-label so you can tell
        which one fired.
        """
        out = agg_df.copy()
        cbv_thr = out["cross_bag_variance_mean"].quantile(
            self.drop_cross_bag_var_pctile)
        out["cross_bag_var_thresh"] = cbv_thr

        def _decide(row):
            tcr_hit = (pd.notna(row["tail_core_ratio_max"]) and
                       row["tail_core_ratio_max"] >
                       self.drop_tail_core_ratio_thresh)
            cbv_hit = (pd.notna(row["cross_bag_variance_mean"]) and
                       row["cross_bag_variance_mean"] > cbv_thr)
            if tcr_hit and cbv_hit:
                return "DROP_tail_and_unstable"
            if tcr_hit:
                return "DROP_tail"
            if cbv_hit:
                return "DROP_unstable"
            if "importance_rank" in row and \
               pd.notna(row["importance_rank"]) and \
               row["importance_rank"] <= top_n_for_expert and \
               pd.notna(row["monotonicity_mean"]) and \
               abs(row["monotonicity_mean"]) < self.expert_monotonicity_thresh:
                return "EXPERT"
            return "GLOBAL"

        out["routing"] = out.apply(_decide, axis=1)
        return out
