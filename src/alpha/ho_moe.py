"""
Hierarchical Orthogonalized Mixture of Experts (HO-MoE) building blocks.

Pass 1 — Macro CMI Tournament
-----------------------------
At the start of every walk-forward fold the trailing 270-day IS panel
(already filtered to the upcoming-period universe by the caller) is mined for
the macro regime separator that unlocks the most non-linear predictive
power. Three explicit candidates compete:

    market_volatility   — cross-sectional mean of per-symbol 30d return vol
    market_liquidity    — cross-sectional sum of daily dollar volume, 7d MA
    market_dispersion   — cross-sectional std of 1d returns

For each candidate R we compute the Conditional Mutual Information
CMI(X_i; Y | R) for every feature X_i in the panel and average across
features. EMA(span=`ema_span`) smooths the per-candidate scores across
folds and the highest-scoring candidate wins the tournament for the next
Pass 2 cycle.

Pass 2 — Macro Separator → TS Neutralization & Market-Wide Regime
-----------------------------------------------------------------
A macro (market-wide) separator has zero cross-sectional variance per
date, so the original CS-OLS neutralization is degenerate. The natural
adaptation is **time-series** neutralization (per symbol, rolling OLS
against the macro series) — for which `ml_utils.neutralize_features_on_adx`
already supplies a fully vectorised implementation. The macro separator
also collapses to a single regime per date, so the existing market-wide
`RegimeSelector` in `train_ebm_signal.py` can route experts directly
once we attach the macro column's quantile-bin label to the panel.

All helpers in this module are strictly point-in-time safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mutual_info_score
except ImportError as e:
    raise ImportError(
        "ho_moe requires scikit-learn: pip install scikit-learn"
    ) from e


# Canonical macro candidate names. Kept as a constant so callers don't drift.
MACRO_CANDIDATES: tuple[str, ...] = (
    "market_volatility",
    "market_liquidity",
    "market_dispersion",
)


# ---------------------------------------------------------------------------
# Macro candidate computation
# ---------------------------------------------------------------------------


def compute_macro_candidates(
    panel: pd.DataFrame,
    date_col: str = "ts",
    return_col: str = "ret_1d",
    vol_col: str = "volatility_30",
    liquidity_log_col: str = "liquidity_log",
    liquidity_smooth: int = 7,
) -> pd.DataFrame:
    """
    Build the three macro regime candidates as market-wide time series and
    attach them to the panel (broadcast: same value across symbols per date).

    Returns
    -------
    panel_out : DataFrame
        Copy of `panel` with the three columns in `MACRO_CANDIDATES` added
        (or overwritten if already present). Any candidate whose source
        column is missing is written as NaN; downstream code is responsible
        for handling NaN candidates.

    Definitions
    -----------
    market_volatility   = mean over symbols of `volatility_30` per date.
                          NaN-tolerant — symbols missing data are skipped.
    market_liquidity    = sum over symbols of np.expm1(liquidity_log)  (i.e.
                          recovered daily dollar volume), then smoothed by a
                          7-day rolling mean. expm1 inverts the log1p applied
                          in build_factor_panel.
    market_dispersion   = std over symbols of `ret_1d` per date.
    """
    out = panel.copy()
    grp = panel.groupby(date_col)

    # market_volatility ────────────────────────────────────────────────────
    if vol_col in panel.columns:
        mv = grp[vol_col].mean()
    else:
        mv = pd.Series(np.nan, index=sorted(panel[date_col].unique()))
    mv = mv.rename("market_volatility")

    # market_liquidity ─────────────────────────────────────────────────────
    if liquidity_log_col in panel.columns:
        dv = panel[[date_col, liquidity_log_col]].copy()
        dv["dollar_volume"] = np.expm1(dv[liquidity_log_col].clip(lower=0))
        ml = dv.groupby(date_col)["dollar_volume"].sum(min_count=1)
        ml = ml.rolling(liquidity_smooth, min_periods=1).mean()
    else:
        ml = pd.Series(np.nan, index=sorted(panel[date_col].unique()))
    ml = ml.rename("market_liquidity")

    # market_dispersion ────────────────────────────────────────────────────
    if return_col in panel.columns:
        md = grp[return_col].std()
    else:
        md = pd.Series(np.nan, index=sorted(panel[date_col].unique()))
    md = md.rename("market_dispersion")

    macros = pd.concat([mv, ml, md], axis=1).reset_index()
    out = out.drop(columns=[c for c in MACRO_CANDIDATES if c in out.columns])
    out = out.merge(macros, on=date_col, how="left")
    return out


# ---------------------------------------------------------------------------
# Binning helper
# ---------------------------------------------------------------------------


def _safe_qcut(
    s: pd.Series,
    q: int,
    drop_zero_var: bool = True,
) -> pd.Series | None:
    """
    Quantile-bin a Series into `q` integer-labelled buckets [0, q-1].

    Returns
    -------
    Series of int8 codes aligned to `s` (NaN entries preserved as -1), or
    None when the input has effectively zero variance and `drop_zero_var`
    is True (the caller should skip this feature).

    Implementation notes
    --------------------
    `duplicates="drop"` lets qcut handle features that are constant on
    chunks of the support (common for ranked / clipped factors). If the
    resulting number of bins collapses below 2, the feature carries no
    information and we return None.
    """
    s = pd.Series(s)
    valid = s.dropna()
    if valid.empty:
        return None
    if drop_zero_var and float(valid.var()) < 1e-18:
        return None
    try:
        codes = pd.qcut(valid, q=q, labels=False, duplicates="drop")
    except ValueError:
        return None
    if codes.nunique(dropna=True) < 2:
        return None
    # Re-align to original index, NaN → -1 sentinel for downstream masking.
    out = pd.Series(-1, index=s.index, dtype="int16")
    out.loc[codes.index] = codes.astype("int16").values
    return out


# ---------------------------------------------------------------------------
# Conditional Mutual Information
# ---------------------------------------------------------------------------


def conditional_mutual_information(
    x_binned: np.ndarray,
    y_binned: np.ndarray,
    r_binned: np.ndarray,
) -> float:
    """
    Compute CMI(X; Y | R) for discrete labels via the expected MI over R:

        CMI(X; Y | R) = sum_r  P(R = r) * MI(X | R=r ;  Y | R=r)

    Inputs must already be discretized to integer labels (any int dtype),
    aligned, and free of NaN (NaN should be encoded as a distinct integer
    or filtered out by the caller). `mutual_info_score` is used per regime
    state — it is fast (ms) and mathematically exact for discrete inputs.
    """
    x_binned = np.asarray(x_binned)
    y_binned = np.asarray(y_binned)
    r_binned = np.asarray(r_binned)

    # Mask: rows where any input is sentinel (-1) are dropped.
    keep = (x_binned >= 0) & (y_binned >= 0) & (r_binned >= 0)
    if keep.sum() == 0:
        return 0.0
    x_binned = x_binned[keep]
    y_binned = y_binned[keep]
    r_binned = r_binned[keep]

    total = len(r_binned)
    cmi = 0.0
    for r_val in np.unique(r_binned):
        mask = r_binned == r_val
        p_r = mask.sum() / total
        if mask.sum() < 4:
            # Not enough samples in this regime for a meaningful MI estimate.
            continue
        mi = mutual_info_score(x_binned[mask], y_binned[mask])
        cmi += p_r * float(mi)
    return cmi


# ---------------------------------------------------------------------------
# Pass 1 — CMI tournament
# ---------------------------------------------------------------------------


def _bin_panel_for_cmi(
    train_panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    candidates: tuple[str, ...],
    q_target: int = 3,
    q_features: int = 5,
    q_candidates: int = 3,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Bin everything required for the CMI tournament once, up front.

    Per the spec:
      - target Y → q_target terciles (default 3)
      - feature X_i → q_features quintiles (default 5)
      - candidate R → q_candidates terciles (default 3)

    Returns
    -------
    y_codes        : int16 array (-1 sentinel for NaN/zero-var)
    feature_codes  : {feature_name: int16 array} — features with no useful
                     binning (NaN-only or zero-variance) are excluded.
    candidate_codes: {candidate_name: int16 array} — candidates with no
                     useful binning are excluded.
    """
    y_codes_series = _safe_qcut(train_panel[target_col], q_target)
    if y_codes_series is None:
        raise ValueError(
            "CMI: target column has zero variance after dropping NaNs; "
            "the fold cannot be scored.")
    y_codes = y_codes_series.values

    feature_codes: dict[str, np.ndarray] = {}
    for f in feature_cols:
        if f not in train_panel.columns:
            continue
        codes = _safe_qcut(train_panel[f], q_features)
        if codes is None:
            continue
        feature_codes[f] = codes.values

    candidate_codes: dict[str, np.ndarray] = {}
    for r in candidates:
        if r not in train_panel.columns:
            continue
        codes = _safe_qcut(train_panel[r], q_candidates)
        if codes is None:
            continue
        candidate_codes[r] = codes.values

    return y_codes, feature_codes, candidate_codes


def discover_regime_separator_cmi(
    train_panel: pd.DataFrame,
    feature_cols: list[str],
    ema_history: dict,
    target_col: str = "y",
    candidates: tuple[str, ...] = MACRO_CANDIDATES,
    ema_span: int = 3,
    q_target: int = 3,
    q_features: int = 5,
    q_candidates: int = 3,
) -> tuple[str, dict, pd.DataFrame]:
    """
    Run the CMI tournament across the macro candidates and return the
    EMA-smoothed winner.

    Parameters
    ----------
    train_panel : DataFrame
        Trailing IS panel, already universe-filtered. Must contain
        `target_col`, all `feature_cols`, and the `candidates` columns.
    feature_cols : list[str]
        Features to average CMI over (the "factor panel" of the spec).
    ema_history : dict
        Mutable {candidate_name: ema_value} updated in place. Pass `{}` on
        the first call — cold-start uses the raw score.
    target_col : str
        Forward-return target column (the EBM training target — caller
        ensures it is the *raw* return, not a cs_rank).
    candidates : tuple[str, ...]
        Candidate column names. Defaults to MACRO_CANDIDATES.
    ema_span : int
        EMA span (folds) for the per-candidate smoothing.
    q_target, q_features, q_candidates : int
        Quantile bin counts. Defaults per spec: 3, 5, 3.

    Returns
    -------
    winner : str
        Candidate name with the highest EMA-smoothed Average_CMI.
    ema_history : dict
        Updated (same object as input, returned for clarity).
    diagnostics : DataFrame
        Per-candidate raw Average_CMI, EMA-smoothed score, and feature
        coverage (count of features that contributed a CMI estimate).
    """
    panel = train_panel.dropna(subset=[target_col]).copy()
    if panel.empty:
        raise ValueError(
            "discover_regime_separator_cmi: no rows after dropping NaN target.")

    y_codes, feature_codes, candidate_codes = _bin_panel_for_cmi(
        panel, feature_cols, target_col, candidates,
        q_target=q_target, q_features=q_features, q_candidates=q_candidates,
    )

    if not candidate_codes:
        raise RuntimeError(
            "discover_regime_separator_cmi: none of the candidates "
            f"{candidates} could be binned. Check the panel.")
    if not feature_codes:
        raise RuntimeError(
            "discover_regime_separator_cmi: no feature could be binned.")

    raw_scores: dict[str, float] = {}
    n_features_used: dict[str, int] = {}

    for r_name, r_codes in candidate_codes.items():
        per_feature = []
        for f_name, f_codes in feature_codes.items():
            cmi = conditional_mutual_information(f_codes, y_codes, r_codes)
            per_feature.append(cmi)
        if not per_feature:
            raw_scores[r_name] = 0.0
            n_features_used[r_name] = 0
            continue
        raw_scores[r_name] = float(np.mean(per_feature))
        n_features_used[r_name] = len(per_feature)

    # EMA smoothing per candidate. alpha = 2/(span+1).
    alpha = 2.0 / (ema_span + 1.0)
    smoothed: dict[str, float] = {}
    for r_name in candidates:
        cur = raw_scores.get(r_name, np.nan)
        if not np.isfinite(cur):
            # Candidate missing this fold — decay its EMA so absent scores
            # eventually lose the tournament rather than holding forever.
            if r_name in ema_history:
                ema_history[r_name] = (1.0 - alpha) * ema_history[r_name]
            smoothed[r_name] = ema_history.get(r_name, np.nan)
        else:
            prev = ema_history.get(r_name, None)
            if prev is None:
                ema_history[r_name] = float(cur)
            else:
                ema_history[r_name] = alpha * float(cur) + (1.0 - alpha) * prev
            smoothed[r_name] = ema_history[r_name]

    # Pick winner. Filter out NaN entries; if all are NaN something is
    # very wrong upstream and we'd rather fail loudly than guess.
    valid_smoothed = {k: v for k, v in smoothed.items() if np.isfinite(v)}
    if not valid_smoothed:
        raise RuntimeError(
            "discover_regime_separator_cmi: all candidates produced NaN "
            "smoothed CMI scores.")
    winner = max(valid_smoothed, key=valid_smoothed.get)

    diagnostics = pd.DataFrame({
        "raw_cmi": pd.Series(raw_scores),
        "ema_cmi": pd.Series(smoothed),
        "n_features_used": pd.Series(n_features_used),
    }).reindex(list(candidates))
    diagnostics["winner"] = diagnostics.index == winner

    return winner, ema_history, diagnostics


# ---------------------------------------------------------------------------
# Pass 2 — Macro separator regime label attachment
# ---------------------------------------------------------------------------


def attach_macro_regime_label(
    panel: pd.DataFrame,
    separator_col: str,
    label_col: str = "ho_moe_regime_enc",
    n_quantiles: int = 3,
    date_col: str = "ts",
) -> pd.DataFrame:
    """
    Add a discrete regime label column derived from the macro separator's
    *time-series* quantile across the rows present in `panel`.

    Because the separator is market-wide (same value per date across all
    symbols), the regime is also market-wide. The label is computed once
    on the de-duplicated date-level series and then broadcast back to the
    long panel.

    Quantile boundaries are computed on the panel slice handed in by the
    caller — for walk-forward use the caller should pass only the IS data
    so the label is point-in-time safe, then apply the same fitted bin
    edges to the OOS slice (see `bin_with_edges`).

    Returns
    -------
    panel_out : DataFrame
        Copy with `label_col` populated (float so it round-trips through
        existing pandas merges; downstream RegimeSelector casts to int).
    """
    out = panel.copy()
    if separator_col not in panel.columns:
        raise KeyError(f"separator_col '{separator_col}' missing.")

    series = (
        panel[[date_col, separator_col]]
        .drop_duplicates(date_col)
        .set_index(date_col)[separator_col]
        .sort_index()
    )
    codes = _safe_qcut(series, q=n_quantiles)
    if codes is None:
        # Degenerate separator — assign middle bucket to everything so the
        # downstream MoE simply uses one expert.
        codes = pd.Series(n_quantiles // 2, index=series.index, dtype="int16")

    label_map = codes.astype("float32").to_dict()
    out[label_col] = out[date_col].map(label_map).astype("float32")
    return out


def fit_macro_regime_bin_edges(
    panel: pd.DataFrame,
    separator_col: str,
    n_quantiles: int = 3,
    date_col: str = "ts",
) -> np.ndarray | None:
    """
    Compute static time-series quantile bin edges from the macro separator
    on `panel` (IS window). Returns a sorted array of internal edges
    suitable for `np.digitize` on out-of-sample dates.

    None signals a degenerate separator (the caller should fall back to a
    constant regime label).
    """
    if separator_col not in panel.columns:
        return None
    series = (
        panel[[date_col, separator_col]]
        .drop_duplicates(date_col)
        .set_index(date_col)[separator_col]
        .sort_index()
        .dropna()
    )
    if len(series) < n_quantiles or float(series.var()) < 1e-18:
        return None
    qs = np.linspace(0, 1, n_quantiles + 1)[1:-1]
    edges = np.quantile(series.values, qs)
    # Strictly monotonic guard — if quantiles collapse on a constant chunk,
    # signal degeneracy.
    if not np.all(np.diff(edges) > 0):
        return None
    return edges


def apply_macro_regime_bin_edges(
    panel: pd.DataFrame,
    separator_col: str,
    edges: np.ndarray,
    label_col: str = "ho_moe_regime_enc",
    date_col: str = "ts",
) -> pd.DataFrame:
    """
    Apply pre-fit bin edges (from `fit_macro_regime_bin_edges`) to a panel
    slice — used for OOS date labelling so the regime mapping is identical
    to the IS one that the experts were trained on.
    """
    out = panel.copy()
    if separator_col not in panel.columns:
        out[label_col] = np.nan
        return out
    series = (
        panel[[date_col, separator_col]]
        .drop_duplicates(date_col)
        .set_index(date_col)[separator_col]
        .sort_index()
    )
    labels = np.digitize(series.values, edges)
    label_map = dict(zip(series.index, labels.astype("float32")))
    out[label_col] = out[date_col].map(label_map).astype("float32")
    return out
