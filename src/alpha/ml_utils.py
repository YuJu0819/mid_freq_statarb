"""
General ML signal utilities — reusable across EBM and other ML strategies.

Functions
---------
normalize_features          feature normalization (CS / TS / rank / none)
neutralize_features_on_adx  remove rolling ADX-beta component from each feature
build_target                forward-return target with optional beta neutralization
predictions_to_weights      quantile selection + balanced weight assignment
_beta_neutralize_wide       core OLS residualization (internal helper)
neutralize_scores           beta-neutralize raw prediction scores before ranking
compute_portfolio_performance  OOS Sharpe / total return from a weight matrix
compute_ic                  OOS Spearman IC time series
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def normalize_features(
    panel: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
    ts_z_window: int = 126,
) -> pd.DataFrame:
    """
    Returns a copy of panel with feature_cols normalized.

    mode
    ----
    cs    : per-date cross-sectional z-score
    ts    : per-symbol rolling z-score (window = ts_z_window)
    rank  : per-date cross-sectional percentile rank → [0, 1]
    none  : no normalization

    Market-level features (same value for all symbols on a given date) have
    CS std ≈ 0.  If > 80% of dates have CS std < 1e-8, fall back to an
    expanding TS z-score so the feature's time variation is preserved.
    """
    panel = panel.copy()
    if mode == "none":
        return panel

    if mode in ("cs", "rank"):
        for col in feature_cols:
            if col not in panel.columns:
                continue
            wide = panel.pivot(index="ts", columns="symbol", values=col)

            cs_std = wide.std(axis=1)
            frac_zero_cs = (cs_std < 1e-8).mean()
            if frac_zero_cs > 0.8:
                # TS z-score: rolling window so stats are not anchored to the
                # panel's start date.  expanding() would give different values
                # when the panel is trimmed (per-epoch efficiency trimming
                # changes the anchor), making results dependent on how much
                # history is loaded.  rolling(ts_z_window) fixes this: as long
                # as ts_z_window warmup bars are present before the first
                # prediction date, the normalized values are identical regardless
                # of panel start.
                mu_ts = wide.rolling(ts_z_window, min_periods=ts_z_window // 4).mean()
                sd_ts = wide.rolling(ts_z_window, min_periods=ts_z_window // 4).std().replace(0, np.nan)
                wide = (wide - mu_ts) / sd_ts
            elif mode == "cs":
                mu = wide.mean(axis=1)
                sd = cs_std.replace(0, np.nan)
                wide = wide.sub(mu, axis=0).div(sd, axis=0)
            else:  # rank
                wide = wide.rank(axis=1, pct=True)

            long = wide.stack(dropna=False).rename(col).reset_index()
            panel = panel.drop(columns=[col]).merge(
                long, on=["ts", "symbol"], how="left")

    elif mode == "ts":
        for col in feature_cols:
            if col not in panel.columns:
                continue
            wide = panel.pivot(index="ts", columns="symbol", values=col)
            mu = wide.rolling(ts_z_window, min_periods=ts_z_window // 4).mean()
            sd = wide.rolling(
                ts_z_window, min_periods=ts_z_window // 4).std().replace(0, np.nan)
            wide = (wide - mu) / sd
            long = wide.stack(dropna=False).rename(col).reset_index()
            panel = panel.drop(columns=[col]).merge(
                long, on=["ts", "symbol"], how="left")

    return panel


# ---------------------------------------------------------------------------
# ADX-beta feature neutralization
# ---------------------------------------------------------------------------

def neutralize_features_on_adx(
    panel: pd.DataFrame,
    feature_cols: list[str],
    adx_col: str = "market_adx",
    window: int = 252,
) -> pd.DataFrame:
    """
    For each (feature, symbol) pair remove the component that co-varies with
    market ADX strength via a rolling OLS:

        feature_i(t) = α_i + β_i * adx(t) + residual_i(t)

    The residual replaces the raw feature value.  β_i is the symbol's
    sensitivity to trending-market conditions; removing it leaves the
    idiosyncratic part of the feature that is independent of whether the
    market is in a strong trend or ranging.

    Parameters
    ----------
    panel       : long-format panel with columns [ts, symbol, feature_cols..., adx_col]
    feature_cols: features to neutralize
    adx_col     : market-wide ADX column (same value for all symbols per date)
    window      : rolling regression window in trading periods (default 252)

    Implementation
    --------------
    Uses the Pearson rolling-covariance formula, fully vectorized:

        roll_cov(feat, adx) = roll_mean(feat * adx) - roll_mean(feat) * roll_mean(adx)
        slope = roll_cov / roll_var(adx)
        intercept = roll_mean(feat) - slope * roll_mean(adx)
        residual = feat - slope * adx - intercept

    Since adx is market-wide (identical across symbols on any date), the ADX
    rolling statistics (mean, var) are computed once and broadcast.
    """
    panel = panel.copy()
    if adx_col not in panel.columns:
        print(f"  [adx_neutral] '{adx_col}' not found in panel — skipping.")
        return panel

    min_p = max(window // 4, 10)

    # ADX is market-wide: take unique ts→adx mapping (same for all symbols)
    adx_series = (
        panel[["ts", adx_col]]
        .drop_duplicates("ts")
        .set_index("ts")[adx_col]
        .sort_index()
    )

    adx_mean = adx_series.rolling(window, min_periods=min_p).mean()
    adx_var  = adx_series.rolling(window, min_periods=min_p).var().replace(0, np.nan)
    adx_sq_mean = (adx_series ** 2).rolling(window, min_periods=min_p).mean()
    # var(adx) = mean(adx²) - mean(adx)² — already handled by .var() above

    neutralized = 0
    for col in feature_cols:
        if col not in panel.columns:
            continue

        wide = panel.pivot(index="ts", columns="symbol", values=col)

        # rolling mean of feature and of feature*adx (vectorized across symbols)
        feat_mean     = wide.rolling(window, min_periods=min_p).mean()
        feat_adx_mean = wide.multiply(adx_series, axis=0).rolling(
            window, min_periods=min_p).mean()

        # rolling cov(feat, adx) = E[feat*adx] - E[feat]*E[adx]
        roll_cov  = feat_adx_mean.subtract(
            feat_mean.multiply(adx_mean, axis=0))
        slope     = roll_cov.divide(adx_var, axis=0)
        intercept = feat_mean.subtract(slope.multiply(adx_mean, axis=0))

        residual  = wide.subtract(
            slope.multiply(adx_series, axis=0)).subtract(intercept)

        long = residual.stack(dropna=False).rename(col).reset_index()
        panel = panel.drop(columns=[col]).merge(long, on=["ts", "symbol"], how="left")
        neutralized += 1

    print(f"  [adx_neutral] Neutralized {neutralized} features on '{adx_col}' "
          f"(window={window}).")
    return panel


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_target(
    panel: pd.DataFrame,
    target_col: str,
    horizon: int,
    target_type: str,
    beta_neutral: bool = False,
    beta_col: str = "beta_60",
) -> pd.DataFrame:
    """
    Adds 'y' and 'y_raw' columns to the panel.

    y_raw : plain decimal forward return — always unmodified, used for
            portfolio PnL computation (Sharpe, cumulative return).

    y     : training target.
            If beta_neutral=True, the forward return is first residualized
            cross-sectionally against beta_col so the model learns to predict
            idiosyncratic alpha rather than beta-driven return.
            Then transformed according to target_type:
              cs_rank : cross-sectional percentile rank per date − 0.5 → (−0.5, 0.5)
              raw     : (beta-neutral) decimal return as-is

    Must be called BEFORE normalize_features so that y_raw is built from the
    original decimal return column, not a z-scored version of it.
    """
    panel = panel.copy()
    wide = panel.pivot(index="ts", columns="symbol", values=target_col)
    fwd_raw = wide.shift(-horizon)

    # y_raw: plain decimal forward return for PnL, never neutralized
    long_raw = fwd_raw.stack(dropna=False).rename("y_raw").reset_index()
    panel = panel.merge(long_raw, on=["ts", "symbol"], how="left")

    # fwd_for_target: optionally beta-neutral version used for training
    if beta_neutral and beta_col in panel.columns:
        beta_wide = panel.pivot(
            index="ts", columns="symbol", values=beta_col
        ).reindex(index=fwd_raw.index, columns=fwd_raw.columns)

        fwd_alpha = fwd_raw.copy()
        for ts in fwd_raw.index:
            y = fwd_raw.loc[ts]
            b = beta_wide.loc[ts]
            valid = y.notna() & b.notna() & ~np.isinf(y) & ~np.isinf(b)
            if valid.sum() < 3 or np.var(b[valid].values) < 1e-8:
                continue
            try:
                slope, intercept = np.polyfit(
                    b[valid].values, y[valid].values, 1)
                fwd_alpha.loc[ts, valid] = (
                    y[valid].values - (slope * b[valid].values + intercept)
                )
            except Exception:
                pass
        fwd_for_target = fwd_alpha
    else:
        fwd_for_target = fwd_raw

    if target_type == "cs_rank":
        fwd = fwd_for_target.rank(axis=1, pct=True) - 0.5
    else:
        fwd = fwd_for_target

    long = fwd.stack(dropna=False).rename("y").reset_index()
    panel = panel.merge(long, on=["ts", "symbol"], how="left")
    return panel


# ---------------------------------------------------------------------------
# Beta neutralization
# ---------------------------------------------------------------------------

def _beta_neutralize_wide(
    signal: pd.DataFrame,
    beta_wide: pd.DataFrame,
    active_mask: str = "nonzero",
) -> pd.DataFrame:
    """
    Cross-sectional OLS residualization of a wide (ts × symbol) signal.

    active_mask
    -----------
    "all"     : residualize every non-NaN asset (use for raw scores)
    "nonzero" : residualize only assets with signal != 0 (use for weights,
                where zero means "not selected" and should remain 0)
    """
    neutralized_rows = []
    for ts in signal.index:
        s = signal.loc[ts]
        b = beta_wide.loc[ts]

        if active_mask == "nonzero":
            active = (s != 0) & s.notna() & b.notna(
            ) & ~np.isinf(s) & ~np.isinf(b)
        else:  # "all"
            active = s.notna() & b.notna() & ~np.isinf(s) & ~np.isinf(b)

        if active.sum() < 2:
            neutralized_rows.append(s)
            continue

        S = s[active].values
        B = b[active].values

        if np.var(B) < 1e-8:
            neutralized_rows.append(s)
            continue

        try:
            slope, intercept = np.polyfit(B, S, 1)
            residuals = S - (slope * B + intercept)
            new_s = s.copy()
            new_s[active] = residuals
            neutralized_rows.append(new_s)
        except Exception:
            neutralized_rows.append(s)

    return pd.DataFrame(
        neutralized_rows, index=signal.index, columns=signal.columns
    )


def neutralize_scores(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    beta_col: str = "beta_60",
) -> pd.DataFrame:
    """
    Remove market-beta from raw prediction scores BEFORE ranking.

    All assets are included in the OLS regression on each date so the ranking
    itself is beta-clean — high-beta assets cannot be over-selected because
    their features are inflated by market exposure.
    """
    if beta_col not in panel.columns:
        raise ValueError(
            f"Beta column '{beta_col}' not found in panel. "
            f"Available columns: {panel.columns.tolist()}"
        )
    beta_wide = panel.pivot(
        index="ts", columns="symbol", values=beta_col
    ).reindex(index=predictions.index, columns=predictions.columns)

    return _beta_neutralize_wide(predictions, beta_wide, active_mask="nonzero")


# ---------------------------------------------------------------------------
# Prediction → weight conversion
# ---------------------------------------------------------------------------

def predictions_to_weights(
    predictions_wide: pd.DataFrame,
    quantile: float,
    max_weight: float,
    weight_mode: str,
    target_gross: float = 1.0,
) -> pd.DataFrame:
    """
    Convert raw prediction scores to a balanced long/short weight matrix.

    Refined to prevent extreme concentration caused by 'Zero-Floor' logic.
    """
    weights = pd.DataFrame(0.0, index=predictions_wide.index,
                           columns=predictions_wide.columns)
    half = target_gross / 2.0

    def _norm_capped(arr: np.ndarray, target: float, max_w: float) -> np.ndarray:
        arr = arr.astype(float).copy()
        if arr.sum() < 1e-12:
            return arr
        arr = arr / arr.sum() * target  # normalize to portfolio target
        arr = arr.clip(max=max_w)       # enforce per-asset cap
        return arr

    def _soft_z_weight(vals: np.ndarray) -> np.ndarray:
        if len(vals) <= 1:
            return np.ones(len(vals))
        z = (vals - vals.mean()) / (vals.std() + 1e-12)
        return np.exp(z.clip(-3, 3))

    for ts, row in predictions_wide.iterrows():
        row = row.dropna()
        if row.empty:
            continue

        n = len(row)
        # Global integer ranks: 1 = lowest score, n = highest score
        # Mirrors the IS calculation in _fold_portfolio_perf exactly.
        int_ranks = row.rank(method="first")

        long_mask = int_ranks > n * (1 - quantile)
        short_mask = int_ranks <= n * quantile

        if not long_mask.any() or not short_mask.any():
            continue

        longs_idx = row[long_mask].index
        shorts_idx = row[short_mask].index

        # --- Compute Weight Magnitudes ---------------------------------------
        if weight_mode == "rank":
            # Within-basket ranks: threshold asset = 1, strongest = n_long/n_short.
            # Re-rank so weight spread is independent of universe size.
            long_w = int_ranks[long_mask].rank(
                method="first").values.astype(float)
            short_w = (n + 1 - int_ranks[short_mask]
                       ).rank(method="first").values.astype(float)

        elif weight_mode == "zscore":
            # Z-score the full universe first so mean/std are not distorted
            # by computing them on the already-selected long/short subsets.
            full_z = (row - row.mean()) / (row.std() + 1e-12)
            long_w = (full_z[long_mask].values.clip(-3, 3))
            short_w = (-full_z[short_mask].values.clip(-3, 3))

        elif weight_mode == "raw":
            # Demean using the full-universe mean so longs and shorts share
            # the same reference point rather than each side being self-demeaned.
            full_demeaned = row - row.mean()
            long_w = full_demeaned[long_mask].values
            short_w = -full_demeaned[short_mask].values

        else:  # equal
            long_w = np.ones(long_mask.sum())
            short_w = np.ones(short_mask.sum())

        # --- Water-filling normalisation (hard cap enforcement) --------------
        long_w = _norm_capped(long_w,  half, max_weight)
        short_w = _norm_capped(short_w, half, max_weight)

        weights.loc[ts, longs_idx] = long_w
        weights.loc[ts, shorts_idx] = -short_w

    return weights

# ---------------------------------------------------------------------------
# OOS evaluation
# ---------------------------------------------------------------------------


def compute_portfolio_performance(
    weights_wide: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict:
    """
    Compute OOS portfolio Sharpe, total return, and daily return series
    from a weight matrix and the panel (which must contain y_raw).
    """
    raw_ret_wide = panel.pivot(
        index="ts", columns="symbol", values="y_raw"
    ).reindex(index=weights_wide.index, columns=weights_wide.columns).fillna(0.0)

    port_rets = (weights_wide * raw_ret_wide).sum(axis=1)
    port_rets = port_rets[port_rets.index.isin(raw_ret_wide.index)]

    if len(port_rets) < 5 or port_rets.std() == 0:
        return {
            "sharpe": np.nan, "total_return": np.nan,
            "win_rate": np.nan, "port_rets": port_rets,
        }

    sharpe = float(port_rets.mean() / port_rets.std() * np.sqrt(252))
    total_ret = float((1 + port_rets).prod() - 1)
    win_rate = float((port_rets > 0).mean())
    return {
        "sharpe": sharpe, "total_return": total_ret,
        "win_rate": win_rate, "port_rets": port_rets,
    }


def compute_ic(
    predictions_wide: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
    target_horizon: int,
) -> pd.Series:
    """
    Information Coefficient: Spearman rank correlation between
    predicted score and realized forward return, per date.
    """
    wide_ret = panel.pivot(index="ts", columns="symbol", values=target_col)
    fwd_ret = wide_ret.shift(-target_horizon)

    ics = {}
    for ts in predictions_wide.index:
        if ts not in fwd_ret.index:
            continue
        pred = predictions_wide.loc[ts].dropna()
        real = fwd_ret.loc[ts].reindex(pred.index).dropna()
        common = pred.index.intersection(real.index)
        if len(common) < 5:
            continue
        ic, _ = stats.spearmanr(pred[common], real[common])
        ics[ts] = ic

    return pd.Series(ics, name="IC")
