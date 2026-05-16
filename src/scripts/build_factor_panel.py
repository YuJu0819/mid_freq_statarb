"""
Build a flat factor panel for ML research.

Loads all raw and derived factors used by the momentum and reversal strategies,
plus the actual strategy signals (weights) if available, and outputs a single
long-format parquet indexed by (ts, symbol).

Output columns
--------------
  -- price --
  ret_1d, ret_5d, ret_20d
  volatility_30                 rolling 30-day return std
  vol_rank_cs                   cross-sectional percentile rank of volatility
  beta_60                       60-day rolling beta to BTC/ETH/SOL
  skewness_90                   90-day return skewness

  -- momentum factors --
  price_roc                     90d price momentum (10d smoothed)
  oi_roc                        90d OI momentum (10d smoothed)
  basis_norm                    basis / futures_close
  basis_mom                     90d change in basis_norm (10d smoothed)
  vol_ratio_sig                 volume ratio signal
  trend_score                   price_roc * (1 + 2*oi_roc)
  sentiment_score               beta-neutralized(basis_mom * vol_ratio_sig * 5)
  combined_score                trend_score + sentiment_score
  funding_z                     14-day funding z-score
  funding_penalty               1.5 / 1.0 / 0.5 boost/neutral/kill
  mom_final_score               beta-neutralized combined_score (exact strategy signal)

  -- reversal factors --
  ls_ratio                      top long/short account ratio
  ls_chg_1d                     1-day change in ls_ratio
  oi_pct_chg_1d                 1-day % change in open_interest
  cs_z_oi_chg                   cross-sectional z-score of oi_pct_chg_1d
  ts_z_oi_chg                   30-day time-series z-score of cs_z_oi_chg
  liquidation_shock             (-ts_z_oi_chg - 0.5).clip(0)
  regime_score                  (5d MA - 50d MA) / 50d std, cs-masked
  interaction_alpha             liquidation_shock * regime_score
  reversal_hawkes               EWM(interaction_alpha, halflife=5)
  rev_final_score               beta-neutralized reversal_hawkes (exact strategy signal)

  -- market regime (market-wide, same for all symbols per day) --
  volatility_regime, trend_regime, skew_regime, market_adx, market_volatility
                                (market_volatility = raw 30d std of EW market
                                 return; continuous alternative to market_adx
                                 for neutralization / regime separation)

  -- delta family ({factor}_delta) --
  <factor>_delta                fac_mean_5 - fac_mean_5_lag_10 for each base factor.
                                Captures rate-of-change of a factor's recent level.
                                Applied to: volatility_30, vol_rank_cs, beta_60,
                                skewness_90, price_roc, oi_roc, basis_norm, basis_mom,
                                vol_ratio_sig, trend_score, sentiment_score,
                                combined_score, funding_z, mom_final_score, ls_ratio,
                                liquidation_shock, regime_score, interaction_alpha,
                                rev_final_score.
                                Skipped: already-differenced (ls_chg_1d, oi_pct_chg_1d,
                                cs_z_oi_chg, ts_z_oi_chg), raw returns (ret_*),
                                categorical (funding_penalty), rev_hawkes.

  -- actual strategy signals (from saved weight parquets) --
  mom_signal                    final portfolio weight from momentum strategy
  rev_signal                    final portfolio weight from reversal strategy

Usage
-----
    python -m src.scripts.build_factor_panel \\
        --run_id batch_v1 \\
        --start_date 2024-01-01 \\
        --end_date   2025-01-01

    # Override universe from config instead of saved universe file
    python -m src.scripts.build_factor_panel \\
        --run_id batch_v1 --start_date 2024-01-01 --end_date 2025-01-01 \\
        --no_cache
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core.utils import load_config
from ..data.loader import DataLoader, _load_local_metrics
from ..data.universe import load_validated_universe
from ..data.rolling_universe import RollingUniverse, build_symbol_active_mask
from .. import factors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_pre_launch_rows(
    panel: pd.DataFrame,
    min_active_days: int = 5,
    return_col: str = "ret_1d",
) -> pd.DataFrame:
    """
    NaN-mask pre-tradeable rows for each symbol.

    A symbol's price series is "pre-launch" when the underlying data is
    forward-filled (or zero-filled) before the symbol was actually trading
    on the venue. In that period `ret_1d == 0` exactly for many consecutive
    days, which downstream produces all-zero rolling features (volatility_30,
    price_roc, mom_final_score, ...). Those synthetic-zero rows survive
    `dropna(subset=["y"])` because y == 0 (not NaN), and end up corrupting
    EBM training as if they were real observations.

    Detection: for each symbol, find the FIRST date `d*` such that the symbol
    has accumulated at least `min_active_days` non-zero, non-NaN returns
    within the panel up to and including `d*`. All rows BEFORE `d*` get
    every numeric column set to NaN. From `d*` onwards the panel is
    untouched.

    Parameters
    ----------
    panel           : long-format panel with [ts, symbol, return_col, ...]
    min_active_days : minimum number of non-zero return days before a symbol
                      is considered tradeable. 5 is enough to distinguish
                      forward-fill from a low-volume day.
    return_col      : column used to detect activity (default "ret_1d")

    Returns
    -------
    panel : same shape, with pre-launch rows NaN-masked.
    """
    if return_col not in panel.columns:
        print(f"  [pre_launch_mask] '{return_col}' missing — skipping.")
        return panel

    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    feature_cols = [c for c in panel.columns
                    if c not in ("ts", "symbol")
                    and pd.api.types.is_numeric_dtype(panel[c])]

    # For each symbol, find first date where cumulative count of non-zero
    # returns reaches min_active_days.
    is_active = (panel[return_col].fillna(0) != 0).astype(int)
    cum_active = is_active.groupby(panel["symbol"]).cumsum()
    # Mask rows where cumulative count is still below threshold
    pre_launch_mask = cum_active < min_active_days

    n_masked = int(pre_launch_mask.sum())
    if n_masked == 0:
        print(f"  [pre_launch_mask] no pre-launch rows detected.")
        return panel

    # Per-symbol first-active date and rows-masked count for reporting.
    # Distinguish symbols that traded from panel start (lose only ~min_active_days
    # rows) from genuinely-late-launching symbols.
    first_active = (panel.loc[~pre_launch_mask]
                    .groupby("symbol")["ts"].min())
    rows_masked_per_sym = pre_launch_mask.groupby(panel["symbol"]).sum()
    panel_start = panel["ts"].min()
    # "Late-launching" = first active date is more than 30 days after panel start
    late_syms = first_active[
        first_active > panel_start + pd.Timedelta(days=30)]

    panel.loc[pre_launch_mask, feature_cols] = np.nan
    print(f"  [pre_launch_mask] masked {n_masked:,} rows "
          f"({n_masked/len(panel):.1%} of panel)  "
          f"threshold = {min_active_days} non-zero return days")
    print(f"  [pre_launch_mask] {len(late_syms)} symbols launched >30 days "
          f"after panel start (the contamination source)")
    if len(late_syms):
        latest = first_active.loc[late_syms.index].sort_values(
            ascending=False).head(5)
        print(f"  [pre_launch_mask] latest-launching: "
              + ", ".join(f"{s}@{d.strftime('%Y-%m-%d')}"
                          for s, d in latest.items()))
    return panel


def mask_post_death_rows(
    panel: pd.DataFrame,
    min_active_days: int = 5,
    return_col: str = "ret_1d",
) -> pd.DataFrame:
    """
    NaN-mask post-death (trailing forward-filled) rows for each symbol.

    Symmetric counterpart to mask_pre_launch_rows. After Binance delists or
    rebrands a symbol (e.g. MATIC→POL, RNDR→RENDER, AGIX→FET, FTM→S), the
    historical data feed keeps emitting the last close price, producing an
    indefinite tail of `ret_1d == 0`. Those synthetic-zero rows survive
    `dropna(subset=["y"])` and pollute training the same way pre-launch
    rows do.

    Detection: for each symbol, find the LAST date `d*` such that the symbol
    still has at least `min_active_days` non-zero, non-NaN returns from
    `d*` onward (i.e., looking forward to panel end). All rows AFTER `d*`
    get every numeric column set to NaN. Implemented as a reverse cumsum,
    fully symmetric to the pre-launch helper.

    Parameters
    ----------
    panel           : long-format panel with [ts, symbol, return_col, ...]
    min_active_days : same threshold as pre-launch (default 5)
    return_col      : column used to detect activity (default "ret_1d")

    Returns
    -------
    panel : same shape, with post-death rows NaN-masked.
    """
    if return_col not in panel.columns:
        print(f"  [post_death_mask] '{return_col}' missing — skipping.")
        return panel

    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    feature_cols = [c for c in panel.columns
                    if c not in ("ts", "symbol")
                    and pd.api.types.is_numeric_dtype(panel[c])]

    # For each symbol, count remaining non-zero returns FROM each row to
    # the symbol's last row (reverse cumsum on the forward-time series).
    is_active = (panel[return_col].fillna(0) != 0).astype(int)
    rev_cum = (
        is_active[::-1]
        .groupby(panel["symbol"][::-1], sort=False)
        .cumsum()[::-1]
    )
    post_death_mask = rev_cum < min_active_days

    n_masked = int(post_death_mask.sum())
    if n_masked == 0:
        print(f"  [post_death_mask] no post-death rows detected.")
        return panel

    last_active = (panel.loc[~post_death_mask]
                   .groupby("symbol")["ts"].max())
    panel_end = panel["ts"].max()
    # "Dead-tail" = last active date is more than 30 days before panel end
    dead_syms = last_active[
        last_active < panel_end - pd.Timedelta(days=30)]

    panel.loc[post_death_mask, feature_cols] = np.nan
    print(f"  [post_death_mask] masked {n_masked:,} rows "
          f"({n_masked/len(panel):.1%} of panel)  "
          f"threshold = {min_active_days} non-zero return days")
    print(f"  [post_death_mask] {len(dead_syms)} symbols dead >30 days "
          f"before panel end (the contamination source)")
    if len(dead_syms):
        earliest_dead = last_active.loc[dead_syms.index].sort_values().head(5)
        print(f"  [post_death_mask] earliest deaths: "
              + ", ".join(f"{s}@{d.strftime('%Y-%m-%d')}"
                          for s, d in earliest_dead.items()))
    return panel


def _neutralize(signal_df: pd.DataFrame, beta_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional beta-neutralization via OLS residuals (matches strategy)."""
    out = signal_df.copy()
    common_idx = signal_df.index.intersection(beta_df.index)
    for ts in common_idx:
        y = signal_df.loc[ts]
        x = beta_df.loc[ts]
        mask = (y != 0) & y.notna() & x.notna() & ~np.isinf(y) & ~np.isinf(x)
        if mask.sum() < 2 or np.var(x[mask].values) < 1e-8:
            continue
        try:
            slope, intercept = np.polyfit(x[mask].values, y[mask].values, 1)
            out.loc[ts, mask] = y[mask].values - \
                (slope * x[mask].values + intercept)
        except Exception:
            pass
    return out.reindex(signal_df.index).fillna(0.0)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def _ts_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu = df.rolling(window, min_periods=window // 2).mean()
    sd = df.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (df - mu) / sd


def _make_wide(data: dict[str, pd.DataFrame], col: str,
               all_ts: pd.Index, fill=np.nan) -> pd.DataFrame:
    """
    Build a wide (ts × symbol) DataFrame for `col`.

    Default fill is NaN (changed from 0.0 in older versions). Filling missing
    raw market data with 0 silently produced corrupted derived factors:
    `oi_pct_chg` exploded to ±1 or ±inf, `basis = 0` was indistinguishable
    from "no contango", `volume_ratio = 0` poisoned downstream rolling means.

    Callers that need a specific neutral value (e.g. `ls_ratio` defaults to
    1.0 when missing) should pass `fill=` explicitly.

    `ffill` is applied so transient one-day gaps don't drop a row, but a
    symbol that was never tracked stays NaN (no false data is invented).
    """
    d = {sym: df.set_index("ts")[col]
         for sym, df in data.items() if col in df.columns}
    return pd.DataFrame(d).reindex(all_ts).ffill().fillna(fill)


def _ffill_with_limit(df: pd.DataFrame, limit_days: int = 5) -> pd.DataFrame:
    """
    Forward-fill with a hard limit on consecutive NaN streaks.

    A 1-2 day API outage should not cascade into a full row drop, but a
    week-long gap is real and should remain NaN so downstream rolling
    computations and the EBM training drop the row rather than learn from
    invented data.
    """
    return df.ffill(limit=limit_days)


def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
    s = wide.stack()
    s.index.names = ["ts", "symbol"]
    s.name = name
    return s


def _delta(wide: pd.DataFrame, mean_window: int = 5, lag: int = 10) -> pd.DataFrame:
    """
    Delta family: rolling(mean_window).mean() - rolling(mean_window).mean().shift(lag).
    Captures the rate-of-change of a factor's recent level vs its level lag periods ago.
    """
    recent = wide.rolling(mean_window, min_periods=1).mean()
    return recent - recent.shift(lag)


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------

def compute_factors(
    mom_data: dict[str, pd.DataFrame],
    ls_store: dict[str, pd.Series],
    oi_store: dict[str, pd.Series] | None = None,
    # matches FinalStrategy(lookback=30) in backtest_multi
    mom_lookback: int = 30,
    smooth_lookback: int = 10,
    vol_lookback: int = 30,
    # 120 days (≈1 quarter) is sufficient to capture funding regime in
    # crypto and shortens warmup vs the previous 180d default. Reduces the
    # under-warmed training fraction at the start of the OOS window.
    funding_lookback: int = 120,
    funding_z_threshold: float = 1.5,
    beta_lookback: int = 60,
    # matches LiquidationReversalStrategy(ts_lookback=80) in backtest_reversal
    rev_ts_lookback: int = 80,
    # matches LiquidationReversalStrategy(sentiment_ma_window=40)
    rev_sentiment_ma: int = 40,
    rev_regime_window: int = 5,
    rev_regime_threshold: float = 0.6,
    # matches LiquidationReversalStrategy(half_life_decay=12)
    rev_halflife: int = 12,
    # Reduced from 90d to 45d — see compute_factors note on skewness.
    skew_lookback: int = 45,
    # Rolling window for liquidity smoothing (dollar volume).
    liquidity_lookback: int = 30,
) -> pd.DataFrame:
    """
    Returns a long-format DataFrame indexed by (ts, symbol) with all factor columns.
    """

    # ── 1. Align timestamps ──────────────────────────────────────────────────
    all_ts = pd.Index([])
    for df in mom_data.values():
        if not df.empty:
            all_ts = all_ts.union(df["ts"])
    all_ts = all_ts.sort_values().unique()

    symbols = list(mom_data.keys())

    # ── 2. Wide raw frames ───────────────────────────────────────────────────
    # All raw market series default to NaN where data is genuinely missing.
    # Earlier the defaults were 0.0, which corrupted downstream:
    #   - basis = 0 looked like "no contango" for symbols with no spot feed
    #   - oi = 0 → oi_pct_chg = ±1 or ±inf
    #   - volume_ratio = 0 poisoned rolling means
    # ffill(limit=5) lets us survive 1-2 day API outages without dropping
    # rows, but a multi-week gap stays NaN.
    closes = _make_wide(mom_data, "futures_close", all_ts)
    basis_wide = _make_wide(mom_data, "basis", all_ts, fill=np.nan)
    vr_wide = _make_wide(mom_data, "volume_ratio", all_ts, fill=np.nan)
    fr_wide = _make_wide(mom_data, "funding_rate", all_ts, fill=np.nan)

    # Liquidity feed: futures volume in BASE units; later we multiply by close
    # to get dollar volume. NaN-default is correct here too.
    fv_wide = _make_wide(mom_data, "futures_volume", all_ts, fill=np.nan)

    # OI: prefer metrics_store (full historical archive) over mom_data
    # because the on-disk metrics CSVs were the canonical source, while
    # mom_data["open_interest"] often defaulted to 0 from a loader fallback.
    # NaN where unavailable, never synthetic 0 (which made oi_pct_chg=±inf).
    if oi_store:
        oi_wide_dict = {}
        for sym in symbols:
            if sym in oi_store and not oi_store[sym].empty:
                oi_wide_dict[sym] = (oi_store[sym].reindex(all_ts)
                                     .ffill(limit=5))
            else:
                oi_wide_dict[sym] = pd.Series(np.nan, index=all_ts)
        oi_wide = pd.DataFrame(oi_wide_dict)
    else:
        oi_wide = _make_wide(mom_data, "open_interest", all_ts, fill=np.nan)

    # Regime columns are market-wide (same for all symbols each day)
    regime_cols = ["volatility_regime", "trend_regime", "skew_regime", "adx",
                   "market_volatility"]
    regime_frames = []
    for col in regime_cols:
        if col in next(iter(mom_data.values()), pd.DataFrame()).columns:
            sample_sym = next(
                sym for sym, df in mom_data.items() if col in df.columns
            )
            series = mom_data[sample_sym].set_index(
                "ts")[col].reindex(all_ts).ffill()
            series.name = "market_adx" if col == "adx" else col
            regime_frames.append(series)

    # L/S ratio: prefer metrics_store (full historical archive) over the
    # recent parquet accumulation store which only covers the last ~30 days.
    ls_wide_dict = {}
    for sym in symbols:
        if sym in ls_store and not ls_store[sym].empty:
            ls_wide_dict[sym] = ls_store[sym].reindex(
                all_ts).ffill().fillna(1.0)
        else:
            ls_wide_dict[sym] = pd.Series(1.0, index=all_ts)
    ls_wide = pd.DataFrame(ls_wide_dict)

    # ── 3. Price-based factors ───────────────────────────────────────────────
    # All raw factors propagate NaN: a missing input must result in a missing
    # output, never a synthetic zero. Final NaN→neutral fills happen ONLY at
    # training time (`_fill_features` after CS-normalization), where 0 means
    # "at the cross-sectional mean" rather than a real-world data value.

    ret_1d = closes.pct_change(1)
    ret_5d = closes.pct_change(5)
    ret_20d = closes.pct_change(20)

    # Volatility: NaN until `vol_lookback // 2` days of valid returns exist.
    volatility = factors.calc_volatility(closes, vol_lookback)

    # Cross-sectional rank of volatility — keep NaN where vol is NaN. The
    # previous .fillna(0.5) silently planted dead/early symbols at the median,
    # creating a synthetic spike at rank=0.5 that EBM would learn from.
    vol_rank_cs = volatility.rank(axis=1, pct=True)

    # Skewness: lookback reduced from 90d → 45d. The 90d window was too
    # smooth for daily prediction (signal barely moved day-to-day) and
    # required a long warmup. 45d still captures multi-week tail asymmetry
    # while emitting a meaningfully time-varying value.
    skewness_45 = factors.calc_skewness(closes, lookback=skew_lookback)

    beta_60 = factors.calc_beta_df(closes, beta_lookback)

    # ── 3b. Liquidity features (NEW) ────────────────────────────────────────
    # In crypto cross-section, dollar volume is one of the strongest single
    # predictors of forward return dispersion (mean-reversion in mega-caps,
    # momentum in mid-caps, noise in low-caps). Without it the EBM cannot
    # distinguish a $50M-volume coin from a $50K-volume coin — they're
    # treated identically in the ranking.
    dollar_volume = closes * fv_wide
    # 30d rolling mean smooths idiosyncratic single-day spikes (e.g.
    # listing day, news event). NaN where insufficient warmup.
    dv_30d_mean = dollar_volume.rolling(
        liquidity_lookback, min_periods=max(liquidity_lookback // 2, 5)
    ).mean()
    # log1p stabilises the skew (volume distributions span 6+ orders of
    # magnitude) — feeds the EBM an ML-friendly numeric scale.
    liquidity_log = np.log1p(dv_30d_mean.clip(lower=0))
    # Cross-sectional percentile rank — symbol-comparable per date.
    liquidity_rank_cs = dv_30d_mean.rank(axis=1, pct=True)

    # ── 4. Momentum factors ──────────────────────────────────────────────────
    # All NaN-propagating: a missing OI on a day yields NaN oi_roc that day,
    # not a synthetic 0 that biases CS normalization on that date.
    price_roc = factors.calc_price_mom(closes, mom_lookback, smooth_lookback)
    oi_roc = factors.calc_oi_mom(oi_wide, mom_lookback, smooth_lookback)
    basis_mom = factors.calc_basis_mom(
        basis_wide, closes, mom_lookback, smooth_lookback)
    # vol_ratio_signal now returns tanh(diff) ∈ (-1, 1), no exponential blowup.
    vol_ratio_s = factors.calc_vol_ratio_signal(
        vr_wide, mom_lookback, mom_lookback)
    funding_z = factors.calc_funding_zscore(fr_wide, funding_lookback)

    # basis_norm: use NaN division (no synthetic 0). Closes are always > 0
    # in valid rows after the pre-launch mask runs, so we just guard against
    # the residual case.
    safe_closes = closes.where(closes > 0, np.nan)
    basis_norm = basis_wide / safe_closes
    # 3-day smoothed companion. The instantaneous `basis_norm` can flip sign
    # on funding settlements; a 3-day mean captures the structural carry
    # rather than the daily noise around it. min_periods=2 keeps coverage
    # high without requiring the full window.
    basis_norm_smooth3 = basis_norm.rolling(3, min_periods=2).mean()

    trend_score = price_roc * (1 + 2 * oi_roc)

    # Sentiment: NaN-safe product. Old code did `.where(valid_sent, 0.0)`
    # which planted 0s on every infinity row → CS-mean bias. Now multiplication
    # naturally propagates NaN, and any inf gets replaced explicitly.
    sentiment_raw = (basis_mom * vol_ratio_s * 5).replace(
        [np.inf, -np.inf], np.nan)
    sentiment_score = _neutralize(sentiment_raw, trend_score)

    combined_score = trend_score + sentiment_score

    # Funding penalty: removed the buggy "kill near zero" clause. Previously
    # any |funding_z| < 0.15 (i.e. neutral funding, the normal market state)
    # halved the signal — punishing the most common condition for no reason.
    # Boost when funding aligns extremely with signal direction, kill only
    # when funding strongly disagrees.
    funding_penalty = pd.DataFrame(1.0, index=combined_score.index,
                                   columns=combined_score.columns)
    boost = ((funding_z > funding_z_threshold) & (combined_score > 0)) | \
            ((funding_z < -funding_z_threshold) & (combined_score < 0))
    kill = ((funding_z < -funding_z_threshold*2) & (combined_score > 0)) | \
           ((funding_z > funding_z_threshold*2) & (combined_score < 0))
    funding_penalty[boost] = 1.5
    funding_penalty[kill] = 0.5

    mom_final_score = _neutralize(combined_score, beta_60)

    # ── 5. Reversal factors ──────────────────────────────────────────────────
    # oi_pct_chg propagates NaN naturally now (no synthetic-0 OI).
    oi_pct_chg = oi_wide.pct_change().replace([np.inf, -np.inf], np.nan)
    ls_chg_1d = ls_wide.diff()

    # 3-day smoothed companions for the noisiest single-day-change features.
    # Daily OI changes can swing ±20% from a single large position; daily L/S
    # ratio diffs jump on news. A 3-day rolling mean captures sustained
    # build/unwind while filtering single-day noise. Both originals are kept
    # as features so the EBM can learn which timescale is more predictive
    # for any given regime.
    oi_pct_chg_smooth3 = oi_pct_chg.rolling(3, min_periods=2).mean()
    ls_chg_smooth3 = ls_chg_1d.rolling(3, min_periods=2).mean()

    cs_z_oi = _cs_zscore(oi_pct_chg)
    ts_z_oi = _ts_zscore(cs_z_oi, rev_ts_lookback)
    liq_shock = (-ts_z_oi - 0.5).clip(lower=0.0)

    # Regime score: keep raw value; SOFT-mask via tanh of CS-z-score instead
    # of hard-thresholding the bottom 60% to exact 0.
    # The old hard-threshold collapsed continuous information into a step
    # function with a synthetic spike at zero, which EBM can't differentiate
    # from a real "neutral" signal. The tanh weighting preserves gradient
    # while down-weighting weak cross-sectional outliers.
    ma_long = closes.rolling(
        rev_sentiment_ma, min_periods=max(rev_sentiment_ma // 2, 10)).mean()
    std_long = closes.rolling(
        rev_sentiment_ma, min_periods=max(rev_sentiment_ma // 2, 10)).std()
    regime_score_raw = ((closes.rolling(rev_regime_window).mean() - ma_long)
                        / std_long.replace(0, np.nan)).clip(-3, 3)
    regime_cs = _cs_zscore(regime_score_raw)
    # tanh(z / threshold) is ~0 at the median, ~±1 at the tails: continuous,
    # bounded, gradient-preserving.
    regime_weight = np.tanh(regime_cs / rev_regime_threshold)
    regime_score = regime_score_raw * regime_weight

    interaction_alpha = liq_shock * regime_score
    rev_hawkes = interaction_alpha.ewm(
        halflife=rev_halflife, min_periods=0).mean()
    rev_final_score = _neutralize(rev_hawkes, beta_60)

    # ── 6. Delta family: fac_mean_5 - fac_mean_5_lag_10 ─────────────────────
    # Skipped: already-differenced series (ls_chg_1d, oi_pct_chg, cs_z_oi,
    #          ts_z_oi), raw returns (ret_*), categorical (funding_penalty),
    #          and rev_hawkes (interaction_alpha_delta already captures this).
    #
    # Also skipped: factors whose underlying lookback >= 40 days.
    # A 10-day delta on a 40+ d stat changes by only ~10/W of the window
    # per step, producing a near-constant, smoothed signal with little
    # cross-sectional discriminatory power:
    #   beta_60        (60d rolling beta)
    #   skewness_45    (45d rolling skewness — was 90d, still slow vs delta)
    #   funding_z      (180d rolling z-score)
    #   liquidation_shock  (built from an 80d TS z-score)
    #   regime_score   (40d rolling MA in denominator)
    #   interaction_alpha  (liq_shock × regime_score — both slow)
    delta_targets = [
        (volatility,        "volatility_30"),   # 30d — 10d lag covers ~33%
        (vol_rank_cs,       "vol_rank_cs"),      # CS-rank of 30d vol
        (price_roc,         "price_roc"),        # 30d momentum
        (oi_roc,            "oi_roc"),           # 30d OI momentum
        (basis_norm,        "basis_norm"),       # spot (daily)
        (basis_mom,         "basis_mom"),        # 30d
        (vol_ratio_s,       "vol_ratio_sig"),    # 30d
        (trend_score,       "trend_score"),      # price_roc × oi_roc (30d)
        (sentiment_score,   "sentiment_score"),  # 30d composite
        (combined_score,    "combined_score"),   # 30d composite
        (mom_final_score,   "mom_final_score"),  # 30d composite
        (ls_wide,           "ls_ratio"),         # daily
        (rev_final_score,   "rev_final_score"),  # EWM halflife=12, responsive
        (liquidity_rank_cs, "liquidity_rank_cs"),  # CS-rank of 30d $-volume
    ]
    delta_parts = [
        _stack(_delta(wide), f"{name}_delta")
        for wide, name in delta_targets
    ]

    # ── 7. Stack all factors to long format ──────────────────────────────────
    print("  Stacking to long format...")
    parts = [
        _stack(ret_1d,          "ret_1d"),
        _stack(ret_5d,          "ret_5d"),
        _stack(ret_20d,         "ret_20d"),
        _stack(volatility,      "volatility_30"),
        _stack(vol_rank_cs,     "vol_rank_cs"),
        _stack(beta_60,         "beta_60"),
        _stack(skewness_45,     "skewness_45"),
        # liquidity (NEW)
        _stack(liquidity_log,    "liquidity_log"),
        _stack(liquidity_rank_cs, "liquidity_rank_cs"),
        # momentum
        _stack(price_roc,       "price_roc"),
        _stack(oi_roc,          "oi_roc"),
        _stack(basis_norm,        "basis_norm"),
        _stack(basis_norm_smooth3, "basis_norm_smooth3"),  # 3d smoothed
        _stack(basis_mom,         "basis_mom"),
        _stack(vol_ratio_s,     "vol_ratio_sig"),
        _stack(trend_score,     "trend_score"),
        _stack(sentiment_score, "sentiment_score"),
        _stack(combined_score,  "combined_score"),
        _stack(funding_z,       "funding_z"),
        _stack(funding_penalty, "funding_penalty"),
        _stack(mom_final_score, "mom_final_score"),
        # reversal
        _stack(ls_wide,         "ls_ratio"),
        _stack(ls_chg_1d,         "ls_chg_1d"),
        _stack(ls_chg_smooth3,    "ls_chg_smooth3"),     # 3d smoothed
        _stack(oi_pct_chg,        "oi_pct_chg_1d"),
        _stack(oi_pct_chg_smooth3, "oi_pct_chg_smooth3"),  # 3d smoothed
        _stack(cs_z_oi,         "cs_z_oi_chg"),
        _stack(ts_z_oi,         "ts_z_oi_chg"),
        _stack(liq_shock,       "liquidation_shock"),
        _stack(regime_score,    "regime_score"),
        _stack(interaction_alpha, "interaction_alpha"),
        _stack(rev_hawkes,      "reversal_hawkes"),
        _stack(rev_final_score, "rev_final_score"),
    ]

    panel = pd.concat(parts + delta_parts, axis=1)

    # Attach market-wide regime columns (broadcast to every symbol)
    for series in regime_frames:
        panel[series.name] = series.reindex(panel.index, level="ts")

    # Numeric encoding of regime labels for ML use.
    # Ordinal values are chosen to reflect the natural ordering of each regime:
    #
    #   volatility_regime_enc : Low=0, Medium=1, High=2
    #     (monotone: higher = more volatile)
    #
    #   trend_regime_enc      : Ranging=0, Weak Trend=1, Strong Trend=2
    #     (monotone: higher = clearer trend, measured by ADX)
    #
    #   skew_regime_enc       : Negative=-1, Neutral=0, Positive=1
    #     (symmetric around 0: sign conveys direction of tail risk)
    #
    # The original string columns are preserved for backtest reporting.
    _VOL_ENC = {"Low Volatility": 0,
                "Medium Volatility": 1, "High Volatility": 2}
    _TREND_ENC = {"Ranging": 0, "Weak Trend": 1, "Strong Trend": 2}
    _SKEW_ENC = {"Negative Skew": -1, "Neutral Skew": 0, "Positive Skew": 1}

    if "volatility_regime" in panel.columns:
        panel["volatility_regime_enc"] = (
            panel["volatility_regime"].map(_VOL_ENC).astype("float32"))
    if "trend_regime" in panel.columns:
        panel["trend_regime_enc"] = (
            panel["trend_regime"].map(_TREND_ENC).astype("float32"))
    if "skew_regime" in panel.columns:
        panel["skew_regime_enc"] = (
            panel["skew_regime"].map(_SKEW_ENC).astype("float32"))

    return panel.reset_index()


# ---------------------------------------------------------------------------
# Load OI + LS ratio from historical metrics CSVs
# ---------------------------------------------------------------------------

def load_metrics_store(
    symbols: list[str],
    start_date: str,
    end_date: str,
    metrics_dir: str = "./data/metrics",
    ls_parquet_dir: str = "./data/ls_ratio",
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """
    Load open_interest and ls_ratio for each symbol, merging two sources:

      Primary  : data/metrics/{symbol}_metrics_*.csv
                 Full historical archive (downloaded by download_metrics.py).
                 Contains both open_interest and ls_ratio.

      Fallback : data/ls_ratio/{symbol}_ls_ratio.parquet
                 Recent accumulation store (download_ls_ratio.py).
                 Covers only the last N days but has finer granularity.

    Returns
    -------
    oi_store  : {symbol: pd.Series(ts → open_interest)}
    ls_store  : {symbol: pd.Series(ts → ls_ratio)}

    Why this replaces the old load_ls_store + _load_local_oi pair:
      - _load_local_oi() used the wrong filename pattern
        ({symbol}-metrics-{date}.csv with hyphens) while the actual files
        use underscores ({symbol}_metrics_{date}.csv).
      - load_ls_store() read only the recent parquet accumulation store,
        missing all historical dates in the backtest window.
    """
    t0 = pd.to_datetime(start_date)
    t1 = pd.to_datetime(end_date)

    oi_store: dict[str, pd.Series] = {}
    ls_store: dict[str, pd.Series] = {}

    for sym in symbols:
        frames = []

        # ── Primary: historical metrics CSVs ─────────────────────────────────
        df_hist = _load_local_metrics(sym, metrics_dir)
        if not df_hist.empty:
            df_hist = df_hist[(df_hist["ts"] >= t0) & (df_hist["ts"] <= t1)]
            frames.append(df_hist)

        # ── Fallback: recent parquet accumulation store ───────────────────────
        parquet_path = os.path.join(ls_parquet_dir, f"{sym}_ls_ratio.parquet")
        if os.path.exists(parquet_path):
            try:
                df_rec = pd.read_parquet(parquet_path)
                df_rec["ts"] = pd.to_datetime(df_rec["ts"])
                df_rec = df_rec[(df_rec["ts"] >= t0) & (df_rec["ts"] <= t1)]
                frames.append(df_rec)
            except Exception as e:
                print(f"  [metrics_store] {sym} parquet: {e}")

        if not frames:
            continue

        merged = (
            pd.concat(frames)
            .sort_values("ts")
            .drop_duplicates("ts", keep="last")
            .reset_index(drop=True)
        )

        if "open_interest" in merged.columns:
            s = merged.set_index("ts")["open_interest"].dropna()
            if not s.empty:
                oi_store[sym] = s

        if "ls_ratio" in merged.columns:
            s = merged.set_index("ts")["ls_ratio"].dropna()
            if not s.empty:
                ls_store[sym] = s

    return oi_store, ls_store


# ---------------------------------------------------------------------------
# Attach strategy signals from saved weight parquets
# ---------------------------------------------------------------------------

def attach_signals(panel: pd.DataFrame, run_id: str,
                   strategy_names: list[str] = ("momentum", "reversal"),
                   signal_cols: list[str] = ("mom_signal", "rev_signal")) -> pd.DataFrame:
    """
    Loads weight parquets and left-joins them onto the panel by (ts, symbol).
    Missing entries become NaN (symbol not in that strategy's universe on that day).
    """
    base = f"./reports/strategies/{run_id}"
    for name, col in zip(strategy_names, signal_cols):
        path = os.path.join(base, f"{name}.parquet")
        if not os.path.exists(path):
            print(
                f"  [signals] {path} not found — column '{col}' will be NaN.")
            panel[col] = np.nan
            continue
        w = pd.read_parquet(path)
        if not pd.api.types.is_datetime64_any_dtype(w.index):
            w.index = pd.to_datetime(w.index)
        # Stack to long
        stacked = w.stack().reset_index()
        stacked.columns = ["ts", "symbol", col]
        panel = panel.merge(stacked, on=["ts", "symbol"], how="left")
        print(f"  [signals] Attached '{col}' from {path}")
    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a flat ML factor panel from momentum + reversal strategy data."
    )
    ap.add_argument("--run_id",     required=True,
                    help="Run ID for loading weight parquets")
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date",   required=True)
    ap.add_argument("--no_cache",   action="store_true",
                    help="Force re-fetch all data")
    ap.add_argument("--out_dir",    default="./data/ml",
                    help="Output directory (default: ./data/ml)")
    ap.add_argument("--no_signals", action="store_true",
                    help="Skip attaching strategy signal columns")
    ap.add_argument("--no_rolling_universe", action="store_true",
                    help="Skip rolling universe NaN masking. Use this when "
                         "building a panel for per-epoch EBM training, which "
                         "needs the full pre-epoch history for each symbol.")
    ap.add_argument("--no_prelaunch_mask", action="store_true",
                    help="Skip pre-launch NaN masking. By default, rows where "
                         "a symbol's price was forward-filled (pre-launch) "
                         "have all numeric features set to NaN so they get "
                         "dropped at training time. Use this flag to keep "
                         "the raw zero-filled rows for inspection / debugging.")
    ap.add_argument("--no_postdeath_mask", action="store_true",
                    help="Skip post-death NaN masking. By default, trailing "
                         "forward-filled rows (after a symbol is delisted or "
                         "rebranded — e.g. MATIC→POL, FTM→S) are NaN-masked "
                         "symmetrically to pre-launch. Use this flag to keep "
                         "the dead-tail rows.")
    ap.add_argument("--prelaunch_min_active_days", type=int, default=5,
                    help="Minimum non-zero return days before a symbol is "
                         "considered tradeable. Used by both pre-launch and "
                         "post-death masks. Rows outside the active range "
                         "are NaN-masked.")
    ap.add_argument("--vol_threshold_mode", choices=["expanding", "rolling"],
                    default="expanding",
                    help="How the volatility_regime quantile thresholds are "
                         "estimated. 'expanding' (default) uses full history "
                         "up to t-1; 'rolling' uses trailing "
                         "--vol_threshold_window days (more adaptive to "
                         "regime shifts in crypto). A/B test by running this "
                         "script once per mode — the output filename includes "
                         "the mode suffix to prevent overwrite.")
    ap.add_argument("--vol_threshold_window", type=int, default=45,
                    help="Trailing window for 'rolling' mode (default 504 = 2y).")
    ap.add_argument("--vol_threshold_min_periods", type=int, default=45,
                    help="Warmup before first non-NaN regime label.")
    # Note: label-flicker hysteresis is NOT a panel-build knob. The trainer
    # applies it at use time via `--moe_hysteresis` (RegimeSelector), which
    # keeps it sweepable without rebuilding the panel.
    args = ap.parse_args()

    cfg = load_config()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Universe ─────────────────────────────────────────────────────────────
    symbols = load_validated_universe(args.start_date, args.end_date)
    if symbols is None:
        print("No validated universe found — falling back to config symbols.")
        symbols = cfg["backtest"]["symbols"]
    print(f"Universe: {len(symbols)} symbols")

    # ── Load momentum data (rich feature set) ─────────────────────────────────
    loader = DataLoader(
        parquet_dir="./cache/parquet",
        local_oi_dir=cfg.get("data", {}).get("oi_dir", "./data/open_interest"),
        local_metrics_dir=cfg.get("data", {}).get(
            "metrics_dir", "./data/metrics"),
        vol_threshold_mode=args.vol_threshold_mode,
        vol_threshold_window=args.vol_threshold_window,
        vol_threshold_min_periods=args.vol_threshold_min_periods,
    )

    print("\nLoading momentum data (this uses the cache when available)...")
    mom_data = loader.load_momentum_universe(
        symbols, args.start_date, args.end_date, no_cache=args.no_cache
    )
    print(f"Loaded {len(mom_data)} symbols.\n")

    if not mom_data:
        print("No data loaded. Exiting.")
        return

    # ── Load OI + L/S ratio from historical metrics CSVs ─────────────────────
    metrics_dir = cfg.get("data", {}).get("metrics_dir", "./data/metrics")
    ls_parquet_dir = "./data/ls_ratio"
    print("Loading OI + L/S ratio from historical metrics archive...")
    oi_store, ls_store = load_metrics_store(
        list(mom_data.keys()), args.start_date, args.end_date,
        metrics_dir=metrics_dir, ls_parquet_dir=ls_parquet_dir,
    )
    print(f"  OI coverage  : {len(oi_store)}/{len(mom_data)} symbols")
    print(f"  L/S coverage : {len(ls_store)}/{len(mom_data)} symbols")

    # ── Compute all factors ───────────────────────────────────────────────────
    print("\nComputing factors...")
    panel = compute_factors(mom_data, ls_store, oi_store=oi_store)
    print(f"  Panel shape: {panel.shape}")

    # ── Pre-launch NaN masking ────────────────────────────────────────────────
    # Symbols whose price data was forward-filled before they actually started
    # trading produce all-zero rolling features that survive dropna(y) and
    # corrupt EBM training. Mask those rows so they become NaN → y becomes
    # NaN at build_target time → row gets dropped.
    if not args.no_prelaunch_mask:
        print("\nMasking pre-launch (forward-filled) rows...")
        panel = mask_pre_launch_rows(
            panel, min_active_days=args.prelaunch_min_active_days)
    else:
        print("\nSkipping pre-launch mask (--no_prelaunch_mask).")

    # ── Post-death NaN masking ────────────────────────────────────────────────
    # Symmetric to pre-launch: after a symbol is delisted/rebranded the data
    # feed keeps emitting the last close → indefinite ret_1d == 0 tail. Same
    # contamination pattern, fixed the same way.
    if not args.no_postdeath_mask:
        print("\nMasking post-death (delisted/rebranded) rows...")
        panel = mask_post_death_rows(
            panel, min_active_days=args.prelaunch_min_active_days)
    else:
        print("\nSkipping post-death mask (--no_postdeath_mask).")

    # ── Rolling Universe Mask ─────────────────────────────────────────────────
    # For each (ts, symbol) row that falls outside the symbol's active epoch,
    # set all factor columns to NaN. The shape of the panel is unchanged —
    # inactive entries remain as rows but with NaN values, so the EBM can
    # distinguish "no signal" from a genuine zero-valued factor.
    #
    # IMPORTANT: Skip this when building a panel for per-epoch EBM training
    # (--no_rolling_universe). The per-epoch pipeline filters to active symbols
    # itself and needs pre-epoch history for TS rolling warmup. Masking it here
    # causes the training window to see mostly NaN → avg symbols degrades across
    # later epochs because their pre-epoch lookback data is zeroed out.
    if not args.no_rolling_universe:
        ru = RollingUniverse()
        if not ru.is_empty():
            ru_epochs = ru.get_epochs(args.start_date, args.end_date)
            if ru_epochs:
                print("\nApplying rolling universe mask to factor panel...")
                factor_cols = [
                    c for c in panel.columns if c not in ("ts", "symbol")]
                panel_ts = pd.to_datetime(panel["ts"])
                active_flag = pd.Series(False, index=panel.index)
                for ep in ru_epochs:
                    ep_start = pd.Timestamp(ep["epoch_start"])
                    ep_end = pd.Timestamp(
                        ep["epoch_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    ep_syms = set(ep["symbols"])
                    in_ep = (panel_ts >= ep_start) & (
                        panel_ts <= ep_end) & panel["symbol"].isin(ep_syms)
                    active_flag |= in_ep
                n_masked = (~active_flag).sum()
                panel.loc[~active_flag, factor_cols] = np.nan
                print(
                    f"  {n_masked:,} rows masked as NaN ({n_masked / len(panel):.1%} of panel).")
    else:
        print("\nSkipping rolling universe NaN mask (--no_rolling_universe). "
              "Full history retained for all symbols.")

    # ── Attach strategy signals ───────────────────────────────────────────────
    if not args.no_signals:
        print("\nAttaching strategy signals...")
        panel = attach_signals(panel, args.run_id)

    # ── Save ──────────────────────────────────────────────────────────────────
    # Filename includes the vol-threshold mode so an A/B run (expanding vs
    # rolling) produces two distinct parquets that downstream consumers
    # (train_ebm_signal, backtest) can point at independently. The legacy
    # filename (no suffix) is preserved for the default 'expanding' mode so
    # existing scripts keep working.
    if args.vol_threshold_mode == "expanding":
        suffix = ""
    elif args.vol_threshold_mode == "rolling":
        suffix = f"_volroll{args.vol_threshold_window}"
    else:
        suffix = f"_vol{args.vol_threshold_mode}"
    out_name = (f"factor_panel_{args.start_date}_{args.end_date}"
                f"{suffix}.parquet")
    out_path = os.path.join(args.out_dir, out_name)
    panel.to_parquet(out_path, index=False)

    print(f"\nFactor panel saved → {out_path}")
    print(f"  Rows   : {len(panel):,}")
    print(f"  Columns: {len(panel.columns)}")
    print(f"  Date range : {panel['ts'].min()} → {panel['ts'].max()}")
    print(f"  Symbols    : {panel['symbol'].nunique()}")
    print(f"\n  Columns:\n  " + "\n  ".join(panel.columns.tolist()))


if __name__ == "__main__":
    main()
