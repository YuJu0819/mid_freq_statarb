"""
Factor-panel assembly, factored out of src/scripts/build_factor_panel.py
during the phase-6 refactor.

Contents
--------
  _make_wide          {sym: df} -> wide (ts × symbol) for one column
  _ffill_with_limit   ffill with a hard limit on consecutive NaN streaks
  _stack              wide → long with a named series and (ts, symbol) index
  _delta              rolling-mean(W) − rolling-mean(W).shift(L)
  compute_factors     the big pipeline: assembles ~60 features from
                      per-symbol momentum data + OI/LS stores into a
                      long-format (ts, symbol, ...) panel

All four lifted verbatim from build_factor_panel — no behaviour change.
The function reuses the canonical CS/TS z-score and beta-neutralization
helpers from phases 1-5 and the per-factor calculators in src/factors.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import factors
from .alpha.neutralize import _neutralize
from .core.cs import _cs_zscore, _ts_zscore


def _make_wide(
    data: "dict[str, pd.DataFrame]",
    col: str,
    all_ts: pd.Index,
    fill=np.nan,
) -> pd.DataFrame:
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


def compute_factors(
    mom_data: "dict[str, pd.DataFrame]",
    ls_store: "dict[str, pd.Series]",
    oi_store: "dict[str, pd.Series] | None" = None,
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
