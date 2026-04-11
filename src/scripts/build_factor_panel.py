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
  volatility_regime, trend_regime, skew_regime, market_adx

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
from .. import factors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    d = {sym: df.set_index("ts")[col]
         for sym, df in data.items() if col in df.columns}
    return pd.DataFrame(d).reindex(all_ts).ffill().fillna(fill)


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
    funding_lookback: int = 180,
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
    closes = _make_wide(mom_data, "futures_close", all_ts)
    basis_wide = _make_wide(mom_data, "basis",          all_ts, fill=0.0)
    vr_wide = _make_wide(mom_data, "volume_ratio",   all_ts, fill=0.0)
    fr_wide = _make_wide(mom_data, "funding_rate",   all_ts, fill=0.0)

    # OI: prefer metrics_store (full historical archive) over mom_data
    # mom_data["open_interest"] is often 0 because _load_local_oi() used the
    # wrong filename pattern ({symbol}-metrics-{date}.csv vs the actual
    # {symbol}_metrics_{date}.csv), so it silently returned empty and the
    # momentum loader fell back to filling open_interest = 0.0.
    if oi_store:
        oi_wide_dict = {}
        for sym in symbols:
            if sym in oi_store and not oi_store[sym].empty:
                oi_wide_dict[sym] = oi_store[sym].reindex(
                    all_ts).ffill().fillna(0.0)
            else:
                oi_wide_dict[sym] = pd.Series(0.0, index=all_ts)
        oi_wide = pd.DataFrame(oi_wide_dict)
    else:
        oi_wide = _make_wide(mom_data, "open_interest", all_ts, fill=0.0)

    # Regime columns are market-wide (same for all symbols each day)
    regime_cols = ["volatility_regime", "trend_regime", "skew_regime", "adx"]
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

    ret_1d = closes.pct_change(1)
    ret_5d = closes.pct_change(5)
    ret_20d = closes.pct_change(20)

    volatility = factors.calc_volatility(closes, vol_lookback).fillna(0.0)
    active_vol = volatility.replace(0.0, np.nan)
    vol_rank_cs = active_vol.rank(axis=1, pct=True).fillna(0.5)

    skewness_90 = factors.calc_skewness(closes, lookback=90)

    beta_60 = factors.calc_beta_df(closes, beta_lookback)

    # ── 4. Momentum factors ──────────────────────────────────────────────────
    price_roc = factors.calc_price_mom(
        closes,  mom_lookback, smooth_lookback).fillna(0.0)
    oi_roc = factors.calc_oi_mom(
        oi_wide,    mom_lookback, smooth_lookback).fillna(0.0)
    basis_mom = factors.calc_basis_mom(
        basis_wide, closes, mom_lookback, smooth_lookback).fillna(0.0)
    vol_ratio_s = factors.calc_vol_ratio_signal(
        vr_wide, mom_lookback, mom_lookback).fillna(1.0)
    funding_z = factors.calc_funding_zscore(
        fr_wide, funding_lookback).fillna(0.0)

    basis_norm = (basis_wide / closes.replace(0, np.nan)).fillna(0.0)

    trend_score = price_roc * (1 + 2 * oi_roc)

    valid_sent = ~(np.isinf(basis_mom) | np.isinf(vol_ratio_s))
    sentiment_raw = (basis_mom * vol_ratio_s * 5).where(valid_sent, 0.0)
    sentiment_score = _neutralize(sentiment_raw, trend_score)

    combined_score = trend_score + sentiment_score

    # Funding penalty
    funding_penalty = pd.DataFrame(1.0, index=combined_score.index,
                                   columns=combined_score.columns)
    boost = ((funding_z > funding_z_threshold) & (combined_score > 0)) | \
            ((funding_z < -funding_z_threshold) & (combined_score < 0))
    kill = ((funding_z < -funding_z_threshold*2) & (combined_score > 0)) | \
        ((funding_z > funding_z_threshold*2) & (combined_score < 0)) | \
        (funding_z.abs() < funding_z_threshold * 0.1)
    funding_penalty[boost] = 1.5
    funding_penalty[kill] = 0.5

    mom_final_score = _neutralize(combined_score, beta_60)

    # ── 5. Reversal factors ──────────────────────────────────────────────────
    oi_pct_chg = oi_wide.pct_change().replace([np.inf, -np.inf], np.nan)
    ls_chg_1d = ls_wide.diff()

    cs_z_oi = _cs_zscore(oi_pct_chg)
    ts_z_oi = _ts_zscore(cs_z_oi, rev_ts_lookback)
    liq_shock = (-ts_z_oi - 0.5).clip(lower=0.0)

    # Regime: (short_MA - long_MA) / long_std, cs-masked
    ma_long = closes.rolling(rev_sentiment_ma).mean()
    std_long = closes.rolling(rev_sentiment_ma).std()
    regime_score_raw = ((closes.rolling(rev_regime_window).mean() - ma_long)
                        / std_long).clip(-3, 3).fillna(0.0)
    regime_cs = _cs_zscore(regime_score_raw)
    regime_score = regime_score_raw.mask(
        regime_cs.abs() < rev_regime_threshold, 0.0)

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
    # A 10-day delta on a 40-90d stat changes by only ~10/W of the window
    # per step, producing a near-constant, smoothed signal with little
    # cross-sectional discriminatory power:
    #   beta_60        (60d rolling beta)
    #   skewness_90    (90d rolling skewness)
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
        _stack(skewness_90,     "skewness_90"),
        # momentum
        _stack(price_roc,       "price_roc"),
        _stack(oi_roc,          "oi_roc"),
        _stack(basis_norm,      "basis_norm"),
        _stack(basis_mom,       "basis_mom"),
        _stack(vol_ratio_s,     "vol_ratio_sig"),
        _stack(trend_score,     "trend_score"),
        _stack(sentiment_score, "sentiment_score"),
        _stack(combined_score,  "combined_score"),
        _stack(funding_z,       "funding_z"),
        _stack(funding_penalty, "funding_penalty"),
        _stack(mom_final_score, "mom_final_score"),
        # reversal
        _stack(ls_wide,         "ls_ratio"),
        _stack(ls_chg_1d,       "ls_chg_1d"),
        _stack(oi_pct_chg,      "oi_pct_chg_1d"),
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

    # ── Attach strategy signals ───────────────────────────────────────────────
    if not args.no_signals:
        print("\nAttaching strategy signals...")
        panel = attach_signals(panel, args.run_id)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_name = f"factor_panel_{args.start_date}_{args.end_date}.parquet"
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
