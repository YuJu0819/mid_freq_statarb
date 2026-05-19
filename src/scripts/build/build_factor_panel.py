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

from ...core.utils import load_config
from ...data.loader import DataLoader, _load_local_metrics
from ...data.universe import load_validated_universe
from ...data.rolling_universe import (
    RollingUniverse, build_symbol_active_mask, resolve_epochs,
)
from ... import factors


# ---------------------------------------------------------------------------
# Phase-6 refactor: helpers relocated to top-level modules.
# Re-exports below keep every existing call site (inside main() and
# any external consumer of these symbols) working unchanged.
#   mask_pre_launch_rows, mask_post_death_rows  → src/factor_masking.py
#   _make_wide, _ffill_with_limit, _stack, _delta,
#   compute_factors                              → src/factor_panel.py
#   load_metrics_store, attach_signals          → src/factor_panel_io.py
# Earlier phase-1 extractions (_neutralize, _cs_zscore, _ts_zscore)
# stay re-exported here for backward compat.
# ---------------------------------------------------------------------------
from ...factor_masking import (  # noqa: E402,F401
    mask_pre_launch_rows, mask_post_death_rows,
)
from ...factor_panel import (  # noqa: E402,F401
    _make_wide, _ffill_with_limit, _stack, _delta, compute_factors,
)
from ...factor_panel_io import (  # noqa: E402,F401
    load_metrics_store, attach_signals,
)
from ...alpha.neutralize import _neutralize  # noqa: E402,F401
from ...core.cs import _cs_zscore, _ts_zscore  # noqa: E402,F401

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
    # Phase-2 refactor: shared preamble helper. The long-format mask loop
    # below is structurally distinct from the wide-DF pattern in the
    # backtest scripts, so it stays local.
    ru_epochs = resolve_epochs(
        args.start_date, args.end_date,
        no_rolling_universe=args.no_rolling_universe)
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
