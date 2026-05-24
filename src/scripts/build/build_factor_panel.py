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
import yaml
from joblib import Parallel, delayed
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
    ap.add_argument("--n_epoch_jobs", type=int, default=-1,
                    help="Parallel workers for per-epoch panel builds. "
                         "-1 = use all cores; 1 = sequential (useful for "
                         "debugging). Each worker computes factors for one "
                         "epoch's universe independently.")
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

    # ── Output suffix (vol_threshold_mode tagging) ───────────────────────────
    if args.vol_threshold_mode == "expanding":
        suffix = ""
    elif args.vol_threshold_mode == "rolling":
        suffix = f"_volroll{args.vol_threshold_window}"
    else:
        suffix = f"_vol{args.vol_threshold_mode}"

    cf_epochs = resolve_epochs(
        args.start_date, args.end_date,
        no_rolling_universe=args.no_rolling_universe)

    def _build_one_panel(
        mom_subset: dict, ls_subset: dict, oi_subset: dict | None, label: str
    ) -> pd.DataFrame:
        """Run compute_factors + masks + attach_signals for one universe."""
        print(f"\nComputing factors [{label}, {len(mom_subset)} symbols]...")
        p = compute_factors(mom_subset, ls_subset, oi_store=oi_subset)
        print(f"  Panel shape: {p.shape}")

        if not args.no_prelaunch_mask:
            print("  Masking pre-launch rows...")
            p = mask_pre_launch_rows(
                p, min_active_days=args.prelaunch_min_active_days)
        if not args.no_postdeath_mask:
            print("  Masking post-death rows...")
            p = mask_post_death_rows(
                p, min_active_days=args.prelaunch_min_active_days)

        if not args.no_signals:
            print("  Attaching strategy signals...")
            p = attach_signals(p, args.run_id)
        return p

    # ── Single-pass mode (--no_rolling_universe) ─────────────────────────────
    # Falls back to the legacy single-file layout for callers that don't want
    # per-epoch panels (e.g. baseline-test runs).
    if not cf_epochs:
        panel = _build_one_panel(mom_data, ls_store, oi_store, "single universe")
        out_name = (f"factor_panel_{args.start_date}_{args.end_date}"
                    f"{suffix}.parquet")
        out_path = os.path.join(args.out_dir, out_name)
        panel.to_parquet(out_path, index=False)
        print(f"\nFactor panel saved → {out_path}")
        print(f"  Rows   : {len(panel):,}")
        print(f"  Columns: {len(panel.columns)}")
        print(f"  Date range : {panel['ts'].min()} → {panel['ts'].max()}")
        print(f"  Symbols    : {panel['symbol'].nunique()}")
        return

    # ── Per-epoch mode (rolling universe enabled) ────────────────────────────
    # For each rolling-universe epoch, build a panel containing the FULL
    # historical date range but restricted to that epoch's universe of
    # symbols. Each symbol keeps its full per-symbol history so TS rolling
    # features warm up correctly; CS columns are computed across only that
    # epoch's universe, so cross-sectional stats are clean. Saved as one
    # parquet per epoch inside a directory keyed by (start, end).
    #
    # Training-time consumers (train_ebm_signal, analyze_*) route each
    # walk-forward fold to the panel whose universe matches the fold's
    # prediction-date epoch, so the model trains on the same CS distribution
    # it will face at prediction time.
    out_dir_name = (f"factor_panel_{args.start_date}_{args.end_date}"
                    f"{suffix}")
    out_dir = os.path.join(args.out_dir, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    n_jobs = args.n_epoch_jobs if args.n_epoch_jobs != 0 else 1
    if n_jobs == -1:
        n_jobs_resolved = max(1, os.cpu_count() or 1)
    else:
        n_jobs_resolved = max(1, min(n_jobs, len(cf_epochs)))
    print(f"\nPer-epoch panels → {out_dir}/")
    print(f"  Building {len(cf_epochs)} per-epoch panel files in parallel "
          f"({n_jobs_resolved} workers).\n"
          f"  Each file contains full {args.start_date}→{args.end_date} "
          f"history restricted to that epoch's universe.\n")

    def _build_and_save(ep, i, n_total):
        ep_syms = list(ep["symbols"])
        snap = ep["snapshot_date"]
        label = (f"epoch {i}/{n_total}  snap={snap}  "
                 f"{ep['epoch_start']} → {ep['epoch_end']}")
        mom_ep = {s: mom_data[s] for s in ep_syms if s in mom_data}
        ls_ep  = {s: ls_store[s] for s in ep_syms if s in ls_store}
        oi_ep  = ({s: oi_store[s] for s in ep_syms if s in oi_store}
                  if oi_store else None)
        if not mom_ep:
            print(f"  [skip] {label}: no data for any symbol in universe.")
            return None
        panel_ep = _build_one_panel(mom_ep, ls_ep, oi_ep, label)
        ep_path = os.path.join(out_dir, f"epoch_{snap}.parquet")
        panel_ep.to_parquet(ep_path, index=False)
        print(f"  Saved → {ep_path}  "
              f"(rows={len(panel_ep):,}, syms={panel_ep['symbol'].nunique()})")
        return ep_path

    if n_jobs_resolved == 1:
        results = [
            _build_and_save(ep, i, len(cf_epochs))
            for i, ep in enumerate(cf_epochs, 1)
        ]
    else:
        # Use loky (subprocess) — each worker computes factors independently
        # so no shared-state contention. Bundles mom_data / oi_store /
        # ls_store get pickled per worker; the per-epoch slicing inside
        # _build_and_save reduces what each worker actually iterates over.
        results = Parallel(n_jobs=n_jobs_resolved, backend="loky")(
            delayed(_build_and_save)(ep, i, len(cf_epochs))
            for i, ep in enumerate(cf_epochs, 1)
        )
    written_paths = [p for p in results if p]

    # Manifest: index file so consumers can discover the per-epoch layout.
    manifest_path = os.path.join(out_dir, "manifest.yaml")
    manifest = {
        "start_date":   args.start_date,
        "end_date":     args.end_date,
        "vol_threshold_mode": args.vol_threshold_mode,
        "epochs": [
            {
                "snapshot_date": ep["snapshot_date"],
                "epoch_start":   ep["epoch_start"],
                "epoch_end":     ep["epoch_end"],
                "n_symbols":     len(ep["symbols"]),
                "file":          f"epoch_{ep['snapshot_date']}.parquet",
            }
            for ep in cf_epochs
        ],
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"\nManifest → {manifest_path}")
    print(f"Wrote {len(written_paths)} per-epoch panel files.")
    print(f"Directory: {out_dir}/")


if __name__ == "__main__":
    main()
