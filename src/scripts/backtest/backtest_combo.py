import argparse
import os
import glob
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from tqdm import tqdm
from ...core.utils import load_config, ensure_dir
from ...data.binance_futures_rest import fetch_futures_klines
from ...data.rolling_universe import (
    RollingUniverse, build_symbol_active_mask, resolve_epochs,
)
from ...data.storage import parquet_path, load_bars, save_bars
from ...backtest.engine import probabilistic_sharpe_ratio
from ...backtest.reporting import plot_equity_curve
from ...portfolio.optimizer import PortfolioOptimizer


# Phase-3 refactor: strategy parquet discovery + alignment lifted to
# src/portfolio/strategy_loader.py. Re-exports keep external callers
# (e.g. ad-hoc analysis notebooks) working through the old path.
from ...portfolio.strategy_loader import (  # noqa: E402,F401
    load_and_align_strategies,
    EXCLUDE_PREFIXES as _EXCLUDE_PREFIXES,
    EXCLUDE_EXACT as _EXCLUDE_EXACT,
)


# Phase-5 refactor: cache-first price loader lifted to src/data/prices.py.
from ...data.prices import fetch_all_prices  # noqa: E402,F401

# Phase-1 refactor: this helper was lifted verbatim to src/data/cache.py.
# Re-export keeps existing call sites (including external test imports of
# `src.scripts.backtest_combo._load_symbol_close_cached`) working.
from ...data.cache import _load_symbol_close_cached  # noqa: E402,F401


# Phase-5 refactor: cross-signal MV machinery lifted to
# src/portfolio/cross_signal_mv.py. Re-exports preserve every existing
# call site (including external imports of these symbols).
from ...portfolio.cross_signal_mv import (  # noqa: E402,F401
    compute_strategy_returns,
    _stable_cov,
    optimize_signal_weights,
)


# Phase-5 refactor: analyze_weight_distribution lifted to
# src/backtest/reporting.py (sits naturally with the other plotting
# functions). Re-export shim keeps external callers working.
from ...backtest.reporting import analyze_weight_distribution  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True,
                    help="Run ID where strategy weights are saved")
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date", required=True)
    ap.add_argument("--perf_start_date", default=None,
                    help="Optional. Trim performance reporting (equity curve, "
                         "Sharpe, weight-distribution plots and CSVs) to this "
                         "start date (YYYY-MM-DD). The saved "
                         "optimized_weights_<method>.parquet still covers the "
                         "full --start_date/--end_date range so downstream "
                         "consumers see uninterrupted history. Use to skip "
                         "the EBM-warmup window when one strategy starts "
                         "later than the others (e.g. EBM begins 2024-01-01 "
                         "while momentum/reversal go back to 2023).")
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="Strategy names to combine (default: all *.parquet in run_dir). "
                         "Example: --strategies momentum reversal ebm")

    # Optimizer Settings
    ap.add_argument("--method", default="linear",
                    choices=["linear", "mean_variance", "equal_weight", "cross_signal_mv"],
                    help="Combination method. cross_signal_mv: MV-optimise strategy blend "
                         "weights first, then apply linear CS allocation per asset.")
    ap.add_argument("--cov_lookback", type=int, default=30,
                    help="Days for covariance calculation")
    ap.add_argument("--risk_aversion", type=float, default=1.0,
                    help="Lambda for Mean-Variance (Higher = Safer)")
    ap.add_argument("--max_leverage", type=float,
                    default=1.0, help="Max Global Leverage")

    # --- FIX 1: Allow user to control concentration ---
    ap.add_argument("--max_position", type=float, default=0.10,
                    help="Max weight per asset (e.g. 0.05 = 20 assets)")
    ap.add_argument("--lambda_ema", type=float, default=0.30,
                    help="EMA smoothing factor for the per-date strategy "
                         "allocation λ in cross_signal_mv. λ_t = ema·λ_new + "
                         "(1-ema)·λ_{t-1}. Range (0, 1]. 1.0 = no smoothing "
                         "(raw MV output, prone to corner-solution flips). "
                         "Default 0.30 (~6-day half-life) — empirically "
                         "strict Pareto improvement over 1.0 on EBM_rolling "
                         "(turnover -14%, Sharpe unchanged). Lower (0.10, "
                         "0.05) gives more smoothing but starts to lag "
                         "genuine regime rotation. Only used for "
                         "cross_signal_mv.")
    ap.add_argument("--weight_ema", type=float, default=1.0,
                    help="EMA smoothing factor for the final per-asset "
                         "weight vector. w_t = ema·w_new + (1-ema)·w_{t-1}. "
                         "Applied AFTER the optimizer regardless of method. "
                         "1.0 = no smoothing (default). Smaller = more "
                         "persistent positions, lower turnover but slower "
                         "alpha capture. Use this when the asset-level "
                         "turnover dominates the λ contribution (typical "
                         "for cross_signal_mv where 80% of turnover comes "
                         "from underlying z-score wiggle, not λ flipping). "
                         "After smoothing, gross is renormalised to "
                         "--max_leverage to keep the leverage budget intact.")
    ap.add_argument("--cov_method",
                    choices=["sample", "ledoit_wolf", "diagonal"],
                    default="ledoit_wolf",
                    help="Covariance estimator for the strategy-blend MV "
                         "step (cross_signal_mv only). With just 3 "
                         "strategies the sample Σ from a 30d window is "
                         "noisy and triggers corner-solution flips. "
                         "'ledoit_wolf' (default) shrinks toward a "
                         "scaled-identity target with the optimal data-"
                         "driven intensity. 'diagonal' zeroes the "
                         "off-diagonals (assumes strategy returns are "
                         "uncorrelated). 'sample' reproduces the old "
                         "behaviour.")
    ap.add_argument("--cov_shrinkage", type=float, default=None,
                    help="Explicit shrinkage intensity α∈[0,1] for "
                         "ledoit_wolf/diagonal. None = data-driven "
                         "(Ledoit-Wolf optimal estimate). Use 0.3-0.5 to "
                         "force more shrinkage than LW chooses.")

    # Transaction cost — matches src/backtest/engine.py:213-316.
    ap.add_argument("--fee_bps", type=float, default=None,
                    help="Round-trip fee in basis points per unit "
                         "turnover. None = read from config.yaml "
                         "(backtest.fee_bps). The per-day TC drag is "
                         "turnover_t × (fee+slippage) / 10000.")
    ap.add_argument("--slippage_bps", type=float, default=None,
                    help="Slippage in bps per unit turnover. None = "
                         "read from config.yaml (backtest.slippage_bps).")
    ap.add_argument("--no_tc", action="store_true",
                    help="Disable transaction-cost deduction entirely "
                         "(gross-only reporting). Useful for A/B against "
                         "the legacy combo output.")

    args = ap.parse_args()

    base_dir = f"./reports/strategies/{args.run_id}"

    # Resolve TC rates: CLI overrides config; --no_tc forces zero.
    _cfg = load_config()
    _cfg_bt = _cfg.get("backtest", {}) if isinstance(_cfg, dict) else {}
    if args.no_tc:
        fee_bps = 0.0
        slip_bps = 0.0
    else:
        fee_bps = (args.fee_bps if args.fee_bps is not None
                   else float(_cfg_bt.get("fee_bps", 0.0)))
        slip_bps = (args.slippage_bps if args.slippage_bps is not None
                    else float(_cfg_bt.get("slippage_bps", 0.0)))
    cost_bps = (fee_bps + slip_bps) / 10000.0
    print(f"\n[TC] fee={fee_bps:.1f} bps  slippage={slip_bps:.1f} bps  "
          f"→ deduction = turnover × {cost_bps:.5f} per day")

    # 1. Load Strategy Signals
    strategies, all_ts, all_syms = load_and_align_strategies(
        base_dir, args.strategies)

    # 1b. Safety-net epoch mask — zero weights outside each symbol's active epoch.
    # Each individual strategy (momentum, reversal, ebm) already applies this mask
    # at generation time; this pass catches any future strategy that doesn't.
    # Phase-2 refactor: only the preamble is shared with other scripts;
    # the in-place mask of wide weight matrices is structurally distinct
    # from the build-a-mask-DataFrame pattern in backtest_reversal/multi,
    # so it stays local.
    ru_epochs = resolve_epochs(args.start_date, args.end_date)
    if ru_epochs:
        print(f"Applying rolling universe epoch mask to loaded strategies "
              f"({len(ru_epochs)} epochs)...")
        for name, df in strategies.items():
            zeroed = 0
            for sym in df.columns:
                ts_series = pd.Series(df.index, index=df.index)
                active = build_symbol_active_mask(sym, ts_series, ru_epochs)
                inactive = ~active.values
                if inactive.any():
                    zeroed += int(inactive.sum())
                    df.loc[inactive, sym] = 0.0
            print(f"  {name}: zeroed {zeroed:,} inactive (date, symbol) entries.")

    # 2. Fetch Market Data
    fetch_start = (pd.to_datetime(args.start_date) -
                   pd.Timedelta(days=args.cov_lookback+20)).strftime('%Y-%m-%d')
    prices_df = fetch_all_prices(all_syms, fetch_start, args.end_date)

    prices_df = prices_df.reindex(all_ts).ffill()
    returns_df = prices_df.pct_change()

    # 3. Prepare Composite Alpha Score
    # Each strategy is cross-sectionally z-scored before averaging so that
    # strategies with larger raw score magnitudes don't dominate the composite.
    #
    # CRITICAL: epoch-masked / pre-launch / post-death cells in each strategy
    # parquet are STORED AS 0.0. If those zeros enter the per-date mean/std
    # they (a) bias mu_t/sd_t by ~100 fake observations, and (b) every
    # inactive symbol receives the same small synthetic z-score = -mu_t/sd_t.
    # The optimizer (optimize_linear has no quantile selection) then trades
    # on those phantom signals → universe explodes from ~150 to ~270 and
    # turnover spikes as the phantom z's flicker day-to-day.
    #
    # Fix: NaN-mask the strategy's zero cells BEFORE z-scoring, keep NaN
    # through the composite (nanmean across strategies), and let the
    # optimizer's .dropna() naturally skip inactive symbols at use time.
    print("\n--- Computing Composite Signal ---")
    n_strategies = len(strategies)
    normed_signals = {}

    for name, df in strategies.items():
        df_active = df.where(df != 0)   # 0 → NaN, real values kept
        mu = df_active.mean(axis=1, skipna=True)
        sd = df_active.std(axis=1, skipna=True).replace(0, np.nan)
        normed = df_active.sub(mu, axis=0).div(sd, axis=0)  # NaN preserved
        normed_signals[name] = normed

    # nanmean across strategies → a symbol active in some strategies on date
    # t gets the mean of those strategies' z-scores; a symbol inactive
    # everywhere stays NaN.
    stacked = np.stack([s.reindex(index=all_ts, columns=all_syms).values
                        for s in normed_signals.values()])
    with np.errstate(invalid="ignore"):
        composite_alpha = pd.DataFrame(
            np.nanmean(stacked, axis=0),
            index=all_ts, columns=all_syms,
        )

    # Pre-compute strategy return series (used only by cross_signal_mv)
    strat_returns_df = None
    if args.method == "cross_signal_mv":
        print("Pre-computing strategy return series for cross-signal MV...")
        strat_returns_df = compute_strategy_returns(strategies, returns_df, all_ts)

    # 4. Run Optimizer (Walk-Forward)
    print(
        f"\n--- Running Optimizer (Method: {args.method}, MaxPos: {args.max_position}) ---")
    optimizer = PortfolioOptimizer(
        max_leverage=args.max_leverage,
        max_position=args.max_position,
        lambda_risk=args.risk_aversion
    )

    final_weights_list = []
    signal_weights_history = {}   # populated for cross_signal_mv only
    prev_lambda: pd.Series | None = None   # EMA state for cross_signal_mv
    prev_w: pd.Series | None = None        # EMA state for per-asset weights

    # Track stats — counted over ALL dates (flat days contribute 0)
    active_asset_counts = []

    for t in tqdm(all_ts):
        if t not in composite_alpha.index:
            final_weights_list.append(pd.Series(0.0, index=all_syms, name=t))
            continue

        alpha_t = composite_alpha.loc[t]
        # NaN = inactive symbol (no strategy fired for it). Treat NaN and
        # exact-zero alike when checking whether anything is tradeable.
        active_signals = alpha_t[alpha_t.notna() & (alpha_t != 0)]

        if active_signals.empty:
            final_weights_list.append(pd.Series(0.0, index=all_syms, name=t))
            active_asset_counts.append(0)
            continue

        w = pd.Series(0.0, index=all_syms)

        if args.method == "equal_weight":
            w = composite_alpha.loc[t].copy()
            # Enforce per-asset position cap before leverage scaling
            w = w.clip(-args.max_position, args.max_position)
            lev = w.abs().sum()
            if lev > args.max_leverage:
                w = w * (args.max_leverage / lev)

        elif args.method == "cross_signal_mv":
            # Step 1: MV-optimise strategy blend weights using past strategy returns
            past_strat = strat_returns_df.loc[:t].iloc[-(args.cov_lookback + 1):-1]

            if len(past_strat) < max(5, args.cov_lookback // 2):
                # Not enough history yet — fall back to equal strategy weights
                signal_weights_raw = pd.Series(
                    1.0 / n_strategies, index=list(normed_signals.keys()))
            else:
                signal_weights_raw = optimize_signal_weights(
                    past_strat, args.risk_aversion,
                    cov_method=args.cov_method,
                    cov_shrinkage=args.cov_shrinkage)

            # EMA-smooth the strategy allocation to suppress corner-solution
            # flips. Without this, MV-on-3-strategies-with-30d-lookback
            # routinely jumps from λ_ebm=1.0 to λ_reversal=1.0 day-over-day,
            # contributing meaningful weight turnover for no real signal.
            if prev_lambda is None or args.lambda_ema >= 1.0:
                signal_weights = signal_weights_raw
            else:
                aligned_prev = prev_lambda.reindex(
                    signal_weights_raw.index).fillna(1.0 / n_strategies)
                signal_weights = (args.lambda_ema * signal_weights_raw
                                  + (1.0 - args.lambda_ema) * aligned_prev)
                # Re-normalise to budget=1 in case prev/new sets differ.
                s = signal_weights.sum()
                if s > 1e-12:
                    signal_weights = signal_weights / s
            prev_lambda = signal_weights

            # Step 2: blend CS z-scored signals using optimal λ.
            # Inactive symbol in strategy X → NaN z in normed_signals[X];
            # (NaN * lam).fillna(0) makes that strategy contribute 0 to
            # the blend for that symbol — instead of poisoning the entire
            # sum to NaN, which would mask genuine signals from other
            # strategies.
            blended = pd.Series(0.0, index=all_syms)
            any_active = pd.Series(False, index=all_syms)
            for name, lam in signal_weights.items():
                if t in normed_signals[name].index:
                    row = normed_signals[name].loc[t]
                    any_active |= row.notna()
                    blended = blended.add((row * lam).fillna(0.0))
            # Symbols inactive across ALL strategies stay out of the trade
            # set (NaN → dropped by optimize_linear's dropna()).
            blended[~any_active] = np.nan

            # Step 3: per-asset linear CS allocation on the blended signal
            w = optimizer.optimize_linear(blended)
            signal_weights_history[t] = signal_weights

        elif args.method == "linear":
            w = optimizer.optimize_linear(alpha_t)

        elif args.method == "mean_variance":
            # Slice returns strictly past data (exclude row at t to avoid look-ahead)
            past_returns = returns_df.loc[:t].iloc[-(args.cov_lookback+1):-1]

            # Guard: if the warmup window is too short, fall back to linear
            if len(past_returns) < max(5, args.cov_lookback // 2):
                w = optimizer.optimize_linear(alpha_t)
                final_weights_list.append(w)
                active_asset_counts.append(int((w != 0).sum()))
                continue

            # --- FIX 2: Relaxed Data Cleaning ---
            # Instead of dropping any column with NaN, we only drop columns that are mostly NaN
            missing_pct = past_returns.isnull().mean()
            valid_cols = missing_pct[missing_pct <
                                     0.2].index  # Allow 20% missing

            # Fill remaining small gaps with 0.0 (neutral return assumption)
            valid_returns = past_returns[valid_cols].fillna(0.0)

            # Intersection with Active Signals
            valid_assets = valid_returns.columns.intersection(
                active_signals.index)

            # --- FIX 3: Fallback Logic ---
            if len(valid_assets) < 5:
                # If we don't have enough data for covariance, use Linear optimizer
                # This prevents the "8-9 asset" trap when data is spotty
                w = optimizer.optimize_linear(alpha_t)
            else:
                cov_matrix = valid_returns[valid_assets].cov()

                # Filter Zero Variance (Zombie Assets)
                variances = np.diag(cov_matrix)
                valid_mask = variances > 1e-8
                final_valid_assets = valid_assets[valid_mask]

                if len(final_valid_assets) < 2:
                    w = optimizer.optimize_linear(alpha_t)
                else:
                    cov_matrix = valid_returns[final_valid_assets].cov()
                    w_opt = optimizer.optimize_mean_variance(
                        alpha_t[final_valid_assets], cov_matrix)
                    w = w_opt.reindex(all_syms).fillna(0.0)

        # ── Final-signal EMA smoothing ───────────────────────────────────
        # Blend the freshly-optimised vector with yesterday's actual
        # positions to suppress per-asset z-score wiggle (the 80% of
        # cross_signal_mv turnover that came from underlying signal
        # noise, not λ flipping). After EMA the gross typically shrinks
        # because the two day's vectors aren't perfectly aligned —
        # rescale back to --max_leverage so the budget stays intact.
        if args.weight_ema < 1.0:
            w = w.reindex(all_syms).fillna(0.0)
            if prev_w is not None:
                w = args.weight_ema * w + (1.0 - args.weight_ema) * prev_w
                gross = float(w.abs().sum())
                if gross > 1e-8 and gross < args.max_leverage:
                    w = w * (args.max_leverage / gross)
            prev_w = w.copy()

        final_weights_list.append(w)
        active_asset_counts.append((w != 0).sum())

    final_weights = pd.DataFrame(final_weights_list)
    final_weights.index = all_ts
    # NaN may slip in when alpha_t had NaN cells (the new NaN-aware path).
    # Convert to explicit zeros before persisting so downstream consumers
    # don't have to be NaN-aware.
    final_weights = final_weights.reindex(columns=all_syms).fillna(0.0)

    avg_assets = np.mean(active_asset_counts) if active_asset_counts else 0
    print(f"\nAverage Active Assets: {avg_assets:.1f}")

    # 5. Save & Report
    # The optimized_weights parquet is saved with the FULL date range so
    # downstream consumers (other backtests, factor panels, live trading)
    # get uninterrupted history. --perf_start_date only trims the in-memory
    # reporting views below; on-disk artefacts stay full.
    save_path = os.path.join(
        base_dir, f"optimized_weights_{args.method}.parquet")
    final_weights.to_parquet(save_path)
    print(f"Optimized weights saved to: {save_path}")

    # Calculate Performance
    lagged_weights = final_weights.shift(1).fillna(0.0)
    aligned_returns = returns_df.fillna(0.0)

    port_rets_gross = (lagged_weights * aligned_returns).sum(axis=1)

    # Transaction cost: daily turnover (Σ|Δw|) × cost_bps. Apply the
    # deduction to the SAME day as the trade (the t-1 rebalance pays the
    # cost charged to day t's return), mirroring backtest/engine.py.
    turnover_series = final_weights.diff().abs().sum(axis=1).fillna(0.0)
    tc_drag = turnover_series * cost_bps
    port_rets = port_rets_gross - tc_drag

    # ── Trim reporting window if requested ───────────────────────────────────
    # We trim AFTER computing port_rets (so the t-1 lagged-weight applied to
    # the day-t return uses the genuine pre-warmup weight when the perf
    # window starts on the same day a strategy boots) — this preserves the
    # first day's return value rather than starting the equity curve from a
    # synthetic flat day.
    perf_weights = final_weights
    perf_rets = port_rets
    perf_rets_gross = port_rets_gross
    perf_turnover = turnover_series
    if args.perf_start_date:
        perf_cut = pd.Timestamp(args.perf_start_date)
        n_before = (final_weights.index < perf_cut).sum()
        perf_weights = final_weights.loc[final_weights.index >= perf_cut]
        perf_rets = port_rets.loc[port_rets.index >= perf_cut]
        perf_rets_gross = port_rets_gross.loc[
            port_rets_gross.index >= perf_cut]
        perf_turnover = turnover_series.loc[turnover_series.index >= perf_cut]
        if perf_weights.empty:
            raise ValueError(
                f"--perf_start_date={args.perf_start_date} leaves no rows "
                f"in the weight matrix (data ends "
                f"{final_weights.index.max().date()}).")
        print(f"\n[perf_start_date={args.perf_start_date}] trimming "
              f"reporting: dropped {n_before} pre-window rows; "
              f"reporting on {len(perf_weights)} rows "
              f"({perf_weights.index.min().date()} → "
              f"{perf_weights.index.max().date()}).")

    initial_cash = 10_000
    equity_curve = initial_cash * (1 + perf_rets).cumprod()
    equity_curve_gross = initial_cash * (1 + perf_rets_gross).cumprod()

    # Plot — uses the net curve as the canonical line so the saved
    # equity_<method>.png shows the realistic post-cost performance.
    eq_df = pd.DataFrame({
        "equity": equity_curve,
        "equity_gross": equity_curve_gross,
    })
    plot_path = os.path.join(base_dir, f"equity_{args.method}.png")
    plot_equity_curve(eq_df[["equity"]], plot_path)

    total_ret = (equity_curve.iloc[-1] / initial_cash) - 1
    total_ret_gross = (equity_curve_gross.iloc[-1] / initial_cash) - 1
    sharpe = ((perf_rets.mean() / perf_rets.std()) * (365 ** 0.5)
              if perf_rets.std() > 0 else 0.0)
    sharpe_gross = ((perf_rets_gross.mean() / perf_rets_gross.std())
                    * (365 ** 0.5) if perf_rets_gross.std() > 0 else 0.0)
    # Probabilistic Sharpe Ratio: P[true_SR > 0] given the OBSERVED daily
    # Sharpe, sample size, skewness, and kurtosis. Computed on the
    # per-period (daily) series, NOT the annualised SR — the engine's
    # helper expects the raw return series. PSR ≥ 0.95 is the customary
    # "statistically significant alpha" threshold (Bailey-Lopez de Prado).
    psr_gross = probabilistic_sharpe_ratio(perf_rets_gross, sr_benchmark=0.0)
    psr_net = probabilistic_sharpe_ratio(perf_rets, sr_benchmark=0.0)
    annual_tc_drag = float(perf_turnover.mean() * cost_bps * 365)

    print("\n==== OPTIMIZED PORTFOLIO RESULTS ====")
    print(f"Method:       {args.method}")
    if args.perf_start_date:
        print(f"Perf window:  {perf_weights.index.min().date()} → "
              f"{perf_weights.index.max().date()}  "
              f"({len(perf_weights)} days)")
    print(f"TC assumption: {fee_bps:.1f} bps fee + {slip_bps:.1f} bps "
          f"slippage  =  {(fee_bps+slip_bps):.1f} bps/turnover-unit")
    print(f"Avg daily turnover: {perf_turnover.mean():.4f}  "
          f"(annualised drag ≈ {annual_tc_drag*100:.2f}%)")
    print(f"{'':<14s}{'Gross':>14s}{'Net (after TC)':>18s}")
    print(f"{'Final Equity':<14s}"
          f"${equity_curve_gross.iloc[-1]:>13,.2f}"
          f"${equity_curve.iloc[-1]:>17,.2f}")
    print(f"{'Total Return':<14s}{total_ret_gross*100:>13.2f}%"
          f"{total_ret*100:>17.2f}%")
    print(f"{'Sharpe Ratio':<14s}{sharpe_gross:>14.2f}{sharpe:>18.2f}")
    print(f"{'PSR(SR*=0)':<14s}{psr_gross:>14.4f}{psr_net:>18.4f}")

    eq_df.to_csv(os.path.join(base_dir, f"equity_{args.method}.csv"))

    # ── Lag analysis: alpha decay under execution delay ──────────────────────
    # The baseline `port_rets` already uses a 1-day execution lag
    # (weight posted at t-1 trades the return on t). The lag-k variant
    # additionally delays execution by k days: weight from t-1-k applied
    # to return on t. If lag-1 retains most of the baseline performance
    # the alpha is slow-moving (high-capacity); if it collapses, the
    # alpha is short-lived (turnover-sensitive).
    print("\n==== ALPHA DECAY (EXECUTION-LAG ANALYSIS, NET OF TC) ====")
    print(f"{'lag':>4s}  {'days_late':>10s}  {'total_ret':>11s}  "
          f"{'sharpe':>8s}  {'PSR':>7s}  "
          f"{'ret_vs_lag0':>12s}  {'sharpe_vs_lag0':>15s}")
    lag_table = []
    lag0_ret = lag0_sharpe = None
    base_w = final_weights   # full-range weights so the shift can pull from
                             # genuine pre-window data without losing rows
    base_r = aligned_returns
    # Turnover schedule is invariant under the time-shift (lag-k just
    # delays everything), so the TC drag matches the gross return shift.
    base_to = final_weights.diff().abs().sum(axis=1).fillna(0.0)
    for k in (0, 1, 2, 3):
        wk = base_w.shift(1 + k).fillna(0.0)
        rk_gross = (wk * base_r).sum(axis=1)
        tc_k = base_to.shift(k).fillna(0.0) * cost_bps
        rk = rk_gross - tc_k
        if args.perf_start_date:
            rk = rk.loc[rk.index >= pd.Timestamp(args.perf_start_date)]
        if rk.empty:
            continue
        eq = (1 + rk).cumprod()
        tot = float(eq.iloc[-1] - 1.0)
        sh = float(rk.mean() / rk.std() * (365 ** 0.5)) if rk.std() > 0 else 0.0
        psr_k = probabilistic_sharpe_ratio(rk, sr_benchmark=0.0)
        if k == 0:
            lag0_ret, lag0_sharpe = tot, sh
            print(f"{k:>4d}  {1 + k:>10d}  {tot*100:>10.2f}%  "
                  f"{sh:>8.2f}  {psr_k:>7.4f}  "
                  f"{'baseline':>12s}  {'baseline':>15s}")
        else:
            ret_ratio = (tot / lag0_ret) if abs(lag0_ret) > 1e-9 else float('nan')
            sh_ratio = (sh / lag0_sharpe) if abs(lag0_sharpe) > 1e-9 else float('nan')
            print(f"{k:>4d}  {1 + k:>10d}  {tot*100:>10.2f}%  "
                  f"{sh:>8.2f}  {psr_k:>7.4f}  "
                  f"{ret_ratio*100:>11.1f}%  "
                  f"{sh_ratio*100:>14.1f}%")
        lag_table.append({
            "lag": k, "days_late": 1 + k,
            "total_return": tot, "sharpe": sh, "psr": psr_k,
        })
    lag_df = pd.DataFrame(lag_table)
    lag_path = os.path.join(base_dir, f"alpha_decay_{args.method}.csv")
    lag_df.to_csv(lag_path, index=False)
    print(f"\nAlpha-decay table saved → {lag_path}")

    # 6. Weight Distribution Analysis  (uses the trimmed view)
    print("\n--- Weight Distribution Analysis ---")
    analyze_weight_distribution(
        perf_weights,
        method=args.method,
        base_dir=base_dir,
        signal_weights_history=signal_weights_history or None,
    )


if __name__ == "__main__":
    main()
