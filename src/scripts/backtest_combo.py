import argparse
import os
import glob
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..core.utils import load_config, ensure_dir
from ..data.binance_futures_rest import fetch_futures_klines
from ..data.rolling_universe import RollingUniverse, build_symbol_active_mask
from ..backtest.reporting import plot_equity_curve
from ..portfolio.optimizer import PortfolioOptimizer


# Files saved by the backtest engine and train_ebm_signal that are NOT
# strategy weight matrices (exclude these from auto-discovery).
_EXCLUDE_PREFIXES = ("optimized_weights_",)
_EXCLUDE_EXACT    = {"ebm_predictions.parquet"}


def load_and_align_strategies(run_dir: str,
                               strategies: list[str] | None = None):
    """
    Loads strategy weight parquets from run_dir and aligns them to a master
    timeline.

    Parameters
    ----------
    run_dir    : directory containing *.parquet weight files
    strategies : explicit list of base names (e.g. ["momentum","reversal","ebm"]).
                 If None, all *.parquet files are auto-discovered (excluding
                 optimized_weights_* and ebm_predictions.parquet).
    """
    if strategies:
        files = []
        for name in strategies:
            p = os.path.join(run_dir, f"{name}.parquet")
            if not os.path.exists(p):
                print(f"  [warn] {p} not found — skipping '{name}'.")
            else:
                files.append(p)
    else:
        all_files = glob.glob(os.path.join(run_dir, "*.parquet"))
        files = [
            f for f in all_files
            if os.path.basename(f) not in _EXCLUDE_EXACT
            and not any(os.path.basename(f).startswith(pfx)
                        for pfx in _EXCLUDE_PREFIXES)
        ]

    if not files:
        raise FileNotFoundError(f"No weight files found in {run_dir}")

    print(
        f"Found {len(files)} strategies: {[os.path.basename(f) for f in files]}")

    raw_strategies = {}
    for f in files:
        name = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        raw_strategies[name] = df

    # Create Union Index/Columns
    master_ts = pd.Index([])
    master_cols = pd.Index([])
    for df in raw_strategies.values():
        master_ts = master_ts.union(df.index)
        master_cols = master_cols.union(df.columns)

    master_ts = master_ts.sort_values().unique()
    master_cols = master_cols.sort_values().unique()

    # Align
    aligned_strategies = {}
    for name, df in raw_strategies.items():
        aligned_df = df.reindex(
            index=master_ts, columns=master_cols).fillna(0.0)
        aligned_strategies[name] = aligned_df

    return aligned_strategies, master_ts, master_cols


def fetch_all_prices(symbols, start_date, end_date):
    """
    Fetches prices for ALL symbols involved to calculate Covariance.
    """
    print(
        f"Fetching price history for {len(symbols)} assets (for Covariance/PnL)...")
    price_frames = {}
    for sym in tqdm(symbols):
        try:
            df = fetch_futures_klines(sym, "1d", start_date, end_date)
            if not df.empty:
                if not pd.api.types.is_datetime64_any_dtype(df['ts']):
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('ts', inplace=True)
                # Prefer futures_close
                price = df['futures_close'] if 'futures_close' in df.columns else df['close']
                price_frames[sym] = price
        except Exception as e:
            print(f"Error fetching {sym}: {e}")

    if not price_frames:
        return pd.DataFrame()

    prices_df = pd.DataFrame(price_frames).ffill()
    return prices_df


def compute_strategy_returns(
    aligned_strategies: dict,
    returns_df: pd.DataFrame,
    all_ts: pd.Index,
) -> pd.DataFrame:
    """
    Computes the daily portfolio return series for each strategy using
    lagged weights — return on day t = weights posted at end-of-day t-1
    dotted with asset returns on day t.

    Returns a DataFrame of shape (n_dates, n_strategies).
    """
    common_cols = returns_df.columns
    strat_rets = {}
    for name, w_df in aligned_strategies.items():
        w_aligned = w_df.reindex(index=all_ts, columns=common_cols).fillna(0.0)
        lagged_w = w_aligned.shift(1).fillna(0.0)
        r_aligned = returns_df.reindex(index=all_ts, columns=common_cols).fillna(0.0)
        strat_rets[name] = (lagged_w * r_aligned).sum(axis=1)
    return pd.DataFrame(strat_rets, index=all_ts)


def optimize_signal_weights(
    strat_ret_window: pd.DataFrame,
    lambda_risk: float,
) -> pd.Series:
    """
    Mean-variance optimization in strategy space.

    Maximises: μᵀλ − lambda_risk · λᵀΣλ
    Subject to: λ ≥ 0,  sum(λ) = 1

    μ = mean daily return per strategy over the lookback window.
    Σ = covariance of daily strategy returns.

    Falls back to equal weights on solver failure or insufficient data.
    """
    names = strat_ret_window.columns.tolist()
    n = len(names)

    if n == 1:
        return pd.Series(1.0, index=names)

    mu = strat_ret_window.mean().values
    Sigma = strat_ret_window.cov().values
    Sigma += np.eye(n) * 1e-8   # small diagonal regularisation → guaranteed PSD

    lam = cp.Variable(n)
    objective = cp.Maximize(
        mu @ lam - lambda_risk * cp.quad_form(lam, cp.psd_wrap(Sigma))
    )
    constraints = [lam >= 0, cp.sum(lam) == 1]

    try:
        cp.Problem(objective, constraints).solve()
        if lam.value is None:
            return pd.Series(1.0 / n, index=names)
        result = pd.Series(lam.value, index=names).clip(lower=0)
        total = result.sum()
        return result / total if total > 1e-8 else pd.Series(1.0 / n, index=names)
    except Exception:
        return pd.Series(1.0 / n, index=names)


def analyze_weight_distribution(
    final_weights: pd.DataFrame,
    method: str,
    base_dir: str,
    signal_weights_history: dict | None = None,
):
    """
    Generates weight distribution analysis for the optimized portfolio.

    Outputs
    -------
    weight_stats_daily_{method}.csv   per-date portfolio metrics
    weight_stats_assets_{method}.csv  per-asset average weight statistics
    weight_distribution_{method}.png  6-panel (or 7-panel for cross_signal_mv) figure
    """
    import warnings
    warnings.filterwarnings("ignore")

    w = final_weights.copy()
    active = w[w.abs().sum(axis=1) > 1e-8]   # dates with at least one position

    # ── Per-date stats ────────────────────────────────────────────────────────
    daily = pd.DataFrame(index=active.index)
    daily["gross_leverage"]  = active.abs().sum(axis=1)
    daily["net_exposure"]    = active.sum(axis=1)
    daily["n_long"]          = (active > 1e-6).sum(axis=1)
    daily["n_short"]         = (active < -1e-6).sum(axis=1)
    # Effective N = 1 / HHI — inverse of Herfindahl concentration index
    sq_sum = (active ** 2).sum(axis=1)
    daily["effective_n"]     = (1.0 / sq_sum.replace(0, np.nan)).fillna(0)
    # Daily turnover = sum of |Δweight|
    daily["turnover"]        = w.diff().abs().sum(axis=1)

    daily_path = os.path.join(base_dir, f"weight_stats_daily_{method}.csv")
    daily.to_csv(daily_path)
    print(f"Daily weight stats saved → {daily_path}")

    # ── Per-asset stats ───────────────────────────────────────────────────────
    long_w  = active.clip(lower=0).replace(0, np.nan)
    short_w = active.clip(upper=0).replace(0, np.nan).abs()

    asset_stats = pd.DataFrame({
        "avg_long_weight":  long_w.mean(),
        "avg_short_weight": short_w.mean(),
        "long_days":        (active > 1e-6).sum(),
        "short_days":       (active < -1e-6).sum(),
        "long_freq":        (active > 1e-6).mean(),
        "short_freq":       (active < -1e-6).mean(),
        "avg_abs_weight":   active.abs().replace(0, np.nan).mean(),
    }).dropna(how="all").sort_values("avg_abs_weight", ascending=False)

    asset_path = os.path.join(base_dir, f"weight_stats_assets_{method}.csv")
    asset_stats.to_csv(asset_path)
    print(f"Asset weight stats saved  → {asset_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    has_signal_hist = bool(signal_weights_history)
    n_rows = 4 if not has_signal_hist else 5
    fig = plt.figure(figsize=(16, 4.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.50, wspace=0.35)

    # 1. Gross leverage + net exposure (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(daily.index, daily["gross_leverage"],
                     alpha=0.25, color="#2196F3", label="Gross Leverage")
    ax1.plot(daily.index, daily["gross_leverage"],
             color="#2196F3", lw=1.2, label="_nolegend_")
    ax1b = ax1.twinx()
    ax1b.plot(daily.index, daily["net_exposure"],
              color="#FF5722", lw=1.0, linestyle="--", label="Net Exposure")
    ax1b.axhline(0, color="gray", lw=0.5, linestyle=":")
    ax1.set_ylabel("Gross Leverage", color="#2196F3")
    ax1b.set_ylabel("Net Exposure", color="#FF5722")
    ax1.set_title(
        f"Gross Leverage & Net Exposure  |  "
        f"avg gross={daily['gross_leverage'].mean():.3f}  "
        f"avg net={daily['net_exposure'].mean():.3f}"
    )
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax1.tick_params(axis="x", labelrotation=30, labelsize=8)

    # 2. Long / short count  (left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(daily.index, daily["n_long"],
             color="#4CAF50", lw=1.2, label="N Long")
    ax2.plot(daily.index, daily["n_short"],
             color="#F44336", lw=1.2, label="N Short")
    ax2.set_ylabel("Asset Count")
    ax2.set_title(
        f"Long / Short Asset Counts  |  "
        f"avg long={daily['n_long'].mean():.1f}  "
        f"avg short={daily['n_short'].mean():.1f}"
    )
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)

    # 3. Effective N & daily turnover  (right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(daily.index, daily["effective_n"],
             color="#9C27B0", lw=1.2, label="Effective N")
    ax3b = ax3.twinx()
    ax3b.plot(daily.index, daily["turnover"].rolling(5).mean(),
              color="#FF9800", lw=1.0, linestyle="--",
              label="5d avg Turnover")
    ax3.set_ylabel("Effective N (1/HHI)", color="#9C27B0")
    ax3b.set_ylabel("Turnover (|Δw|)", color="#FF9800")
    ax3.set_title(
        f"Diversification & Turnover  |  "
        f"avg eff_N={daily['effective_n'].mean():.1f}  "
        f"avg turnover={daily['turnover'].mean():.3f}"
    )
    lines_a, labs_a = ax3.get_legend_handles_labels()
    lines_b, labs_b = ax3b.get_legend_handles_labels()
    ax3.legend(lines_a + lines_b, labs_a + labs_b, fontsize=8)
    ax3.tick_params(axis="x", labelrotation=30, labelsize=8)

    # 4. Weight magnitude histogram — longs vs shorts  (left)
    ax4 = fig.add_subplot(gs[2, 0])
    all_longs  = active.values[active.values >  1e-6].flatten()
    all_shorts = active.values[active.values < -1e-6].flatten()
    bins = np.linspace(0, active.abs().max().max() * 1.05, 40)
    ax4.hist(all_longs,   bins=bins, color="#4CAF50", alpha=0.65, label="Long weights")
    ax4.hist(all_shorts * -1, bins=bins, color="#F44336", alpha=0.65, label="Short weights")
    ax4.set_xlabel("|Weight|")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Weight Magnitude Distribution")
    ax4.legend(fontsize=8)
    ax4.axvline(np.mean(np.abs(all_longs))  if len(all_longs)  else 0,
                color="#2E7D32", lw=1.0, linestyle="--", label="_nolegend_")
    ax4.axvline(np.mean(np.abs(all_shorts)) if len(all_shorts) else 0,
                color="#B71C1C", lw=1.0, linestyle="--", label="_nolegend_")

    # 5. Top 15 assets by avg |weight|  (right)
    ax5 = fig.add_subplot(gs[2, 1])
    top_assets = asset_stats["avg_abs_weight"].dropna().head(15).sort_values()
    bar_c = []
    for sym in top_assets.index:
        lf = asset_stats.loc[sym, "long_freq"]
        sf = asset_stats.loc[sym, "short_freq"]
        bar_c.append("#4CAF50" if lf >= sf else "#F44336")
    ax5.barh(top_assets.index, top_assets.values, color=bar_c, alpha=0.80)
    ax5.set_xlabel("Avg |Weight|")
    ax5.set_title("Top 15 Assets by Avg |Weight|\n(green = more often long, red = more often short)")
    ax5.tick_params(axis="y", labelsize=7)

    # 6. Long/short weight concentration over time — stacked bars  (full width)
    ax6 = fig.add_subplot(gs[3, :])
    roll_long  = (active.clip(lower=0)  > 1e-6).sum(axis=1).rolling(21).mean()
    roll_short = (active.clip(upper=0) < -1e-6).sum(axis=1).rolling(21).mean()
    ax6.stackplot(daily.index,
                  [roll_long.reindex(daily.index).fillna(0),
                   roll_short.reindex(daily.index).fillna(0)],
                  labels=["21d avg N Long", "21d avg N Short"],
                  colors=["#A5D6A7", "#EF9A9A"], alpha=0.85)
    ax6.set_ylabel("Asset count (21d rolling avg)")
    ax6.set_title("Portfolio Breadth Over Time")
    ax6.legend(fontsize=8, loc="upper left")
    ax6.tick_params(axis="x", labelrotation=30, labelsize=8)

    # 7. Strategy λ weights over time — stacked area (cross_signal_mv only)
    if has_signal_hist:
        ax7 = fig.add_subplot(gs[4, :])
        lam_df = pd.DataFrame(signal_weights_history).T.sort_index()
        lam_df = lam_df.reindex(daily.index).ffill().fillna(0.0)

        palette = ["#2196F3", "#FF9800", "#4CAF50",
                   "#F44336", "#9C27B0", "#00BCD4", "#795548"]
        colors7 = palette[:len(lam_df.columns)]
        ax7.stackplot(lam_df.index,
                      [lam_df[c].values for c in lam_df.columns],
                      labels=lam_df.columns.tolist(),
                      colors=colors7, alpha=0.80)
        ax7.set_ylim(0, 1)
        ax7.set_ylabel("Strategy weight λ")
        ax7.set_title("Cross-Signal MV: Strategy Allocation Over Time")
        ax7.legend(fontsize=8, loc="upper left")
        ax7.tick_params(axis="x", labelrotation=30, labelsize=8)

    fig.suptitle(
        f"Weight Distribution Analysis  |  method={method}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    plot_path = os.path.join(base_dir, f"weight_distribution_{method}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Weight distribution plot saved → {plot_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Weight Distribution Summary  ({method})")
    print(sep)
    print(f"  Active days       : {len(daily)} / {len(final_weights)}")
    print(f"  Avg gross leverage: {daily['gross_leverage'].mean():.4f}")
    print(f"  Avg net exposure  : {daily['net_exposure'].mean():.4f}")
    print(f"  Avg N long        : {daily['n_long'].mean():.1f}")
    print(f"  Avg N short       : {daily['n_short'].mean():.1f}")
    print(f"  Avg effective N   : {daily['effective_n'].mean():.1f}")
    print(f"  Avg daily turnover: {daily['turnover'].mean():.4f}")
    print(f"  Max single weight : {active.values.max():.4f}")
    print(f"  Min single weight : {active.values.min():.4f}")
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True,
                    help="Run ID where strategy weights are saved")
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date", required=True)
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

    args = ap.parse_args()

    base_dir = f"./reports/strategies/{args.run_id}"

    # 1. Load Strategy Signals
    strategies, all_ts, all_syms = load_and_align_strategies(
        base_dir, args.strategies)

    # 1b. Safety-net epoch mask — zero weights outside each symbol's active epoch.
    # Each individual strategy (momentum, reversal, ebm) already applies this mask
    # at generation time; this pass catches any future strategy that doesn't.
    ru = RollingUniverse()
    if not ru.is_empty():
        ru_epochs = ru.get_epochs(args.start_date, args.end_date)
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
    print("\n--- Computing Composite Signal ---")
    n_strategies = len(strategies)
    composite_alpha = pd.DataFrame(0.0, index=all_ts, columns=all_syms)
    normed_signals = {}   # kept for cross_signal_mv blend

    for name, df in strategies.items():
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        normed = df.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
        normed_signals[name] = normed
        composite_alpha = composite_alpha.add(normed)

    composite_alpha = composite_alpha / n_strategies

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

    # Track stats — counted over ALL dates (flat days contribute 0)
    active_asset_counts = []

    for t in tqdm(all_ts):
        if t not in composite_alpha.index:
            final_weights_list.append(pd.Series(0.0, index=all_syms, name=t))
            continue

        alpha_t = composite_alpha.loc[t]
        # Filter strictly 0 signals to speed up optimization
        active_signals = alpha_t[alpha_t != 0]

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
                signal_weights = pd.Series(
                    1.0 / n_strategies, index=list(normed_signals.keys()))
            else:
                signal_weights = optimize_signal_weights(past_strat, args.risk_aversion)

            # Step 2: blend CS z-scored signals using optimal λ
            blended = pd.Series(0.0, index=all_syms)
            for name, lam in signal_weights.items():
                if t in normed_signals[name].index:
                    blended = blended.add(normed_signals[name].loc[t] * lam)

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

        final_weights_list.append(w)
        active_asset_counts.append((w != 0).sum())

    final_weights = pd.DataFrame(final_weights_list)
    final_weights.index = all_ts

    avg_assets = np.mean(active_asset_counts) if active_asset_counts else 0
    print(f"\nAverage Active Assets: {avg_assets:.1f}")

    # 5. Save & Report
    save_path = os.path.join(
        base_dir, f"optimized_weights_{args.method}.parquet")
    final_weights.to_parquet(save_path)
    print(f"Optimized weights saved to: {save_path}")

    # Calculate Performance
    lagged_weights = final_weights.shift(1).fillna(0.0)
    aligned_returns = returns_df.fillna(0.0)

    port_rets = (lagged_weights * aligned_returns).sum(axis=1)

    initial_cash = 10_000
    equity_curve = initial_cash * (1 + port_rets).cumprod()

    # Plot
    eq_df = pd.DataFrame({'equity': equity_curve})
    plot_path = os.path.join(base_dir, f"equity_{args.method}.png")
    plot_equity_curve(eq_df, plot_path)

    total_ret = (equity_curve.iloc[-1] / initial_cash) - 1
    sharpe = (port_rets.mean() / port_rets.std()) * \
        (365**0.5) if port_rets.std() > 0 else 0

    print("\n==== OPTIMIZED PORTFOLIO RESULTS ====")
    print(f"Method:       {args.method}")
    print(f"Final Equity: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    eq_df.to_csv(os.path.join(base_dir, f"equity_{args.method}.csv"))

    # 6. Weight Distribution Analysis
    print("\n--- Weight Distribution Analysis ---")
    analyze_weight_distribution(
        final_weights,
        method=args.method,
        base_dir=base_dir,
        signal_weights_history=signal_weights_history or None,
    )


if __name__ == "__main__":
    main()
