import pandas as pd
from ..core.types import Order, BacktestResult
from ..portfolio.paperbroker import PaperBroker
import numpy as np
from scipy import stats as scipy_stats
from ..core.utils import ensure_dir
import os


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """
    Compute the Probabilistic Sharpe Ratio (PSR) — the probability that
    the true Sharpe ratio exceeds *sr_benchmark*, adjusting for skewness
    and kurtosis of the observed return series.

    Bailey & López de Prado (2012).
    """
    n = len(returns)
    if n < 3:
        return np.nan
    sr = returns.mean() / (returns.std() + 1e-12)
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())  # excess kurtosis
    sr_std = np.sqrt(
        (1 - skew * sr + (kurt / 4) * sr ** 2) / (n - 1)
    )
    if sr_std < 1e-12:
        return np.nan
    z = (sr - sr_benchmark) / sr_std
    return float(scipy_stats.norm.cdf(z))


def run_multi_asset(data: dict[str, pd.DataFrame], strategy, cfg: dict) -> BacktestResult:
    broker = PaperBroker(
        cash=cfg["backtest"]["initial_cash"],
        fee_bps=cfg["backtest"]["fee_bps"],
        slippage_bps=cfg["backtest"]["slippage_bps"],
    )
    rebalance_period = pd.to_timedelta(cfg["backtest"]["rebalance_period"])
    symbols = list(data.keys())

    all_ts = pd.concat([df["ts"]
                       for df in data.values()]).sort_values().unique()
    timeline = pd.to_datetime(all_ts, unit='ms')

    equities = []
    trades = []
    all_score_components = []
    last_rebalance_ts = timeline[0] if len(
        timeline) > 0 else pd.Timestamp.now()

    for i, ts in enumerate(timeline):
        current_prices = {}
        latest_rows = {}
        for sym in symbols:
            price_series = data[sym][data[sym]["ts"] <= all_ts[i]]
            if not price_series.empty:
                latest_row = price_series.iloc[-1]
                current_prices[sym] = latest_row["futures_close"]
                latest_rows[sym] = latest_row

        assets_to_price = {
            s for s in broker.positions if broker.positions[s]['qty'] != 0}

        # --- Market to Market ---
        if all(s in current_prices for s in assets_to_price):
            snap = broker.mark_to_market(current_prices)

            # Capture Daily Regime Data
            regimes = {
                'volatility_regime': 'Unknown',
                'trend_regime': 'Unknown',
                'skew_regime': 'Unknown'
            }
            if latest_rows:
                sample_row = next(iter(latest_rows.values()))
                regimes['volatility_regime'] = sample_row.get(
                    'volatility_regime', 'Unknown')
                regimes['trend_regime'] = sample_row.get(
                    'trend_regime', 'Unknown')
                regimes['skew_regime'] = sample_row.get(
                    'skew_regime', 'Unknown')

            equities.append({
                "ts": ts,
                "equity": snap["equity"],
                **regimes
            })

        if ts < last_rebalance_ts + rebalance_period:
            continue

        last_rebalance_ts = ts

        strategy_data = {sym: df[df["ts"] <= all_ts[i]]
                         for sym, df in data.items()}
        signals, score_components = strategy.on_rebalance(strategy_data)

        if score_components:
            for symbol, components in score_components.items():
                components['ts'] = ts
                components['symbol'] = symbol

                # --- REGIMES ---
                latest_row_for_sym = latest_rows.get(symbol)
                if latest_row_for_sym is not None:
                    components['volatility_regime'] = latest_row_for_sym.get(
                        'volatility_regime', 'Unknown')
                    components['trend_regime'] = latest_row_for_sym.get(
                        'trend_regime', 'Unknown')
                    components['skew_regime'] = latest_row_for_sym.get(
                        'skew_regime', 'Unknown')
                else:
                    components['volatility_regime'] = 'Unknown'
                    components['trend_regime'] = 'Unknown'
                    components['skew_regime'] = 'Unknown'

                # --- V-- NEW: INJECT POSITION & PRICE FOR PNL ANALYSIS --V ---
                # This allows us to calculate exactly how much PnL this asset generates
                current_pos = broker.positions.get(symbol, {})
                components['position_qty'] = current_pos.get('qty', 0.0)
                components['close_price'] = current_prices.get(symbol, np.nan)
                # -------------------------------------------------------------

                all_score_components.append(components)

        if not signals:
            continue

        snap = broker.mark_to_market(current_prices)
        for symbol, sig in signals.items():
            target_weight = sig.weight
            last_price = current_prices.get(symbol)
            if last_price is None:
                continue

            notional = target_weight * snap["equity"]
            current_qty = broker.positions.get(symbol, {"qty": 0.0})["qty"]
            target_qty = notional / last_price
            delta = target_qty - current_qty

            if abs(delta) < 1e-12:
                continue

            side = "BUY" if delta > 0 else "SELL"
            order: Order = {"symbol": symbol, "side": side,
                            "qty": abs(delta), "order_type": "MARKET"}

            current_row = latest_rows.get(symbol)
            regimes = {
                'volatility_regime': current_row.get('volatility_regime', 'Unknown'),
                'trend_regime': current_row.get('trend_regime', 'Unknown'),
                'skew_regime': current_row.get('skew_regime', 'Unknown')
            }

            fill = broker.execute(order, last_price, regimes)

            if fill.get("pnl", 0.0) != 0.0:
                trades.append({
                    "ts": ts,
                    "symbol": symbol,
                    "side": side,
                    "qty": fill["qty"],
                    "price": fill["price"],
                    "pnl": fill["pnl"],
                    "volatility_regime": fill["volatility_regime"],
                    "trend_regime": fill["trend_regime"],
                    "skew_regime": fill["skew_regime"]
                })

    if equities:
        eq = pd.DataFrame(equities).drop_duplicates(
            "ts").set_index("ts").sort_index()
    else:
        eq = pd.DataFrame({'ts': [timeline[0] if len(timeline) > 0 else pd.Timestamp.now(
        )], 'equity': [cfg['backtest']['initial_cash']]}).set_index('ts')

    tr = pd.DataFrame(trades)
    score_df = pd.DataFrame(all_score_components)

    ret = eq["equity"].pct_change().fillna(0.0)

    summary = {
        "final_equity": float(eq["equity"].iloc[-1]) if not eq.empty else cfg['backtest']['initial_cash'],
        "return_pct": float((eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100) if not eq.empty and eq["equity"].iloc[0] != 0 else 0.0,
        "sharpe_daily": float((ret.mean() / (ret.std() + 1e-12)) * (365 ** 0.5)) if len(eq) > 2 else 0.0,
        "prob_sharpe_ratio": probabilistic_sharpe_ratio(ret) if len(eq) > 2 else np.nan,
        "daily_win_rate_pct": float((ret > 0).mean() * 100) if not ret.empty else 0.0,
        "trades": int(len(tr))
    }

    return BacktestResult(equity_curve=eq, trades=tr, summary=summary, score_history=score_df)


def run_vectorized_backtest(
    data: dict[str, pd.DataFrame],
    strategy,
    cfg: dict,
    run_id: str = 'default',
    file_name: str = 'default',
    epoch_mask_df: "pd.DataFrame | None" = None,
) -> BacktestResult:
    """
    Calculates PnL using (Signal * Return) - Costs.
    Uses 'generate_all_signals' for true vectorization.

    epoch_mask_df : optional wide boolean DataFrame (index=dates, columns=symbols).
        True  → symbol is in the active epoch on that date (weight allowed).
        False → weight forced to 0.  Applied AFTER signal generation so that
                lookback windows can use pre-epoch price history without penalty.
    """
    initial_cash = cfg["backtest"]["initial_cash"]
    cost_bps = (cfg["backtest"]["fee_bps"] +
                cfg["backtest"]["slippage_bps"]) / 10000

    # 1. Align Prices (for Returns Calculation)
    # Build as DatetimeIndex explicitly — pd.Index([]).union(...) can produce an
    # object Index in pandas 2.0+, which prevents epoch_mask_df from aligning.
    all_ts_set: set = set()
    for df in data.values():
        if not df.empty:
            all_ts_set.update(df['ts'].tolist())
    all_ts = pd.DatetimeIndex(sorted(all_ts_set))

    # Wide Close Prices
    closes_dict = {sym: df.set_index(
        'ts')['futures_close'] for sym, df in data.items()}
    prices_df = pd.DataFrame(closes_dict).reindex(all_ts).ffill()

    # 2. Generate Signals (Vectorized)
    # epoch_mask_df is forwarded so the strategy can exclude inactive symbols
    # from cross-sectional operations (rank, z-score) while still using their
    # pre-epoch price history for rolling lookback computations.
    print("Generating signals (Vectorized)...")
    weights_df, score_history_df = strategy.generate_all_signals(
        data, epoch_mask_df=epoch_mask_df)

    # Apply rolling-universe epoch mask to weights (NOT to input price data).
    # This lets lookback windows use pre-epoch price history while still
    # ensuring the portfolio holds 0 weight outside a symbol's active epoch.
    if epoch_mask_df is not None and not epoch_mask_df.empty:
        common_syms = weights_df.columns.intersection(epoch_mask_df.columns)
        if len(common_syms):
            aligned_mask = epoch_mask_df.reindex(
                index=weights_df.index, columns=weights_df.columns
            ).fillna(False)
            weights_df = weights_df.where(aligned_mask, other=0.0)

            # Apply the same epoch mask to score_history_df so that the
            # quantile analysis only sees positions the portfolio actually held.
            if (score_history_df is not None and not score_history_df.empty
                    and 'ts' in score_history_df.columns
                    and 'symbol' in score_history_df.columns
                    and 'position_qty' in score_history_df.columns):
                # Stack the boolean mask into a long-form Series keyed by (ts, symbol)
                active_long = aligned_mask.stack()
                active_long.index.names = ['ts', 'symbol']
                # Build a lookup index from score_history rows
                sh_keys = pd.MultiIndex.from_arrays(
                    [score_history_df['ts'], score_history_df['symbol']],
                    names=['ts', 'symbol'],
                )
                is_active = active_long.reindex(
                    sh_keys, fill_value=False).values
                score_history_df.loc[~is_active, 'position_qty'] = 0.0

    # NaN means no signal → flatten the position (treat as 0 weight).
    # We deliberately do NOT ffill here: holding a stale weight on missing signal days
    # would silently overleverage the portfolio.
    weights_df = weights_df.reindex(all_ts).fillna(0.0)
    output_dir = ensure_dir(f"./reports/strategies/{run_id}")
    weights_path = os.path.join(output_dir, f"{file_name}.parquet")

    weights_df.to_parquet(weights_path)
    print(f"Weights saved to: {weights_path}")
    # 3. Calculate Vectorized PnL
    returns_df = prices_df.pct_change().fillna(0.0)

    def _build_equity(lag: int) -> pd.Series:
        w = weights_df.shift(lag).fillna(0.0)
        gross = (w * returns_df).sum(axis=1)
        to = weights_df.diff().abs().sum(axis=1).fillna(0.0)
        net = gross - to * cost_bps
        return initial_cash * (1 + net).cumprod()

    # Lag weights by 1: Weights calculated at T act on Returns at T+1
    equity_lag1 = _build_equity(1)
    equity_lag5 = _build_equity(2)
    equity_lag10 = _build_equity(3)

    port_rets_net = (equity_lag1 / equity_lag1.shift(1) - 1).fillna(0.0)

    # 4. Construct Equity Curve
    equity_curve = equity_lag1

    eq_df = pd.DataFrame({
        'equity': equity_lag1,
        'equity_lag2': equity_lag5,
        'equity_lag3': equity_lag10,
        'ts': equity_lag1.index,
    })

    # 5. Summary
    summary = {
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_cash,
        "return_pct": float((equity_lag1.iloc[-1] / equity_lag1.iloc[0] - 1) * 100) if not equity_lag1.empty else 0.0,
        "sharpe_daily": float(port_rets_net.mean() / port_rets_net.std() * (365**0.5)) if port_rets_net.std() != 0 else 0.0,
        "prob_sharpe_ratio": probabilistic_sharpe_ratio(port_rets_net),
        "turnover_avg": float(weights_df.diff().abs().sum(axis=1).fillna(0.0).mean())
    }

    return BacktestResult(
        equity_curve=eq_df,
        trades=pd.DataFrame(),  # Empty for vectorized
        summary=summary,
        score_history=score_history_df  # Populated for analysis
    )
