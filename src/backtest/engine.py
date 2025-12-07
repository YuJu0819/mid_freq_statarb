import pandas as pd
from dataclasses import dataclass
from ..core.types import Order
from ..portfolio.paperbroker import PaperBroker
import numpy as np  # Added for NaN handling


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    summary: dict
    score_history: pd.DataFrame


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
        "daily_win_rate_pct": float((ret > 0).mean() * 100) if not ret.empty else 0.0,
        "trades": int(len(tr))
    }

    return BacktestResult(equity_curve=eq, trades=tr, summary=summary, score_history=score_df)
