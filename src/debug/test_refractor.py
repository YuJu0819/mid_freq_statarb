import pandas as pd
import numpy as np
import pytest
from src.alpha.pipeline import AlphaPipeline
from src.portfolio.optimizer import PortfolioOptimizer
from src.strategy.distributed import DistributedStrategy

# --- 1. Test the Alpha Pipeline ---


def test_alpha_pipeline():
    print("\n--- Testing Alpha Pipeline ---")
    # Create fake data for 3 assets over 100 days
    dates = pd.date_range(start="2024-01-01", periods=100)
    data = {}

    for sym in ['BTC', 'ETH', 'SOL']:
        df = pd.DataFrame(index=dates)
        df['futures_close'] = np.random.randn(100).cumsum() + 100
        df['open_interest'] = np.random.randn(100).cumsum() + 1000
        df['ts'] = dates
        data[sym] = df

    pipeline = AlphaPipeline(lookback=10, smooth=3, vol_lookback=10)
    scores = pipeline.run(data)

    print("Scores Output:\n", scores)

    assert not scores.empty, "Pipeline returned empty scores!"
    assert len(scores) == 3, "Should have scores for all 3 assets"
    print("✅ Alpha Pipeline Passed")

# --- 2. Test the Optimizer (The Math) ---


def test_optimizer():
    print("\n--- Testing Portfolio Optimizer ---")

    # Scenario: BTC is great, ETH is okay, SOL is terrible
    alpha_scores = pd.Series({'BTC': 2.5, 'ETH': 0.5, 'SOL': -3.0})

    optimizer = PortfolioOptimizer(
        max_leverage=1.0, max_position=0.4, dollar_neutral=True)
    weights = optimizer.optimize(alpha_scores)

    print("Optimized Weights:\n", weights)

    # Validation Rules
    net_exposure = weights.sum()
    gross_exposure = weights.abs().sum()
    max_w = weights.abs().max()

    print(f"Net Exposure (Target 0.0): {net_exposure:.4f}")
    print(f"Gross Exposure (Target 1.0): {gross_exposure:.4f}")

    assert abs(net_exposure) < 1e-4, "Optimizer failed Dollar Neutrality!"
    assert gross_exposure <= 1.0 + 1e-4, "Optimizer exceeded Leverage!"
    assert max_w <= 0.4 + 1e-4, "Optimizer exceeded Position Limit!"

    # Logic check: BTC should be Long, SOL should be Short
    assert weights['BTC'] > 0, "BTC should be Long"
    assert weights['SOL'] < 0, "SOL should be Short"

    print("✅ Optimizer Passed")

# --- 3. Test Full Strategy Integration ---


def test_strategy_integration():
    print("\n--- Testing Full Strategy ---")
    # Mock Data
    dates = pd.date_range(start="2024-01-01", periods=50)
    data = {
        'BTC': pd.DataFrame({'futures_close': np.arange(50), 'open_interest': np.arange(50)}, index=dates),
        'ETH': pd.DataFrame({'futures_close': np.arange(50)[::-1], 'open_interest': np.arange(50)}, index=dates)
    }

    strat = DistributedStrategy(lookback=10)
    signals, debug = strat.on_rebalance(data)

    print("Generated Signals:", signals.keys())

    assert len(signals) > 0, "Strategy failed to produce signals"
    print("✅ Full Strategy Integration Passed")


if __name__ == "__main__":
    test_alpha_pipeline()
    test_optimizer()
    test_strategy_integration()
