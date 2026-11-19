import numpy as np

from bayesian_investment.strategies.registry import build_strategies


def test_twelve_strategies_generate_bounded_weights(prices):
    strategies = build_strategies()
    assert len(strategies) >= 12
    for strategy in strategies:
        weights = strategy.generate_weights(prices)
        assert weights.shape == prices.shape
        assert np.isfinite(weights).all().all()
        assert (weights.abs().sum(axis=1) <= 1.000001).all()
