import pandas as pd
import pytest

from bayesian_investment.backtesting.engine import backtest


def test_next_bar_and_cost_accounting():
    idx = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    result = backtest(prices, weights, fee_bps=10, slippage_bps=5)
    assert result.returns.iloc[0] == pytest.approx(0.0)
    assert result.returns.iloc[1] == pytest.approx(0.0985)
    assert result.gross_returns.iloc[1] == pytest.approx(0.1)
    assert result.turnover.sum() == pytest.approx(1)


def test_leverage_is_enforced(prices):
    result = backtest(prices, prices * 0 + 2)
    assert (result.weights.abs().sum(axis=1) <= 1.000001).all()
