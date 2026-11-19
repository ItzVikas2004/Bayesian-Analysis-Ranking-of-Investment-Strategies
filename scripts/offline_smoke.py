"""Small synthetic end-to-end smoke run for network-free CI only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_investment.backtesting.engine import backtest
from bayesian_investment.bayesian.model import fit_hierarchical_model
from bayesian_investment.bayesian.ranking import posterior_rankings
from bayesian_investment.portfolio.construction import walk_forward_score_portfolio
from bayesian_investment.strategies.registry import build_strategies


def main() -> None:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-01", periods=350)
    columns = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
    innovations = rng.normal(0.0003, 0.01, (len(index), len(columns)))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(innovations, axis=0)), index=index, columns=columns
    )
    returns = {}
    for strategy in build_strategies():
        returns[strategy.name] = backtest(prices, strategy.generate_weights(prices)).returns
    strategy_returns = pd.DataFrame(returns)
    weekly = strategy_returns.resample("W-FRI").apply(lambda values: (1 + values).prod() - 1)
    idata = fit_hierarchical_model(
        weekly, draws=20, tune=20, chains=2, seed=42, target_accept=0.9, periods_per_year=52
    )
    rankings, _ = posterior_rankings(idata)
    portfolio, allocations = walk_forward_score_portfolio(
        strategy_returns, "2021-01-01", lookback_days=126, top_n=3
    )
    assert len(rankings) == len(build_strategies())
    assert np.isclose(rankings["p_best"].sum(), 1)
    assert portfolio.notna().all()
    assert not allocations.empty
    print("Offline synthetic pipeline smoke test passed")


if __name__ == "__main__":
    main()
