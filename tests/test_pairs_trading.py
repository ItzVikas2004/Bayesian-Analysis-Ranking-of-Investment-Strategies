import numpy as np
import pandas as pd
import pytest

from bayesian_investment.statistics.tests import adf_test, ols_hedge_ratio
from bayesian_investment.strategies.pairs_trading import pairs_weights


def test_ols_recovers_log_hedge_ratio():
    x = pd.Series(np.exp(np.linspace(4, 5, 300)))
    y = np.exp(0.2 + 1.5 * np.log(x))
    result = ols_hedge_ratio(y, x)
    assert result["beta"] == pytest.approx(1.5, rel=1e-5)
    assert result["residual"].abs().max() < 1e-8


def test_adf_stationary_series():
    rng = np.random.default_rng(1)
    x = np.zeros(1000)
    for i in range(1, len(x)):
        x[i] = 0.5 * x[i - 1] + rng.normal()
    assert adf_test(pd.Series(x))["pvalue"] < 0.05


def test_pairs_weights_gross_bound(prices):
    weights, details = pairs_weights(prices, "KO", "PEP", 1.2, 20, 1, 0.2)
    assert (weights.abs().sum(axis=1) <= 1.000001).all()
    assert set(details) == {"spread", "zscore", "state"}
