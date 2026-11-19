import pandas as pd
import pytest

from bayesian_investment.backtesting.metrics import maximum_drawdown, performance_metrics


def test_drawdown_and_var_definition():
    r = pd.Series([0.1, -0.2, 0.05])
    assert maximum_drawdown(r) == pytest.approx(-0.2)
    m = performance_metrics(r)
    assert m["var_95"] >= 0
    assert m["cvar_95"] >= m["var_95"]


def test_zero_volatility_is_undefined():
    assert pd.isna(performance_metrics(pd.Series([0.0, 0.0, 0.0]))["sharpe"])
