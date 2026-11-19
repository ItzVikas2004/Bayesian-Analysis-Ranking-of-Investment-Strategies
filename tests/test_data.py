import pandas as pd
import pytest

from bayesian_investment.data.loader import returns_from_prices, validate_prices


def test_returns_compound_correctly():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.date_range("2020-01-01", periods=3))
    r = returns_from_prices(p)
    assert r.iloc[1, 0] == pytest.approx(0.1)
    assert (1 + r.dropna()).prod().iloc[0] == pytest.approx(0.99)


def test_duplicate_dates_rejected(prices):
    bad = pd.concat([prices, prices.iloc[[-1]]])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prices(bad)
