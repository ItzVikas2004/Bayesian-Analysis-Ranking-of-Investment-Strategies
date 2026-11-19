import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=500)
    returns = rng.normal(0.0003, 0.01, (500, 5))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=index,
        columns=["SPY", "QQQ", "KO", "PEP", "TLT"],
    )
