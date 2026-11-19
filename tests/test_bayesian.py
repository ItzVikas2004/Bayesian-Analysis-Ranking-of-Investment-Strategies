import numpy as np
import pandas as pd
import pytest

from bayesian_investment.bayesian.model import fit_hierarchical_model
from bayesian_investment.bayesian.ranking import posterior_rankings


@pytest.mark.slow
def test_bayesian_shapes_and_probabilities():
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal([0.001, 0], [0.01, 0.012], (60, 2)), columns=["Buy-and-Hold", "Other"]
    )
    idata = fit_hierarchical_model(returns, draws=25, tune=25, chains=2, seed=1, target_accept=0.85)
    ranks, _ = posterior_rankings(idata)
    assert len(ranks) == 2
    assert ranks["p_best"].sum() == pytest.approx(1)
    assert ranks["p_top_3"].between(0, 1).all()
