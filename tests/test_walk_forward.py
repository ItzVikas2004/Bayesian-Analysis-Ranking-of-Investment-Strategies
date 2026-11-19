import numpy as np
import pandas as pd

from bayesian_investment.portfolio.construction import walk_forward_score_portfolio


def test_walk_forward_does_not_change_past_when_future_changes():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=800)
    r = pd.DataFrame(rng.normal(0, 0.01, (800, 4)), index=idx, columns=list("ABCD"))
    split = "2022-06-01"
    one, w1 = walk_forward_score_portfolio(r, split, 252, 2)
    changed = r.copy()
    changed.loc["2022-10-01":, "A"] += 0.1
    two, w2 = walk_forward_score_portfolio(changed, split, 252, 2)
    assert np.allclose(one.loc[:"2022-09-30"], two.loc[:"2022-09-30"], atol=1e-15)
    assert (w1.filter(like="weight_").fillna(0).sum(axis=1) <= 1.000001).all()
