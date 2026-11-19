"""Training-selected, rolling z-score statistical arbitrage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_investment.statistics.tests import adf_test, cointegration_test, ols_hedge_ratio


def select_pairs(prices: pd.DataFrame, candidates: list[list[str]], significance: float = 0.05):
    """Select with Bonferroni-adjusted Engle-Granger tests on supplied training data."""
    rows = []
    alpha = significance / max(1, len(candidates))
    for a, b in candidates:
        if a not in prices or b not in prices:
            continue
        coint_result = cointegration_test(prices[a], prices[b])
        hedge = ols_hedge_ratio(prices[a], prices[b])
        adf = adf_test(hedge["residual"])
        rows.append(
            {
                "asset_a": a,
                "asset_b": b,
                "correlation": prices[[a, b]].pct_change().corr().iloc[0, 1],
                "coint_statistic": coint_result["statistic"],
                "coint_pvalue": coint_result["pvalue"],
                "adf_statistic": adf["statistic"],
                "adf_pvalue": adf["pvalue"],
                "alpha": hedge["alpha"],
                "beta": hedge["beta"],
                "selected": coint_result["pvalue"] < alpha and adf["pvalue"] < alpha,
            }
        )
    return pd.DataFrame(rows)


def pairs_weights(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    beta: float,
    lookback: int = 126,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dollar-normalized long/short weights based on a fixed training hedge ratio."""
    spread = np.log(prices[asset_a]) - beta * np.log(prices[asset_b])
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    z = (spread - mean) / std
    state = pd.Series(np.nan, index=prices.index)
    state[z > entry_z] = -1
    state[z < -entry_z] = 1
    state[z.abs() < exit_z] = 0
    state = state.ffill().fillna(0)
    denom = 1 + abs(beta)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[asset_a] = state / denom
    weights[asset_b] = -state * beta / denom
    return weights, pd.DataFrame({"spread": spread, "zscore": z, "state": state})
