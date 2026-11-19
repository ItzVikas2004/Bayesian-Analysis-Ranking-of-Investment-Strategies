"""Strictly chronological strategy-portfolio rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def static_portfolios(strategy_returns: pd.DataFrame, selection_end: str) -> dict[str, pd.Series]:
    available = strategy_returns.loc[:selection_end]
    sharpe = available.mean() / available.std() * np.sqrt(252)
    best = sharpe.idxmax()
    return {
        "Equal-Weighted Strategies": strategy_returns.mean(axis=1),
        f"Static Best ({best})": strategy_returns[best],
        "static_best_name": best,
    }


def walk_forward_score_portfolio(
    strategy_returns: pd.DataFrame, test_start: str, lookback_days=756, top_n=3
) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly empirical-Bayes shrinkage scores using only prior daily returns."""
    test = strategy_returns.loc[test_start:]
    months = test.index.to_period("M")
    weights = pd.DataFrame(0.0, index=test.index, columns=test.columns)
    records = []
    for period in months.unique():
        dates = test.index[months == period]
        cutoff = dates[0]
        hist = strategy_returns.loc[strategy_returns.index < cutoff].tail(lookback_days)
        raw = hist.mean() / hist.std() * np.sqrt(252)
        reliability = hist.count() / (hist.count() + 252)
        score = raw * reliability
        selected = score.nlargest(top_n).index
        w = pd.Series(0.0, index=test.columns)
        positive = score[selected].clip(lower=0)
        w[selected] = positive / positive.sum() if positive.sum() > 0 else 1 / len(selected)
        weights.loc[dates] = w.values
        records.append(
            {
                "rebalance_date": cutoff,
                "selected": ";".join(selected),
                **{f"weight_{k}": v for k, v in w.items() if v > 0},
            }
        )
    executed = weights.shift(1).fillna(0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    returns = (executed * test).sum(axis=1) - turnover * 0.0015
    return returns.rename("Bayesian Portfolio"), pd.DataFrame(records)
