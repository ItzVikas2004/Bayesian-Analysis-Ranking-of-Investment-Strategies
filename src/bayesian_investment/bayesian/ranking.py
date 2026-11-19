"""Posterior risk-adjusted leaderboard calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def posterior_rankings(idata, benchmark: str = "Buy-and-Hold") -> tuple[pd.DataFrame, pd.DataFrame]:
    mu = (
        idata.posterior["mu"].stack(sample=("chain", "draw")).transpose("sample", "strategy").values
        / 100
    )
    sigma = (
        idata.posterior["sigma"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "strategy")
        .values
        / 100
    )
    names = idata.posterior.coords["strategy"].values.astype(str)
    periods_per_year = idata.attrs.get("periods_per_year", 252)
    sharpe = mu / sigma * np.sqrt(periods_per_year)
    order = np.argsort(-sharpe, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(1, len(names) + 1)
    bench_idx = list(names).index(benchmark) if benchmark in names else 0
    rows = []
    for i, name in enumerate(names):
        lo, hi = np.quantile(sharpe[:, i], [0.025, 0.975])
        rows.append(
            {
                "strategy": name,
                "posterior_mean_period_return": mu[:, i].mean(),
                "posterior_sharpe": sharpe[:, i].mean(),
                "sharpe_ci_lower": lo,
                "sharpe_ci_upper": hi,
                "p_sharpe_gt_zero": (sharpe[:, i] > 0).mean(),
                "p_beats_benchmark": (sharpe[:, i] > sharpe[:, bench_idx]).mean()
                if i != bench_idx
                else 0.5,
                "p_best": (ranks[:, i] == 1).mean(),
                "expected_rank": ranks[:, i].mean(),
                "p_top_3": (ranks[:, i] <= 3).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["expected_rank", "strategy"]), pd.DataFrame(
        sharpe, columns=names
    )
