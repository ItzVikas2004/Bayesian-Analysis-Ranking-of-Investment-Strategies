"""Heavy-tailed hierarchical strategy-return model."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def fit_hierarchical_model(
    returns: pd.DataFrame,
    draws=1000,
    tune=1000,
    chains=4,
    seed=42,
    target_accept=0.95,
    periods_per_year=252,
):
    """Fit Laplace period returns after scaling by 100 for stable weak priors."""
    long = returns.stack().dropna()
    strategy_names = list(returns.columns)
    codes = pd.Categorical(long.index.get_level_values(1), categories=strategy_names).codes
    observed = long.to_numpy() * 100
    with pm.Model(coords={"strategy": strategy_names, "observation": np.arange(len(observed))}):
        idx = pm.Data("strategy_idx", codes, dims="observation")
        mu_global = pm.Normal("mu_global", 0, 0.5)
        tau = pm.LogNormal("tau", mu=np.log(0.1), sigma=0.5)
        raw_offset = pm.Normal("raw_offset", 0, 1, dims="strategy")
        centered_offset = raw_offset - pt.mean(raw_offset)
        mu = pm.Deterministic("mu", mu_global + tau * centered_offset, dims="strategy")
        sigma = pm.HalfNormal("sigma", 2.0, dims="strategy")
        pm.Laplace(
            "returns",
            mu=mu[idx],
            b=sigma[idx] / np.sqrt(2),
            observed=observed,
            dims="observation",
        )
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 2),
            random_seed=seed,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=False,
            nuts_sampler="numpyro",
        )
        pm.sample_posterior_predictive(
            idata, random_seed=seed, extend_inferencedata=True, progressbar=False
        )
    idata.attrs["periods_per_year"] = periods_per_year
    return idata


def diagnostics(idata) -> dict:
    summary = az.summary(idata, var_names=["mu", "sigma", "tau", "mu_global"])
    divergences = int(idata.sample_stats["diverging"].sum())
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "worst_rhat_parameter": str(summary["r_hat"].idxmax()),
        "min_bulk_ess": float(summary["ess_bulk"].min()),
        "worst_bulk_ess_parameter": str(summary["ess_bulk"].idxmin()),
        "min_tail_ess": float(summary["ess_tail"].min()),
        "divergences": divergences,
        "reliable": bool(
            summary["r_hat"].max() <= 1.01 and divergences == 0 and summary["ess_bulk"].min() >= 100
        ),
    }
