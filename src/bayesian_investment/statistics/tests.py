"""Auditable ADF, Engle-Granger, and OLS wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def adf_test(series: pd.Series) -> dict:
    """Test H0: unit root against H1: stationarity, with AIC lag selection."""
    x = series.dropna().astype(float)
    statistic, pvalue, lags, nobs, critical, _ = adfuller(x, autolag="AIC", result_object=False)
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "lags": lags,
        "nobs": nobs,
        **{f"critical_{k}": v for k, v in critical.items()},
    }


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> dict:
    data = pd.concat([y, x], axis=1).dropna()
    ly = np.log(data.iloc[:, 0])
    lx = np.log(data.iloc[:, 1])
    fit = sm.OLS(ly, sm.add_constant(lx)).fit()
    residual = ly - fit.predict(sm.add_constant(lx))
    return {
        "alpha": float(fit.params.iloc[0]),
        "beta": float(fit.params.iloc[1]),
        "r_squared": float(fit.rsquared),
        "residual": residual,
    }


def cointegration_test(y: pd.Series, x: pd.Series) -> dict:
    data = pd.concat([y, x], axis=1).dropna()
    statistic, pvalue, critical = coint(
        np.log(data.iloc[:, 0]), np.log(data.iloc[:, 1]), trend="c", autolag="aic"
    )
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "critical_1pct": critical[0],
        "critical_5pct": critical[1],
        "critical_10pct": critical[2],
    }
