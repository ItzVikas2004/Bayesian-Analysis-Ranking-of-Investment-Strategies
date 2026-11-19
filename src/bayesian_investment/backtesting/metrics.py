"""Consistent performance and tail-risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min()) if len(wealth) else np.nan


def performance_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {
            k: np.nan
            for k in [
                "total_return",
                "cagr",
                "annual_volatility",
                "sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
                "var_95",
                "cvar_95",
            ]
        }
    years = len(r) / TRADING_DAYS
    total = float((1 + r).prod() - 1)
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    annual_mean = float(r.mean() * TRADING_DAYS)
    sharpe = annual_mean / vol if vol > 0 else np.nan
    downside = float(r[r < 0].std(ddof=1) * np.sqrt(TRADING_DAYS))
    sortino = annual_mean / downside if downside > 0 else np.nan
    mdd = maximum_drawdown(r)
    q = float(r.quantile(0.05))
    var = max(0.0, -q)
    tail = r[r <= q]
    cvar = max(0.0, -float(tail.mean())) if len(tail) else np.nan
    wins = r[r > 0]
    losses = r[r < 0]
    profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan
    active = r != 0
    return {
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "var_95": var,
        "cvar_95": cvar,
        "win_rate": float((r > 0).mean()),
        "profit_factor": profit_factor,
        "annual_turnover": float(turnover.mean() * TRADING_DAYS)
        if turnover is not None
        else np.nan,
        "number_of_trades": int((turnover > 1e-10).sum()) if turnover is not None else 0,
        "average_active_return": float(r[active].mean()) if active.any() else np.nan,
        "trading_exposure": float(active.mean()),
    }
