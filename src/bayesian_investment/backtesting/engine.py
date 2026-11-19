"""Next-bar close-to-close vectorized backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


def backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000,
    fee_bps: float = 10,
    slippage_bps: float = 5,
    max_gross_exposure: float = 1.0,
) -> BacktestResult:
    """Apply signals one bar later; costs are charged on absolute weight changes."""
    weights = (
        target_weights.reindex_like(prices).fillna(0).clip(-max_gross_exposure, max_gross_exposure)
    )
    gross = weights.abs().sum(axis=1)
    scale = (max_gross_exposure / gross).clip(upper=1).fillna(1)
    weights = weights.mul(scale, axis=0)
    executed = weights.shift(1).fillna(0)
    asset_returns = prices.pct_change(fill_method=None).fillna(0)
    gross_returns = (executed * asset_returns).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    costs = turnover * (fee_bps + slippage_bps) / 10_000
    net = gross_returns - costs
    equity = initial_capital * (1 + net).cumprod()
    return BacktestResult(
        net.rename("return"), gross_returns, equity.rename("equity"), executed, turnover, costs
    )
