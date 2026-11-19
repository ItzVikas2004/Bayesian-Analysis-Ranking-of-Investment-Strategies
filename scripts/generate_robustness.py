"""Regenerate lightweight robustness tables from an existing full experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesian_investment.backtesting.engine import backtest
from bayesian_investment.backtesting.metrics import performance_metrics
from bayesian_investment.data.loader import download_prices
from bayesian_investment.pipeline import (
    block_bootstrap_sharpe_difference,
    generate_report,
    load_config,
)
from bayesian_investment.strategies.registry import build_strategies


def main() -> None:
    cfg = load_config("config/default.yaml")
    data_cfg = cfg["data"]
    bt = cfg["backtest"]
    prices, _ = download_prices(
        data_cfg["universe"], data_cfg["start"], data_cfg["end"], data_cfg["cache_dir"]
    )
    sensitivity_rows = []
    for strategy in build_strategies():
        outcome = backtest(
            prices,
            strategy.generate_weights(prices),
            bt["initial_capital"],
            bt["fee_bps"],
            bt["slippage_bps"],
            bt["max_gross_exposure"],
        )
        for cost_bps in [0, bt["fee_bps"] + bt["slippage_bps"], 30]:
            adjusted = outcome.gross_returns - outcome.turnover * cost_bps / 10_000
            summary = performance_metrics(adjusted, outcome.turnover)
            sensitivity_rows.append(
                {
                    "strategy": strategy.name,
                    "cost_bps": cost_bps,
                    **{key: summary[key] for key in ["cagr", "sharpe", "max_drawdown", "var_95"]},
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv("results/transaction_cost_sensitivity.csv", index=False)

    metadata = json.loads(Path("results/experiment_metadata.json").read_text(encoding="utf-8"))
    comparison_returns = pd.read_csv("results/portfolio_returns.csv", index_col=0, parse_dates=True)
    bootstrap = block_bootstrap_sharpe_difference(
        comparison_returns,
        "Bayesian Portfolio",
        metadata["static_best"],
        seed=metadata["seed"],
    )
    bootstrap.to_csv("results/bootstrap_robustness.csv", index=False)
    generate_report(
        metadata,
        pd.read_csv("results/metrics.csv"),
        pd.read_csv("results/bayesian_rankings.csv"),
        pd.read_csv("results/portfolio_comparison.csv"),
        pd.read_csv("results/claim_verification.csv"),
        pd.read_csv("results/pair_selection.csv"),
        sensitivity,
        bootstrap,
    )


if __name__ == "__main__":
    main()
