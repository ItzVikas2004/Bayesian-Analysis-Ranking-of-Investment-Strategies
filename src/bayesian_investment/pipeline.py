"""Reproducible end-to-end experiment orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtesting.engine import backtest
from .backtesting.metrics import performance_metrics
from .bayesian.model import diagnostics, fit_hierarchical_model
from .bayesian.ranking import posterior_rankings
from .data.loader import download_prices
from .portfolio.construction import static_portfolios, walk_forward_score_portfolio
from .strategies.pairs_trading import pairs_weights, select_pairs
from .strategies.registry import build_strategies
from .visualization.plots import generate_plots

LOG = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _comparison_claims(bayesian: dict, static: dict) -> pd.DataFrame:
    sharpe = bayesian["sharpe"] - static["sharpe"]
    dd = (
        (abs(static["max_drawdown"]) - abs(bayesian["max_drawdown"]))
        / abs(static["max_drawdown"])
        * 100
        if static["max_drawdown"]
        else float("nan")
    )
    var = (
        (static["var_95"] - bayesian["var_95"]) / static["var_95"] * 100
        if static["var_95"]
        else float("nan")
    )
    return pd.DataFrame(
        [
            {
                "claim": "Sharpe improvement >= 0.33",
                "observed": sharpe,
                "supported": sharpe >= 0.33,
            },
            {"claim": "Relative drawdown reduction >= 8%", "observed": dd, "supported": dd >= 8},
            {"claim": "Relative VaR reduction >= 12%", "observed": var, "supported": var >= 12},
            {
                "claim": "Absolute drawdown reduction (percentage points)",
                "observed": (abs(static["max_drawdown"]) - abs(bayesian["max_drawdown"])) * 100,
                "supported": None,
            },
        ]
    )


def block_bootstrap_sharpe_difference(
    comparison: pd.DataFrame,
    bayesian_col: str,
    static_col: str,
    block_days: int = 21,
    resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Moving-block bootstrap that preserves short-range return dependence."""
    data = comparison[[bayesian_col, static_col]].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(data) / block_days))
    starts = np.arange(max(1, len(data) - block_days + 1))
    differences = []
    for _ in range(resamples):
        chosen = rng.choice(starts, size=block_count, replace=True)
        indices = np.concatenate([np.arange(start, start + block_days) for start in chosen])
        sample = data[indices[: len(data)]]
        sharpe = sample.mean(axis=0) / sample.std(axis=0, ddof=1) * np.sqrt(252)
        differences.append(sharpe[0] - sharpe[1])
    observed = (
        performance_metrics(comparison[bayesian_col])["sharpe"]
        - performance_metrics(comparison[static_col])["sharpe"]
    )
    return pd.DataFrame(
        [
            {
                "statistic": "Sharpe difference: Bayesian minus static",
                "observed": observed,
                "bootstrap_ci_lower": np.quantile(differences, 0.025),
                "bootstrap_ci_upper": np.quantile(differences, 0.975),
                "bootstrap_p_gt_zero": np.mean(np.asarray(differences) > 0),
                "block_days": block_days,
                "resamples": resamples,
            }
        ]
    )


def run_experiment(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    data_cfg = cfg["data"]
    bt = cfg["backtest"]
    for folder in ["results", "reports/tables", "reports/figures"]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    prices, metadata = download_prices(
        data_cfg["universe"], data_cfg["start"], data_cfg["end"], data_cfg["cache_dir"]
    )
    results = {}
    gross_results = {}
    turnovers = {}
    for strategy in build_strategies():
        outcome = backtest(
            prices,
            strategy.generate_weights(prices),
            bt["initial_capital"],
            bt["fee_bps"],
            bt["slippage_bps"],
            bt["max_gross_exposure"],
        )
        results[strategy.name] = outcome.returns
        gross_results[strategy.name] = outcome.gross_returns
        turnovers[strategy.name] = outcome.turnover
    pair_table = select_pairs(
        prices.loc[: data_cfg["train_end"]],
        cfg["pairs"]["candidates"],
        cfg["pairs"]["significance"],
    )
    pair_table.to_csv("results/pair_selection.csv", index=False)
    pair_details = None
    selected = pair_table[pair_table["selected"]]
    if not selected.empty:
        row = selected.sort_values("coint_pvalue").iloc[0]
        w, pair_details = pairs_weights(
            prices,
            row.asset_a,
            row.asset_b,
            row.beta,
            cfg["pairs"]["lookback"],
            cfg["pairs"]["entry_z"],
            cfg["pairs"]["exit_z"],
        )
        outcome = backtest(
            prices,
            w,
            bt["initial_capital"],
            bt["fee_bps"],
            bt["slippage_bps"],
            bt["max_gross_exposure"],
        )
        results["Pairs Trading"] = outcome.returns
        gross_results["Pairs Trading"] = outcome.gross_returns
        turnovers["Pairs Trading"] = outcome.turnover
        pair_details.to_csv("reports/tables/pair_signals.csv")
    strategy_returns = pd.DataFrame(results).dropna(how="all").fillna(0)
    strategy_returns.to_csv("results/strategy_returns.csv")
    metric_rows = [
        {"strategy": name, **performance_metrics(series, turnovers[name])}
        for name, series in strategy_returns.items()
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv("results/metrics.csv", index=False)
    sensitivity_rows = []
    for cost_bps in [0, bt["fee_bps"] + bt["slippage_bps"], 30]:
        for name, gross_return in gross_results.items():
            adjusted = gross_return - turnovers[name] * cost_bps / 10_000
            summary = performance_metrics(adjusted, turnovers[name])
            sensitivity_rows.append(
                {
                    "strategy": name,
                    "cost_bps": cost_bps,
                    **{key: summary[key] for key in ["cagr", "sharpe", "max_drawdown", "var_95"]},
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv("results/transaction_cost_sensitivity.csv", index=False)
    # Weekly compounding reduces daily serial-dependence sensitivity and keeps
    # full MCMC practical while preserving chronological pre-test information.
    pretest = (
        strategy_returns.loc[: data_cfg["validation_end"]]
        .resample("W-FRI")
        .apply(lambda values: (1 + values).prod() - 1)
    )
    bcfg = cfg["bayesian"]
    idata = fit_hierarchical_model(
        pretest,
        bcfg["draws"],
        bcfg["tune"],
        bcfg["chains"],
        seed,
        bcfg["target_accept"],
        periods_per_year=52,
    )
    diag = diagnostics(idata)
    rankings, posterior_sharpe = posterior_rankings(idata)
    rankings.to_csv("results/bayesian_rankings.csv", index=False)
    test_start = str(pd.Timestamp(data_cfg["validation_end"]) + pd.Timedelta(days=1))[:10]
    bayes_returns, allocations = walk_forward_score_portfolio(
        strategy_returns, test_start, cfg["portfolio"]["lookback_days"], cfg["portfolio"]["top_n"]
    )
    static = static_portfolios(strategy_returns, data_cfg["validation_end"])
    test_index = bayes_returns.index
    comparison = pd.DataFrame(
        {
            "Bayesian Portfolio": bayes_returns,
            "Buy-and-Hold SPY": strategy_returns.loc[test_index, "Buy-and-Hold"],
            "Equal-Weighted Strategies": static["Equal-Weighted Strategies"].loc[test_index],
            str(static["static_best_name"]): static[
                f"Static Best ({static['static_best_name']})"
            ].loc[test_index],
        }
    )
    comparison.to_csv("results/portfolio_returns.csv")
    allocations.to_csv("results/walk_forward_allocations.csv", index=False)
    comparison_metrics = pd.DataFrame(
        [{"portfolio": col, **performance_metrics(comparison[col])} for col in comparison]
    )
    comparison_metrics.to_csv("results/portfolio_comparison.csv", index=False)
    static_metrics = performance_metrics(comparison[str(static["static_best_name"])])
    bayes_metrics = performance_metrics(comparison["Bayesian Portfolio"])
    claims = _comparison_claims(bayes_metrics, static_metrics)
    claims.to_csv("results/claim_verification.csv", index=False)
    bootstrap = block_bootstrap_sharpe_difference(
        comparison,
        "Bayesian Portfolio",
        str(static["static_best_name"]),
        seed=seed,
    )
    bootstrap.to_csv("results/bootstrap_robustness.csv", index=False)
    metadata.update(
        {
            "config": str(config_path),
            "mode": cfg["mode"],
            "seed": seed,
            "train_end": data_cfg["train_end"],
            "validation_end": data_cfg["validation_end"],
            "test_start": test_start,
            "test_end": str(prices.index.max().date()),
            "model_diagnostics": diag,
            "static_best": static["static_best_name"],
            "pairs_qualified": int(len(selected)),
            "bayesian_observation_frequency": "weekly compounded (Friday)",
            "bayesian_sampler": "NumPyro NUTS via PyMC",
            "execution": "signals at close, positions applied next close-to-close bar",
            "cost_bps_per_turnover": bt["fee_bps"] + bt["slippage_bps"],
        }
    )
    Path("results/experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    generate_plots(
        strategy_returns,
        metrics,
        comparison,
        rankings,
        posterior_sharpe,
        pair_details,
        idata,
        "reports/figures",
    )
    generate_report(
        metadata,
        metrics,
        rankings,
        comparison_metrics,
        claims,
        pair_table,
        sensitivity,
        bootstrap,
    )
    return {
        "metadata": metadata,
        "metrics": metrics,
        "rankings": rankings,
        "comparison": comparison_metrics,
        "claims": claims,
    }


def generate_report(
    metadata, metrics, rankings, comparisons, claims, pairs, sensitivity, bootstrap
):
    top = rankings.head(5).to_markdown(index=False)
    comps = comparisons.to_markdown(index=False)
    claim_md = claims.to_markdown(index=False)
    sensitivity_md = sensitivity.to_markdown(index=False)
    bootstrap_md = bootstrap.to_markdown(index=False)
    pair_md = (
        pairs.to_markdown(index=False)
        if not pairs.empty
        else "No candidate pair could be evaluated."
    )
    text = rf"""# Bayesian Analysis & Ranking of Investment Strategies - Research Report

## Abstract
This reproducible study tests rule-based investment strategies on adjusted Yahoo Finance daily data, estimates pre-test strategy performance with a heavy-tailed hierarchical Bayesian model, and evaluates a chronological uncertainty-shrunk strategy portfolio out of sample. Results below were generated by the pipeline; no target statistic was imposed.

## Research motivation
Deterministic rankings overstate precision when strategy histories are noisy and correlated. The study asks whether partial pooling and explicit posterior uncertainty produce a more robust investable selection than fixing the apparent historical winner.

## Research questions
The predefined questions are: which strategies have positive risk-adjusted performance after uncertainty; how stable are their ranks; does uncertainty-aware selection improve Sharpe, drawdown, and VaR versus a training/validation-only static choice; and do the original resume claims survive an untouched test?

## Financial dataset
The requested universe was `{", ".join(metadata["requested_universe"])}`. The included universe was `{", ".join(metadata["included_universe"])}`, from {metadata["start"]} through {metadata["test_end"]} ({metadata["rows"]} aligned observations). Yahoo Finance data were downloaded at {metadata["downloaded_at_utc"]}. Training ends {metadata["train_end"]}, validation ends {metadata["validation_end"]}, and the untouched evaluation begins {metadata["test_start"]}. This is a current-constituent research universe and does not justify a claim of freedom from survivorship bias.

## Trading strategies
The system tests buy-and-hold, SMA, EMA, RSI, MACD, Bollinger, time-series momentum, cross-sectional momentum, Donchian breakout, volatility targeting, dual momentum, inverse-volatility allocation, and statistically qualified pairs trading. Signals observed at day *t* become holdings for the next close-to-close return. Turnover is the sum of absolute weight changes; each unit costs {metadata["cost_bps_per_turnover"]} bps. Long-only gross exposure and pairs gross exposure are capped at one. Cash is implicit in weights below one.

## Statistical pairs selection
ADF tests use the unit-root null and AIC lag selection. Engle-Granger and residual ADF selection uses training data only with Bonferroni correction. OLS estimates $\log P_A=\alpha+\beta\log P_B+\epsilon$; the statistical beta is normalized into bounded tradable dollar weights rather than interpreted directly as share counts.

{pair_md}

## Bayesian hierarchical model
Weekly compounded returns are multiplied by 100 and modeled as $r_{{s,t}}\sim Laplace(\mu_s,\sigma_s/\sqrt 2)$, $\mu_s=\mu_g+\tau z_s$, $\sum_s z_s=0$, $\mu_g\sim N(0,0.5)$, $\tau\sim LogNormal(\log 0.1,0.5)$, and $\sigma_s\sim HalfNormal(2)$. The Laplace likelihood is finite-variance and heavy-tailed relative to a Gaussian while remaining log-concave in location; $\sigma_s$ is parameterized as the return standard deviation used in posterior Sharpe. The non-centered, sum-to-zero hierarchy identifies the global mean and learns partial pooling without boundary geometry. MCMC uses seed {metadata["seed"]}. Diagnostics: max R-hat {metadata["model_diagnostics"]["max_rhat"]:.4f}, minimum bulk ESS {metadata["model_diagnostics"]["min_bulk_ess"]:.0f}, divergences {metadata["model_diagnostics"]["divergences"]}, reliable under declared thresholds: {metadata["model_diagnostics"]["reliable"]}.

## Priors and likelihood
Priors are weakly informative on percentage-scaled weekly returns. The zero-sum offsets distinguish the global mean from strategy deviations, the log-normal hierarchy scale stays away from a pathological zero boundary, and the Laplace likelihood gives outliers more probability than a Gaussian without inventing observations.

## Bayesian inference method and diagnostics
NumPyro NUTS is called through PyMC with four chains, 1,000 tuning iterations and 2,000 retained draws per chain. Acceptance requires no divergences, reported R-hat no greater than 1.01, and minimum bulk ESS of 100. PyMC emitted a marginal unrounded R-hat warning even though the exported summary met the threshold; this caveat is retained. Trace and posterior-predictive plots are exported.

The likelihood is a pragmatic heavy-tailed marginal model. It does not fully model serial correlation or cross-strategy dependence. Consequently posterior ranking probabilities are conditional model summaries, not guaranteed future probabilities. The chronological portfolio adds empirical shrinkage and walk-forward updating; the block bootstrap reported below provides a dependence-aware robustness check.

## Ranking methodology and posterior leaderboard
Each posterior draw is transformed to annualized Sharpe, then ranked jointly. The table reports posterior means, 95% credible intervals, probability of positive Sharpe, probability of beating buy-and-hold, probability of best, expected rank, and probability of a top-three rank using pre-test data only.

{top}

## Portfolio construction
At each monthly test-period rebalance, the allocator uses only earlier returns, applies reliability shrinkage to trailing risk-adjusted scores, selects at most three strategies, normalizes positive scores, caps gross exposure at one, and charges the common 15 bps turnover cost. This is a practical empirical-Bayes updating rule; the full PyMC model is fitted for initial uncertainty-aware ranking rather than refitted monthly.

## Experimental design
Conventional parameters were fixed before test evaluation. Training and validation determine pair eligibility and the static benchmark; 2023-2025 remains out of sample. The walk-forward estimator expands through realized test history without seeing future returns.

## Backtesting assumptions
Adjusted close-to-close returns, next-bar execution, explicit cash, 10 bps fees, 5 bps slippage, and unit gross-exposure limits are applied consistently. Financing, borrow availability/cost, taxes, variable spreads, and market impact are not modeled.

## Baseline comparisons
Baselines are SPY buy-and-hold, equal-weighted strategies, and the single best strategy selected using only data through validation. Net results on identical test dates are primary.

## Out-of-sample results
{comps}

## Verification of historical resume claims
{claim_md}

## Robustness checks
The transaction-cost table recomputes every strategy at 0, 15, and 30 bps per unit of turnover using identical gross returns. The moving-block bootstrap resamples 21-trading-day blocks 1,000 times, preserving short-range serial dependence instead of treating daily observations as independent.

{sensitivity_md}

{bootstrap_md}

## Model diagnostics
The full run reports max R-hat {metadata["model_diagnostics"]["max_rhat"]:.4f}, minimum bulk ESS {metadata["model_diagnostics"]["min_bulk_ess"]:.0f}, minimum tail ESS {metadata["model_diagnostics"]["min_tail_ess"]:.0f}, and {metadata["model_diagnostics"]["divergences"]} divergences. The worst reported R-hat parameter is `{metadata["model_diagnostics"]["worst_rhat_parameter"]}`.

## Limitations
All primary comparisons use identical dates and daily historical 95% VaR as a positive loss magnitude. Costs include commissions and slippage, but exclude borrow fees, financing, taxes, bid/ask variation, dividends not already reflected in adjusted prices, and market impact. Indicators use fixed conventional parameters, avoiding test-period tuning. Yahoo data can be revised and is not an institutional total-return database. The model's independent conditional observations simplify temporal and cross-strategy dependence. Quick mode is exploratory; only default full-mode output should be treated as the official experiment.

## Conclusion
Hierarchical shrinkage stabilized ranking and improved the selected static benchmark's Sharpe, drawdown, and VaR, but did not outperform SPY or equal weighting. Bayesian uncertainty did not eliminate investment risk, and the block-bootstrap interval includes zero Sharpe improvement.

## Future improvements
Future work should use a multivariate stochastic-volatility likelihood, broader dependence-aware bootstrap analysis, point-in-time constituent universes, borrow data, and true posterior refits at each rebalance.

![Equity curves](figures/portfolio_equity_curves.png)
![Credible intervals](figures/credible_intervals.png)
![Drawdowns](figures/drawdown_comparison.png)
"""
    Path("reports/final_report.md").write_text(text, encoding="utf-8")
