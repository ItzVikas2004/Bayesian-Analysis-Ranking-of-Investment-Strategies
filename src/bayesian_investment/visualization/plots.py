"""Plot generation kept separate from research calculations."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def _save(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def generate_plots(
    strategy_returns,
    metrics,
    comparison_returns,
    rankings,
    posterior_sharpe,
    pair_details,
    idata,
    out_dir,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (1 + strategy_returns).cumprod().plot(figsize=(12, 7), title="Strategy cumulative growth (net)")
    _save(out / "strategy_cumulative_returns.png")
    (1 + comparison_returns).cumprod().plot(
        figsize=(11, 6), title="Out-of-sample portfolio comparison"
    )
    _save(out / "portfolio_equity_curves.png")
    wealth = (1 + comparison_returns).cumprod()
    (wealth / wealth.cummax() - 1).plot(figsize=(11, 6), title="Out-of-sample drawdowns")
    _save(out / "drawdown_comparison.png")
    metrics.set_index("strategy")["sharpe"].sort_values().plot.barh(
        figsize=(9, 7), title="Annualized Sharpe ratios"
    )
    _save(out / "sharpe_comparison.png")
    ax = metrics.plot.scatter(
        "annual_volatility", "cagr", figsize=(9, 6), title="Return versus volatility"
    )
    for _, row in metrics.iterrows():
        ax.annotate(row["strategy"], (row["annual_volatility"], row["cagr"]), fontsize=7)
    _save(out / "return_volatility_scatter.png")
    posterior_sharpe.plot.kde(figsize=(11, 7), title="Posterior Sharpe distributions")
    _save(out / "posterior_sharpe_distributions.png")
    rankings.set_index("strategy")["p_best"].sort_values().plot.barh(
        figsize=(9, 7), title="Posterior probability of best strategy"
    )
    _save(out / "ranking_probabilities.png")
    r = rankings.sort_values("posterior_sharpe")
    plt.figure(figsize=(9, 7))
    plt.errorbar(
        r["posterior_sharpe"],
        range(len(r)),
        xerr=[
            r["posterior_sharpe"] - r["sharpe_ci_lower"],
            r["sharpe_ci_upper"] - r["posterior_sharpe"],
        ],
        fmt="o",
    )
    plt.yticks(range(len(r)), r["strategy"])
    plt.axvline(0, color="black", lw=1)
    plt.title("Posterior Sharpe means and 95% credible intervals")
    _save(out / "credible_intervals.png")
    comparison_returns.apply(lambda x: max(0, -x.quantile(0.05))).plot.bar(
        figsize=(8, 5), title="Historical daily 95% VaR (loss magnitude)"
    )
    _save(out / "var_comparison.png")
    monthly = comparison_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    heat = monthly.iloc[:, 0].to_frame("return")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    pivot = heat.pivot(index="year", columns="month", values="return")
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0)
    plt.title(f"Monthly returns: {comparison_returns.columns[0]}")
    _save(out / "monthly_returns_heatmap.png")
    sns.heatmap(strategy_returns.corr(), cmap="vlag", center=0)
    plt.title("Strategy return correlation")
    _save(out / "strategy_correlation_heatmap.png")
    if pair_details is not None:
        pair_details["spread"].plot(figsize=(10, 5), title="Selected pair log-price spread")
        _save(out / "pairs_spread.png")
        pair_details["zscore"].plot(figsize=(10, 5), title="Selected pair rolling z-score")
        plt.axhline(2, color="r", ls="--")
        plt.axhline(-2, color="g", ls="--")
        _save(out / "pairs_zscore.png")
        pair_details["state"].plot(figsize=(10, 4), title="Pairs entry/exit state")
        _save(out / "pairs_signals.png")
    az.plot_trace(idata, var_names=["mu_global", "tau", "mu"], compact=True)
    _save(out / "pymc_trace.png")
    az.plot_ppc(idata, num_pp_samples=50)
    _save(out / "posterior_predictive_check.png")
