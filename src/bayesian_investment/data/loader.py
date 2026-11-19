"""Yahoo Finance loader with retry, validation, and local caching."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

LOG = logging.getLogger(__name__)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and align adjusted closing prices without fabricating observations."""
    if prices.empty:
        raise ValueError("No market observations were downloaded")
    if prices.index.has_duplicates:
        raise ValueError("Duplicate market timestamps detected")
    prices = prices.sort_index().astype(float)
    if (prices <= 0).any().any():
        raise ValueError("Non-positive adjusted prices detected")
    usable = prices.columns[prices.notna().sum() >= max(30, int(len(prices) * 0.8))]
    if len(usable) == 0:
        raise ValueError("No asset has sufficient price history")
    dropped = sorted(set(prices.columns) - set(usable))
    if dropped:
        LOG.warning("Excluding assets with insufficient history: %s", dropped)
    return prices.loc[:, usable].ffill(limit=3).dropna()


def download_prices(
    tickers: list[str], start: str, end: str, cache_dir: str | Path, retries: int = 3
) -> tuple[pd.DataFrame, dict]:
    """Download auto-adjusted daily close prices, using a parameter-specific cache."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    yf_cache = cache / "yfinance"
    yf_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yf_cache))
    key = f"prices_{start}_{end}_{'-'.join(tickers)}".replace("/", "-")
    path = cache / f"{key}.csv"
    meta_path = cache / f"{key}.json"
    if path.exists() and meta_path.exists():
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        prices = validate_prices(prices)
        if "SPY" not in tickers or "SPY" in prices.columns:
            return prices, json.loads(meta_path.read_text(encoding="utf-8"))
        LOG.warning("Ignoring cached dataset because required SPY benchmark is absent")
    last_error: Exception | None = None
    provider_end = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=provider_end,
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if len(tickers) == 1:
                close.columns = tickers
            prices = validate_prices(close)
            if "SPY" in tickers and "SPY" not in prices.columns:
                raise ValueError("Required SPY benchmark was not downloaded")
            metadata = {
                "provider": "Yahoo Finance via yfinance",
                "downloaded_at_utc": datetime.now(UTC).isoformat(),
                "requested_universe": tickers,
                "included_universe": prices.columns.tolist(),
                "start": start,
                "requested_end_inclusive": end,
                "provider_end_exclusive": provider_end,
                "rows": len(prices),
            }
            prices.to_csv(path)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return prices, metadata
        except Exception as exc:  # provider/network errors vary
            last_error = exc
            LOG.warning("Download attempt %s/%s failed: %s", attempt + 1, retries, exc)
            time.sleep(2**attempt)
    raise RuntimeError(
        "Yahoo Finance download failed; synthetic data were not substituted"
    ) from last_error


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple close-to-close returns."""
    return prices.pct_change(fill_method=None).replace([float("inf"), -float("inf")], pd.NA)
