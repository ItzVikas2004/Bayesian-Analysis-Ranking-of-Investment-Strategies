"""Operational single- and multi-asset systematic strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, normalize_long_only


def _primary(prices: pd.DataFrame) -> str:
    """Use SPY for single-asset rules, falling back only for generic test fixtures."""
    return "SPY" if "SPY" in prices.columns else str(prices.columns[0])


class BuyHold(Strategy):
    def __init__(self):
        super().__init__("Buy-and-Hold")

    def generate_weights(self, prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w[_primary(prices)] = 1.0
        return w


class SMACrossover(Strategy):
    def __init__(self, fast=20, slow=100):
        super().__init__("SMA Crossover", {"fast": fast, "slow": slow})

    def generate_weights(self, prices):
        asset = _primary(prices)
        s = prices[asset]
        x = (
            s.rolling(self.parameters["fast"]).mean() > s.rolling(self.parameters["slow"]).mean()
        ).astype(float)
        w = prices * 0.0
        w[asset] = x
        return w


class EMACrossover(Strategy):
    def __init__(self, fast=12, slow=26):
        super().__init__("EMA Crossover", {"fast": fast, "slow": slow})

    def generate_weights(self, prices):
        asset = _primary(prices)
        s = prices[asset]
        x = (
            s.ewm(span=self.parameters["fast"], adjust=False).mean()
            > s.ewm(span=self.parameters["slow"], adjust=False).mean()
        ).astype(float)
        w = prices * 0.0
        w[asset] = x
        return w


class RSIMeanReversion(Strategy):
    def __init__(self, window=14):
        super().__init__("RSI Mean Reversion", {"window": window})

    def generate_weights(self, prices):
        asset = _primary(prices)
        s = prices[asset]
        d = s.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / self.parameters["window"], adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / self.parameters["window"], adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        state = pd.Series(np.nan, index=s.index)
        state[rsi < 30] = 1
        state[rsi > 55] = 0
        w = prices * 0.0
        w[asset] = state.ffill().fillna(0)
        return w


class MACDTrend(Strategy):
    def __init__(self):
        super().__init__("MACD Trend Following", {"fast": 12, "slow": 26, "signal": 9})

    def generate_weights(self, prices):
        asset = _primary(prices)
        s = prices[asset]
        macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        x = (macd > macd.ewm(span=9, adjust=False).mean()).astype(float)
        w = prices * 0.0
        w[asset] = x
        return w


class BollingerMeanReversion(Strategy):
    def __init__(self, window=20, width=2.0):
        super().__init__("Bollinger Mean Reversion", {"window": window, "width": width})

    def generate_weights(self, prices):
        asset = _primary(prices)
        s = prices[asset]
        m = s.rolling(self.parameters["window"]).mean()
        sd = s.rolling(self.parameters["window"]).std()
        state = pd.Series(np.nan, index=s.index)
        state[s < m - self.parameters["width"] * sd] = 1
        state[s >= m] = 0
        w = prices * 0.0
        w[asset] = state.ffill().fillna(0)
        return w


class TimeSeriesMomentum(Strategy):
    def __init__(self, lookback=252):
        super().__init__("Time-Series Momentum", {"lookback": lookback})

    def generate_weights(self, prices):
        return normalize_long_only(
            (prices.pct_change(self.parameters["lookback"]) > 0).astype(float)
        )


class CrossSectionalMomentum(Strategy):
    def __init__(self, lookback=252, top=3):
        super().__init__("Cross-Sectional Momentum", {"lookback": lookback, "top": top})

    def generate_weights(self, prices):
        ranks = prices.pct_change(self.parameters["lookback"]).rank(axis=1, ascending=False)
        return normalize_long_only((ranks <= self.parameters["top"]).astype(float))


class DonchianBreakout(Strategy):
    def __init__(self, entry=55, exit_window=20):
        super().__init__("Donchian Breakout", {"entry": entry, "exit": exit_window})

    def generate_weights(self, prices):
        asset = _primary(prices)
        s = prices[asset]
        state = pd.Series(np.nan, index=s.index)
        state[s > s.shift(1).rolling(self.parameters["entry"]).max()] = 1
        state[s < s.shift(1).rolling(self.parameters["exit"]).min()] = 0
        w = prices * 0.0
        w[asset] = state.ffill().fillna(0)
        return w


class VolatilityTargeting(Strategy):
    def __init__(self, target=0.10, window=20):
        super().__init__("Volatility Targeting", {"target": target, "window": window})

    def generate_weights(self, prices):
        asset = _primary(prices)
        vol = prices[asset].pct_change().rolling(self.parameters["window"]).std() * np.sqrt(252)
        x = (self.parameters["target"] / vol).clip(0, 1).fillna(0)
        w = prices * 0.0
        w[asset] = x
        return w


class DualMomentum(Strategy):
    def __init__(self, lookback=252):
        super().__init__("Dual Momentum Rotation", {"lookback": lookback})

    def generate_weights(self, prices):
        mom = prices.pct_change(self.parameters["lookback"])
        best = mom.eq(mom.max(axis=1), axis=0) & (mom > 0)
        return normalize_long_only(best.astype(float))


class InverseVolatility(Strategy):
    def __init__(self, window=60):
        super().__init__("Inverse-Volatility Portfolio", {"window": window})

    def generate_weights(self, prices):
        inv = 1 / prices.pct_change().rolling(self.parameters["window"]).std().replace(0, np.nan)
        return normalize_long_only(inv)
