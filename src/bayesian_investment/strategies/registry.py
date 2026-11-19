"""Configuration-friendly strategy registry."""

from .implementations import (
    BollingerMeanReversion,
    BuyHold,
    CrossSectionalMomentum,
    DonchianBreakout,
    DualMomentum,
    EMACrossover,
    InverseVolatility,
    MACDTrend,
    RSIMeanReversion,
    SMACrossover,
    TimeSeriesMomentum,
    VolatilityTargeting,
)

STRATEGIES = {
    "buy_hold": BuyHold,
    "sma": SMACrossover,
    "ema": EMACrossover,
    "rsi": RSIMeanReversion,
    "macd": MACDTrend,
    "bollinger": BollingerMeanReversion,
    "time_series_momentum": TimeSeriesMomentum,
    "cross_sectional_momentum": CrossSectionalMomentum,
    "donchian": DonchianBreakout,
    "volatility_targeting": VolatilityTargeting,
    "dual_momentum": DualMomentum,
    "inverse_volatility": InverseVolatility,
}


def build_strategies():
    return [constructor() for constructor in STRATEGIES.values()]
