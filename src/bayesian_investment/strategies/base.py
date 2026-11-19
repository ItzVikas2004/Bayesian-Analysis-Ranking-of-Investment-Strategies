"""Common strategy protocol and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Strategy(ABC):
    """Base class: strategies transform historical prices into target weights."""

    name: str
    parameters: dict = field(default_factory=dict)

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return date-by-asset close-observable target weights."""

    def get_parameters(self) -> dict:
        return dict(self.parameters)


def normalize_long_only(score: pd.DataFrame) -> pd.DataFrame:
    score = score.clip(lower=0).fillna(0)
    total = score.sum(axis=1).replace(0, float("nan"))
    return score.div(total, axis=0).fillna(0)
