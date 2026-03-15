"""Market regime engine with weekly EMA200 detection and bear allocation policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MarketRegime(str, Enum):
    """Supported market regimes."""

    BULL = "BULL"
    BEAR = "BEAR"


@dataclass(frozen=True)
class RegimeDecision:
    """Output payload for one regime evaluation cycle."""

    evaluated: bool
    evaluation_date: str
    regime: MarketRegime
    switched: bool
    target_allocation: dict[str, float]


class BearRegimeAllocationEngine:
    """Weekly regime evaluator and allocation policy manager.

    Detection rule:
    - BULL when ``index_close >= ema200``
    - BEAR when ``index_close < ema200``

    Behavior:
    - Weekly evaluation only (idempotent within same week).
    - BEAR target allocation:
      - 60% Liquid ETF
      - 40% Gold ETF
    - Automatic recovery to BULL allocation when signal returns bullish.
    """

    BEAR_ALLOCATION = {"LIQUID_ETF": 0.60, "GOLD_ETF": 0.40}

    def __init__(
        self,
        evaluation_frequency: str = "weekly",
        default_bull_allocation: dict[str, float] | None = None,
    ) -> None:
        if evaluation_frequency != "weekly":
            raise ValueError("evaluation_frequency must be 'weekly'.")
        self.evaluation_frequency = evaluation_frequency
        self.default_bull_allocation = default_bull_allocation or {}

        self._current_regime: MarketRegime = MarketRegime.BULL
        self._last_evaluated_week: tuple[int, int] | None = None

    @property
    def current_regime(self) -> MarketRegime:
        """Return currently active regime state."""
        return self._current_regime

    def evaluate(
        self,
        as_of_date: pd.Timestamp,
        index_close: float,
        ema200: float,
    ) -> RegimeDecision:
        """Evaluate regime signal and return idempotent decision payload."""
        ts = pd.Timestamp(as_of_date).normalize()
        if pd.isna(index_close) or pd.isna(ema200):
            raise ValueError("index_close and ema200 must be valid numeric values.")
        if index_close <= 0 or ema200 <= 0:
            raise ValueError("index_close and ema200 must be greater than 0.")

        week_key = self._week_key(ts)
        if self._last_evaluated_week == week_key:
            return RegimeDecision(
                evaluated=False,
                evaluation_date=ts.date().isoformat(),
                regime=self._current_regime,
                switched=False,
                target_allocation=self.target_allocation(),
            )

        self._last_evaluated_week = week_key
        new_regime = MarketRegime.BULL if float(index_close) >= float(ema200) else MarketRegime.BEAR
        switched = new_regime != self._current_regime
        self._current_regime = new_regime

        return RegimeDecision(
            evaluated=True,
            evaluation_date=ts.date().isoformat(),
            regime=self._current_regime,
            switched=switched,
            target_allocation=self.target_allocation(),
        )

    def target_allocation(self) -> dict[str, float]:
        """Return target allocation for current regime."""
        if self._current_regime == MarketRegime.BEAR:
            return dict(self.BEAR_ALLOCATION)
        return dict(self.default_bull_allocation)

    def reset(self) -> None:
        """Reset runtime state for deterministic replay/testing."""
        self._current_regime = MarketRegime.BULL
        self._last_evaluated_week = None

    def _week_key(self, ts: pd.Timestamp) -> tuple[int, int]:
        iso = ts.isocalendar()
        return int(iso.year), int(iso.week)
