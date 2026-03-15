"""Liquidity filtering based on rolling average turnover."""

from __future__ import annotations

import pandas as pd

from core.config.config_loader import StrategyConfig
from core.data.price_loader import PriceDataLoader


class LiquidityFilter:
    """Evaluate if a symbol meets the minimum liquidity threshold."""

    _LOOKBACK_DAYS = 20

    def __init__(self, price_loader: PriceDataLoader, config: StrategyConfig) -> None:
        """Create a liquidity filter from loader and strategy configuration."""
        self._price_loader = price_loader
        self._min_avg_turnover = float(config.liquidity.min_avg_turnover)

    def is_liquid(self, symbol: str, as_of_date: pd.Timestamp) -> bool:
        """Return whether ``symbol`` is liquid at ``as_of_date``.

        Liquidity is defined as the mean 20-day turnover (``close * volume``)
        ending at ``as_of_date`` being above or equal to the configured minimum.
        """
        try:
            history = self._price_loader.get_history(
                symbol=symbol, end_date=as_of_date, lookback=self._LOOKBACK_DAYS
            )
        except (FileNotFoundError, ValueError):
            return False

        if len(history) < self._LOOKBACK_DAYS:
            return False

        turnover = history["close"] * history["volume"]
        if turnover.isna().any():
            return False

        avg_turnover = float(turnover.mean())
        return avg_turnover >= self._min_avg_turnover
