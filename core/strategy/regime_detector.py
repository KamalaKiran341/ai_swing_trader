"""Market regime detection based on long-term trend of an index."""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum

import pandas as pd

from core.data.price_loader import PriceDataLoader


class Regime(str, Enum):
    """Supported market regimes."""

    BULL = "BULL"
    BEAR = "BEAR"


class MarketRegimeDetector:
    """Detect market regime from index close relative to rolling SMA."""

    _MAX_CACHE_ENTRIES = 256

    def __init__(
        self,
        price_loader: PriceDataLoader,
        index_symbol: str = "NIFTY",
        sma_window: int = 200,
        evaluation_frequency: str = "weekly",
    ) -> None:
        """Create a new regime detector.

        Args:
            price_loader: Loader for index OHLCV data.
            index_symbol: Symbol used as the market index benchmark.
            sma_window: Rolling close window used for SMA calculation.
            evaluation_frequency: Frequency of regime evaluations.
        """
        if sma_window <= 0:
            raise ValueError("sma_window must be greater than 0.")
        if evaluation_frequency != "weekly":
            raise ValueError("evaluation_frequency must be 'weekly'.")

        self._price_loader = price_loader
        self._index_symbol = index_symbol
        self._sma_window = sma_window
        self._evaluation_frequency = evaluation_frequency
        self._regime_cache: OrderedDict[pd.Timestamp, Regime] = OrderedDict()

    def get_regime(self, as_of_date: pd.Timestamp) -> Regime:
        """Return the market regime for ``as_of_date``.

        The detector evaluates at weekly frequency by mapping ``as_of_date`` to
        week-end (Friday), then using the most recent trading day on or before
        that date.
        """
        eval_date = self._normalize_evaluation_date(as_of_date)

        cached = self._regime_cache.get(eval_date)
        if cached is not None:
            self._regime_cache.move_to_end(eval_date)
            return cached

        try:
            history = self._price_loader.get_history(
                symbol=self._index_symbol,
                end_date=eval_date,
                lookback=self._sma_window,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f"Index data for symbol '{self._index_symbol}' is unavailable."
            ) from exc
        if len(history) < self._sma_window:
            raise ValueError(
                f"Insufficient history for index '{self._index_symbol}' as of {eval_date.date()}: "
                f"need {self._sma_window} rows, got {len(history)}."
            )

        close = history["close"]
        if close.isna().any():
            raise ValueError(
                f"Index close data contains NaN values for symbol '{self._index_symbol}'."
            )

        sma = close.rolling(window=self._sma_window, min_periods=self._sma_window).mean().iloc[-1]
        latest_close = close.iloc[-1]
        regime = Regime.BULL if float(latest_close) >= float(sma) else Regime.BEAR

        self._regime_cache[eval_date] = regime
        self._regime_cache.move_to_end(eval_date)
        if len(self._regime_cache) > self._MAX_CACHE_ENTRIES:
            self._regime_cache.popitem(last=False)

        return regime

    def _normalize_evaluation_date(self, as_of_date: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(as_of_date)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if self._evaluation_frequency == "weekly":
            week_end = ts.to_period("W-FRI").end_time.normalize()
            if week_end > ts.normalize():
                week_end = week_end - pd.Timedelta(days=7)
            return week_end
        return ts
