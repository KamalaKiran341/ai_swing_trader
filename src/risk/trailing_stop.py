"""ATR-based trailing stop utilities for long positions."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

Scalar: TypeAlias = float | int | np.floating
PriceInput: TypeAlias = Scalar | pd.Series


class ATRTrailingStop:
    """Long-only ATR trailing stop engine.

    The stop is initialized below entry and can only tighten upward.
    """

    def __init__(self, atr_multiple: float = 2.0) -> None:
        """Create a trailing stop engine.

        Args:
            atr_multiple: Multiplier applied to ATR for stop distance.
        """
        if atr_multiple <= 0:
            raise ValueError("atr_multiple must be greater than 0.")
        self.atr_multiple = float(atr_multiple)

    def initialize_stop(self, entry_price: PriceInput, atr: PriceInput) -> PriceInput:
        """Initialize stop as ``entry_price - atr_multiple * atr`` for long positions.

        Stops are floored at zero to avoid negative values.
        """
        candidate = _subtract_atr_distance(entry_price, atr, self.atr_multiple)
        return _clip_non_negative(candidate)

    def update_stop(
        self,
        current_stop: PriceInput,
        current_price: PriceInput,
        atr: PriceInput,
    ) -> PriceInput:
        """Update stop using long-only trailing rule.

        New stop candidate: ``current_price - atr_multiple * atr``.
        The stop can only move upward, so the result is
        ``max(current_stop, new_stop_candidate)`` with NaN-safe behavior.
        """
        candidate = _clip_non_negative(
            _subtract_atr_distance(current_price, atr, self.atr_multiple)
        )
        return _nan_safe_max(current_stop, candidate)

    def check_exit(self, current_price: PriceInput, stop_price: PriceInput) -> bool | pd.Series:
        """Return whether long position should exit (`price <= stop`)."""
        if isinstance(current_price, pd.Series) and isinstance(stop_price, pd.Series):
            return current_price.astype(float).le(stop_price.astype(float))
        if isinstance(current_price, pd.Series):
            return current_price.astype(float).le(float(stop_price))
        if isinstance(stop_price, pd.Series):
            return pd.Series(float(current_price), index=stop_price.index).le(
                stop_price.astype(float)
            )

        if pd.isna(current_price) or pd.isna(stop_price):
            return False
        return bool(float(current_price) <= float(stop_price))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute classic ATR from OHLC data using rolling mean.

    ATR is based on True Range (TR):
    ``max(high - low, abs(high - prev_close), abs(low - prev_close))``.
    """
    if period <= 0:
        raise ValueError("period must be greater than 0.")

    required = {"high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for ATR: {missing}")

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


def _subtract_atr_distance(price: PriceInput, atr: PriceInput, multiple: float) -> PriceInput:
    if isinstance(price, pd.Series) and isinstance(atr, pd.Series):
        return price.astype(float) - (multiple * atr.astype(float))
    if isinstance(price, pd.Series):
        return price.astype(float) - (multiple * float(atr))
    if isinstance(atr, pd.Series):
        return float(price) - (multiple * atr.astype(float))
    return float(price) - (multiple * float(atr))


def _clip_non_negative(value: PriceInput) -> PriceInput:
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0)
    if pd.isna(value):
        return float("nan")
    return max(0.0, float(value))


def _nan_safe_max(left: PriceInput, right: PriceInput) -> PriceInput:
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        combined = pd.concat([left.astype(float), right.astype(float)], axis=1)
        return combined.max(axis=1, skipna=True)
    if isinstance(left, pd.Series):
        combined = pd.concat(
            [left.astype(float), pd.Series(float(right), index=left.index)], axis=1
        )
        return combined.max(axis=1, skipna=True)
    if isinstance(right, pd.Series):
        combined = pd.concat(
            [pd.Series(float(left), index=right.index), right.astype(float)], axis=1
        )
        return combined.max(axis=1, skipna=True)

    left_nan = pd.isna(left)
    right_nan = pd.isna(right)
    if left_nan and right_nan:
        return float("nan")
    if left_nan:
        return float(right)
    if right_nan:
        return float(left)
    return max(float(left), float(right))
