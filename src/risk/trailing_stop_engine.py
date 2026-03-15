"""ATR trailing-stop engine (Method B) for long positions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrailingStopSnapshot:
    """Daily trailing-stop state snapshot."""

    highest_close: float
    stop_price: float
    exit_triggered: bool


class ATRTrailingStopEngine:
    """Method-B ATR trailing stop engine.

    Method B definition:
    ``stop = highest_close_since_entry - atr_multiple * atr``

    Design guarantees:
    - Stop is monotonic non-decreasing.
    - NaN/invalid ATR does not corrupt stop state.
    - Gap-down exits are safely captured via ``price <= stop``.
    """

    def __init__(self, atr_multiple: float = 2.0) -> None:
        if atr_multiple <= 0:
            raise ValueError("atr_multiple must be greater than 0.")
        self.atr_multiple = float(atr_multiple)

    def initialize(self, entry_price: float, atr: float | None) -> TrailingStopSnapshot:
        """Initialize trailing stop state at position entry."""
        if entry_price <= 0:
            raise ValueError("entry_price must be greater than 0.")
        highest_close = float(entry_price)
        stop = self._compute_candidate_stop(highest_close=highest_close, atr=atr)
        if stop is None:
            stop = max(highest_close * 0.90, 0.0)
        return TrailingStopSnapshot(
            highest_close=highest_close, stop_price=float(stop), exit_triggered=False
        )

    def evaluate_day(
        self,
        current_close: float,
        atr: float | None,
        previous_snapshot: TrailingStopSnapshot,
    ) -> TrailingStopSnapshot:
        """Evaluate stop and exit state for one trading day.

        Args:
            current_close: Current daily close price.
            atr: ATR value for the day. If NaN/missing/non-positive, stop is not widened.
            previous_snapshot: Prior state from previous day.
        """
        if current_close <= 0:
            raise ValueError("current_close must be greater than 0.")

        new_highest = max(float(previous_snapshot.highest_close), float(current_close))
        candidate = self._compute_candidate_stop(highest_close=new_highest, atr=atr)

        if candidate is None:
            # Missing ATR: keep monotonic state unchanged.
            new_stop = float(previous_snapshot.stop_price)
        else:
            candidate = max(candidate, 0.0)
            new_stop = max(float(previous_snapshot.stop_price), candidate)

        exit_triggered = self.should_exit(current_price=current_close, stop_price=new_stop)
        return TrailingStopSnapshot(
            highest_close=float(new_highest),
            stop_price=float(new_stop),
            exit_triggered=bool(exit_triggered),
        )

    def should_exit(self, current_price: float, stop_price: float) -> bool:
        """Return True when long position should exit (gap-down safe)."""
        if current_price <= 0:
            raise ValueError("current_price must be greater than 0.")
        if stop_price < 0:
            raise ValueError("stop_price cannot be negative.")
        return bool(float(current_price) <= float(stop_price))

    def _compute_candidate_stop(self, highest_close: float, atr: float | None) -> float | None:
        if atr is None or pd.isna(atr):
            return None
        atr_value = float(atr)
        if atr_value <= 0:
            return None
        return float(highest_close - self.atr_multiple * atr_value)
