"""Vectorized technical indicators for price analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Return the simple moving average of a series."""
    _validate_window(window)
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Return Wilder's Relative Strength Index (RSI).

    The implementation uses Wilder's recursive smoothing with an initial
    average based on the first ``window`` deltas.
    """
    _validate_window(window)
    if series.empty:
        return series.astype(float)

    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = _wilder_smooth(gains, window, start=1)
    avg_loss = _wilder_smooth(losses, window, start=1)

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))

    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    result = result.mask(both_zero, 50.0)
    result = result.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    result = result.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return result.clip(lower=0.0, upper=100.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Return Wilder's Average True Range (ATR)."""
    _validate_window(window)
    _validate_aligned_inputs(high, low, close)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return _wilder_smooth(true_range, window, start=0)


def momentum(series: pd.Series, window: int) -> pd.Series:
    """Return rate-of-change style momentum: ``(price / price.shift(window)) - 1``."""
    _validate_window(window)

    previous = series.shift(window)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = series / previous
    ratio = ratio.where(previous != 0.0)
    return ratio - 1.0


def _wilder_smooth(series: pd.Series, window: int, start: int) -> pd.Series:
    """Apply Wilder smoothing with classic initialization."""
    smooth = pd.Series(np.nan, index=series.index, dtype=float)
    end = start + window
    if len(series) < end:
        return smooth

    initial_average = series.iloc[start:end].mean()
    seed = pd.Series(np.nan, index=series.index, dtype=float)
    seed.iloc[end - 1] = float(initial_average)
    if len(series) > end:
        seed.iloc[end:] = series.iloc[end:]
    return seed.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()


def _validate_window(window: int) -> None:
    if window <= 0:
        raise ValueError("window must be greater than 0.")


def _validate_aligned_inputs(high: pd.Series, low: pd.Series, close: pd.Series) -> None:
    if not high.index.equals(low.index) or not high.index.equals(close.index):
        raise ValueError("high, low, and close must share the same index.")
