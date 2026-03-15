from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.analytics.indicators import atr, momentum, rsi, sma


def test_sma_correct_rolling_behavior() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, window=3)

    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(result, expected)


def test_sma_insufficient_window() -> None:
    series = pd.Series([1.0, 2.0])
    result = sma(series, window=3)

    assert result.isna().all()


def test_rsi_known_value_reference() -> None:
    closes = pd.Series(
        [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
            46.03,
            46.41,
            46.22,
            45.64,
            46.21,
        ]
    )
    result = rsi(closes, window=14)

    assert result.iloc[14] == pytest.approx(70.46, abs=0.05)


def test_rsi_flat_price_handling() -> None:
    closes = pd.Series([100.0] * 30)
    result = rsi(closes, window=14)

    assert result.iloc[14:].eq(50.0).all()


def test_rsi_bounds_between_zero_and_hundred() -> None:
    closes = pd.Series([100.0, 90.0, 95.0, 85.0, 110.0, 80.0, 120.0] * 5, dtype=float)
    result = rsi(closes, window=14).dropna()

    assert (result >= 0.0).all()
    assert (result <= 100.0).all()


def test_atr_correct_true_range() -> None:
    idx = pd.RangeIndex(4)
    high = pd.Series([11.0, 13.0, 15.0, 14.0], index=idx)
    low = pd.Series([9.0, 11.0, 12.0, 10.0], index=idx)
    close = pd.Series([10.0, 12.0, 14.0, 11.0], index=idx)

    result = atr(high, low, close, window=3)

    # True ranges: 2, 3, 3, 4 -> first ATR (index 2) is mean(2,3,3) = 8/3
    assert result.iloc[2] == pytest.approx(8.0 / 3.0)


def test_atr_gap_handling() -> None:
    idx = pd.RangeIndex(3)
    high = pd.Series([10.0, 12.0, 13.0], index=idx)
    low = pd.Series([9.0, 11.0, 12.0], index=idx)
    close = pd.Series([10.0, 11.0, 12.0], index=idx)

    # Day 1 has upward gap from 10 to 12; TR should account for abs(high-prev_close)=2
    result = atr(high, low, close, window=2)
    assert result.iloc[1] == pytest.approx(1.5)


def test_atr_insufficient_window() -> None:
    high = pd.Series([10.0, 11.0])
    low = pd.Series([9.0, 10.0])
    close = pd.Series([9.5, 10.5])

    result = atr(high, low, close, window=3)
    assert result.isna().all()


def test_momentum_correct_calculation() -> None:
    series = pd.Series([100.0, 110.0, 121.0, 133.1])
    result = momentum(series, window=1)

    expected = pd.Series([np.nan, 0.1, 0.1, 0.1])
    pd.testing.assert_series_equal(result, expected)


def test_momentum_nan_handling() -> None:
    series = pd.Series([100.0, np.nan, 120.0, 130.0, 0.0, 140.0])
    result = momentum(series, window=1)

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[2])
    assert np.isnan(result.iloc[5])  # division by zero previous value
