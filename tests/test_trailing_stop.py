from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.trailing_stop import ATRTrailingStop, compute_atr


def test_stop_initializes_correctly() -> None:
    engine = ATRTrailingStop(atr_multiple=2.0)
    stop = engine.initialize_stop(entry_price=100.0, atr=5.0)

    assert stop == pytest.approx(90.0)


def test_stop_only_tightens_upward() -> None:
    engine = ATRTrailingStop(atr_multiple=2.0)
    current_stop = 90.0

    # Candidate below current stop should not loosen stop.
    same_stop = engine.update_stop(current_stop=current_stop, current_price=95.0, atr=4.0)
    assert same_stop == pytest.approx(90.0)

    # Candidate above current stop should tighten upward.
    tighter_stop = engine.update_stop(current_stop=current_stop, current_price=110.0, atr=4.0)
    assert tighter_stop == pytest.approx(102.0)


def test_exit_triggers_correctly_with_gap_down() -> None:
    engine = ATRTrailingStop()

    assert engine.check_exit(current_price=100.0, stop_price=99.0) is False
    assert engine.check_exit(current_price=99.0, stop_price=99.0) is True
    assert engine.check_exit(current_price=92.0, stop_price=99.0) is True


def test_atr_calculation_correct() -> None:
    df = pd.DataFrame(
        {
            "high": [12.0, 13.0, 15.0, 14.0],
            "low": [10.0, 11.0, 12.0, 10.0],
            "close": [11.0, 12.0, 14.0, 11.0],
        }
    )
    result = compute_atr(df, period=3)

    # TR = [2, 2, 3, 4] -> ATR at idx2 = (2+2+3)/3 = 7/3
    # next ATR (rolling mean) at idx3 = (2+3+4)/3 = 3
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(7.0 / 3.0)
    assert result.iloc[3] == pytest.approx(3.0)


def test_works_on_sample_ohlc_data_with_series_and_nans() -> None:
    df = pd.DataFrame(
        {
            "high": [101.0, 103.0, 104.0, np.nan, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0, np.nan, 103.0, 104.0],
            "close": [100.0, 102.0, 103.0, np.nan, 105.0, 106.0],
        }
    )
    atr = compute_atr(df, period=3)
    engine = ATRTrailingStop(atr_multiple=2.0)

    prices = pd.Series([100.0, 102.0, 104.0, 103.0, 106.0, 108.0])
    initial = engine.initialize_stop(entry_price=prices, atr=atr.fillna(0.0))
    updated = engine.update_stop(current_stop=initial, current_price=prices, atr=atr.fillna(0.0))

    assert isinstance(updated, pd.Series)
    assert (updated >= 0.0).all()

    exits = engine.check_exit(current_price=prices, stop_price=updated)
    assert isinstance(exits, pd.Series)
    assert exits.dtype == bool
