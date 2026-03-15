from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.signal_engine import SignalEngineConfig, SignalScoringEngine


def _engine() -> SignalScoringEngine:
    return SignalScoringEngine(
        SignalEngineConfig(
            technical_weight=0.60,
            fundamental_weight=0.40,
            minimum_score_threshold=70.0,
        )
    )


def test_strong_bullish_case_passes_threshold() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "technical_score": [90.0],
            "fundamental_score": [85.0],
        }
    )

    result = _engine().score_candidates(df)

    assert result.iloc[0]["hybrid_score"] == pytest.approx(88.0)
    assert bool(result.iloc[0]["is_selected"]) is True


def test_weak_case_fails_threshold() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["BBB"],
            "technical_score": [50.0],
            "fundamental_score": [55.0],
        }
    )

    result = _engine().score_candidates(df)

    assert result.iloc[0]["hybrid_score"] == pytest.approx(52.0)
    assert bool(result.iloc[0]["is_selected"]) is False


def test_boundary_case_around_threshold_70() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["EXACT", "BELOW"],
            "technical_score": [70.0, 69.0],
            "fundamental_score": [70.0, 70.0],
        }
    )

    result = _engine().score_candidates(df, sort_descending=False)

    exact = result[result["symbol"] == "EXACT"].iloc[0]
    below = result[result["symbol"] == "BELOW"].iloc[0]

    assert exact["hybrid_score"] == pytest.approx(70.0)
    assert bool(exact["is_selected"]) is True
    assert below["hybrid_score"] == pytest.approx(69.4)
    assert bool(below["is_selected"]) is False


def test_missing_fundamentals_safety() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["HAS_FUND", "MISSING_FUND"],
            "technical_score": [80.0, 80.0],
            "fundamental_score": [75.0, np.nan],
        }
    )

    result = _engine().score_candidates(df, sort_descending=False)
    missing = result[result["symbol"] == "MISSING_FUND"].iloc[0]
    valid = result[result["symbol"] == "HAS_FUND"].iloc[0]

    assert np.isnan(missing["hybrid_score"])
    assert bool(missing["is_selected"]) is False
    assert valid["hybrid_score"] == pytest.approx(78.0)
    assert bool(valid["is_selected"]) is True


def test_nan_atr_percentile_safety() -> None:
    # ATR percentile often feeds technical_score; NaN should not crash scoring.
    df = pd.DataFrame(
        {
            "symbol": ["GOOD", "NAN_ATR_PERCENTILE"],
            "technical_score": [82.0, np.nan],
            "fundamental_score": [80.0, 80.0],
        }
    )

    result = _engine().score_candidates(df, sort_descending=False)
    nan_row = result[result["symbol"] == "NAN_ATR_PERCENTILE"].iloc[0]
    good_row = result[result["symbol"] == "GOOD"].iloc[0]

    assert np.isnan(nan_row["hybrid_score"])
    assert bool(nan_row["is_selected"]) is False
    assert good_row["hybrid_score"] == pytest.approx(81.2)
    assert bool(good_row["is_selected"]) is True
