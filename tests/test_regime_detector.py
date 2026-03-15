from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.data.price_loader import PriceDataLoader
from core.strategy.regime_detector import MarketRegimeDetector, Regime


class CountingPriceDataLoader:
    """Delegating loader used to validate detector caching behavior."""

    def __init__(self, delegate: PriceDataLoader) -> None:
        self._delegate = delegate
        self.call_count = 0

    def get_history(self, symbol: str, end_date: pd.Timestamp, lookback: int) -> pd.DataFrame:
        self.call_count += 1
        return self._delegate.get_history(symbol=symbol, end_date=end_date, lookback=lookback)


def _write_symbol_parquet(base_path: Path, symbol: str, close: pd.Series) -> None:
    prices_dir = base_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    periods = len(close)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=periods),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1_000_000.0] * periods,
            "adj_close": close,
        }
    )
    frame.to_parquet(prices_dir / f"{symbol}.parquet", index=False)


def test_bull_regime_detection(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    close = pd.Series([100.0] * 219 + [120.0], dtype=float)
    _write_symbol_parquet(base_path, "NIFTY", close)

    detector = MarketRegimeDetector(PriceDataLoader(str(base_path), use_adjusted=False))
    regime = detector.get_regime(pd.Timestamp("2024-11-01"))

    assert regime is Regime.BULL


def test_bear_regime_detection(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    close = pd.Series([100.0] * 219 + [80.0], dtype=float)
    _write_symbol_parquet(base_path, "NIFTY", close)

    detector = MarketRegimeDetector(PriceDataLoader(str(base_path), use_adjusted=False))
    regime = detector.get_regime(pd.Timestamp("2024-11-01"))

    assert regime is Regime.BEAR


def test_exact_boundary_close_equals_sma_is_bull(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    close = pd.Series([100.0] * 220, dtype=float)
    _write_symbol_parquet(base_path, "NIFTY", close)

    detector = MarketRegimeDetector(PriceDataLoader(str(base_path), use_adjusted=False))
    regime = detector.get_regime(pd.Timestamp("2024-11-01"))

    assert regime is Regime.BULL


def test_insufficient_history_raises(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    close = pd.Series([100.0] * 50, dtype=float)
    _write_symbol_parquet(base_path, "NIFTY", close)

    detector = MarketRegimeDetector(PriceDataLoader(str(base_path), use_adjusted=False))
    with pytest.raises(ValueError, match="Insufficient history"):
        detector.get_regime(pd.Timestamp("2024-11-01"))


def test_non_trading_day_handling(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    close = pd.Series([100.0] * 220, dtype=float)
    _write_symbol_parquet(base_path, "NIFTY", close)

    detector = MarketRegimeDetector(PriceDataLoader(str(base_path), use_adjusted=False))
    friday_regime = detector.get_regime(pd.Timestamp("2024-11-01"))  # Friday
    weekend_regime = detector.get_regime(pd.Timestamp("2024-11-03"))  # Sunday

    assert weekend_regime is friday_regime


def test_weekly_evaluation_consistency(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    close = pd.Series([100.0] * 219 + [110.0], dtype=float)
    _write_symbol_parquet(base_path, "NIFTY", close)

    detector = MarketRegimeDetector(PriceDataLoader(str(base_path), use_adjusted=False))
    tuesday = detector.get_regime(pd.Timestamp("2024-11-05"))
    thursday = detector.get_regime(pd.Timestamp("2024-11-07"))

    assert tuesday is thursday


def test_caching_behavior_sanity(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    close = pd.Series([100.0] * 220, dtype=float)
    _write_symbol_parquet(base_path, "NIFTY", close)

    counting_loader = CountingPriceDataLoader(PriceDataLoader(str(base_path), use_adjusted=False))
    detector = MarketRegimeDetector(counting_loader)
    detector.get_regime(pd.Timestamp("2024-11-07"))
    detector.get_regime(pd.Timestamp("2024-11-07"))
    detector.get_regime(pd.Timestamp("2024-11-05"))

    assert counting_loader.call_count == 1
