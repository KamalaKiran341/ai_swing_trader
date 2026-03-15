from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config.config_loader import (
    BearAllocationConfig,
    LiquidityConfig,
    StrategyConfig,
)
from core.data.liquidity_filter import LiquidityFilter
from core.data.price_loader import PriceDataLoader


def _build_config(min_avg_turnover: float) -> StrategyConfig:
    return StrategyConfig(
        score_threshold=70,
        max_positions=10,
        rebalance_frequency="monthly",
        atr_multiple=2.0,
        bear_allocation=BearAllocationConfig(liquid_etf=0.6, gold_etf=0.4),
        liquidity=LiquidityConfig(min_avg_turnover=min_avg_turnover),
    )


def _write_symbol_parquet(base_path: Path, symbol: str, frame: pd.DataFrame) -> None:
    prices_dir = base_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(prices_dir / f"{symbol}.parquet", index=False)


def _frame(close: float = 100.0, volume: float = 1_000_000.0, periods: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=periods, freq="D"),
            "open": [close] * periods,
            "high": [close + 1] * periods,
            "low": [close - 1] * periods,
            "close": [close] * periods,
            "volume": [volume] * periods,
            "adj_close": [close] * periods,
        }
    )


def test_passes_when_above_threshold(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "LIQ", _frame(close=100.0, volume=1_500_000.0))
    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    liquidity_filter = LiquidityFilter(loader, _build_config(min_avg_turnover=100_000_000.0))

    assert liquidity_filter.is_liquid("LIQ", pd.Timestamp("2024-01-25"))


def test_fails_when_below_threshold(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "ILLQ", _frame(close=50.0, volume=100_000.0))
    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    liquidity_filter = LiquidityFilter(loader, _build_config(min_avg_turnover=10_000_000.0))

    assert not liquidity_filter.is_liquid("ILLQ", pd.Timestamp("2024-01-25"))


def test_insufficient_history_returns_false(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "SHORT", _frame(periods=10))
    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    liquidity_filter = LiquidityFilter(loader, _build_config(min_avg_turnover=1.0))

    assert not liquidity_filter.is_liquid("SHORT", pd.Timestamp("2024-01-10"))


def test_handles_zero_volume_days(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    frame = _frame(close=100.0, volume=2_000_000.0)
    frame.loc[5:10, "volume"] = 0.0
    _write_symbol_parquet(base_path, "ZERO", frame)
    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    liquidity_filter = LiquidityFilter(loader, _build_config(min_avg_turnover=150_000_000.0))

    assert not liquidity_filter.is_liquid("ZERO", pd.Timestamp("2024-01-25"))


def test_handles_nans_safely(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    frame = _frame(close=100.0, volume=2_000_000.0)
    frame.loc[20, "close"] = float("nan")
    _write_symbol_parquet(base_path, "NAN", frame)
    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    liquidity_filter = LiquidityFilter(loader, _build_config(min_avg_turnover=1.0))

    assert not liquidity_filter.is_liquid("NAN", pd.Timestamp("2024-01-25"))


def test_vectorized_behavior_sanity(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.Series(range(100, 130), dtype=float)
    volume = pd.Series([1_000_000.0] * 30, dtype=float)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
            "adj_close": close,
        }
    )
    _write_symbol_parquet(base_path, "VEC", frame)

    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    expected = float((close.iloc[-20:] * volume.iloc[-20:]).mean())
    liquidity_filter = LiquidityFilter(loader, _build_config(min_avg_turnover=expected))

    assert liquidity_filter.is_liquid("VEC", pd.Timestamp("2024-01-30"))


def test_missing_symbol_returns_false(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    liquidity_filter = LiquidityFilter(loader, _build_config(min_avg_turnover=1.0))

    assert not liquidity_filter.is_liquid("MISSING", pd.Timestamp("2024-01-30"))
