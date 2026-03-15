from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config.config_loader import BearAllocationConfig, LiquidityConfig, StrategyConfig
from core.data.price_loader import PriceDataLoader
from core.strategy.scoring_engine import FundamentalDataProvider, ScoringEngine, StockScore


class MockFundamentals(FundamentalDataProvider):
    def __init__(self, mapping: dict[str, dict]) -> None:
        self._mapping = mapping

    def get_fundamentals(self, symbol: str, as_of_date: pd.Timestamp) -> dict:
        _ = as_of_date
        return self._mapping.get(symbol, {})


def _build_config(score_threshold: int = 0) -> StrategyConfig:
    return StrategyConfig(
        score_threshold=score_threshold,
        max_positions=10,
        rebalance_frequency="monthly",
        atr_multiple=2.0,
        bear_allocation=BearAllocationConfig(liquid_etf=0.6, gold_etf=0.4),
        liquidity=LiquidityConfig(min_avg_turnover=50_000_000.0),
    )


def _write_symbol_parquet(
    base_path: Path,
    symbol: str,
    closes: np.ndarray,
    volume: float = 1_000_000.0,
) -> None:
    prices_dir = base_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    periods = len(closes)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=periods),
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": [volume] * periods,
            "adj_close": closes,
        }
    )
    frame.to_parquet(prices_dir / f"{symbol}.parquet", index=False)


def _build_engine(tmp_path: Path, threshold: int = 0) -> tuple[ScoringEngine, pd.Timestamp]:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAA", np.linspace(100.0, 220.0, 300))
    _write_symbol_parquet(base_path, "BBB", np.linspace(150.0, 155.0, 300))
    _write_symbol_parquet(base_path, "CCC", np.linspace(220.0, 100.0, 300))

    fundamentals = MockFundamentals(
        {
            "AAA": {"earnings_growth": 0.30, "roe": 0.28, "dividend_yield": 0.01},
            "BBB": {"earnings_growth": 0.10, "roe": 0.14, "dividend_yield": 0.02},
            "CCC": {"earnings_growth": 0.02, "roe": 0.05, "dividend_yield": 0.03},
        }
    )
    loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)
    engine = ScoringEngine(loader, fundamentals, _build_config(score_threshold=threshold))
    as_of_date = pd.Timestamp("2024-02-23")
    return engine, as_of_date


def test_scoring_runs_end_to_end(tmp_path: Path) -> None:
    engine, as_of_date = _build_engine(tmp_path, threshold=0)

    results = engine.score_universe(["AAA", "BBB", "CCC"], as_of_date)

    assert results
    assert all(isinstance(item, StockScore) for item in results)


def test_weights_sum_correctly(tmp_path: Path) -> None:
    engine, as_of_date = _build_engine(tmp_path, threshold=0)
    results = engine.score_universe(["AAA", "BBB", "CCC"], as_of_date)

    for item in results:
        expected = 0.60 * item.technical_score + 0.40 * item.fundamental_score
        assert item.score == pytest.approx(expected)


def test_handles_missing_fundamentals(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAA", np.linspace(100.0, 210.0, 300))
    _write_symbol_parquet(base_path, "BBB", np.linspace(100.0, 205.0, 300))
    fundamentals = MockFundamentals(
        {"AAA": {"earnings_growth": 0.2, "roe": 0.2, "dividend_yield": 0.01}}
    )
    engine = ScoringEngine(
        PriceDataLoader(base_path=str(base_path), use_adjusted=False),
        fundamentals,
        _build_config(score_threshold=0),
    )

    results = engine.score_universe(["AAA", "BBB"], pd.Timestamp("2024-02-23"))
    by_symbol = {item.symbol: item for item in results}

    assert "BBB" in by_symbol
    assert 0.0 <= by_symbol["BBB"].fundamental_score <= 100.0


def test_deterministic_ranking(tmp_path: Path) -> None:
    engine, as_of_date = _build_engine(tmp_path, threshold=0)

    first = engine.score_universe(["AAA", "BBB", "CCC"], as_of_date)
    second = engine.score_universe(["AAA", "BBB", "CCC"], as_of_date)

    assert [(s.symbol, s.score) for s in first] == [(s.symbol, s.score) for s in second]


def test_score_bounds_zero_to_hundred(tmp_path: Path) -> None:
    engine, as_of_date = _build_engine(tmp_path, threshold=0)
    results = engine.score_universe(["AAA", "BBB", "CCC"], as_of_date)

    for item in results:
        assert 0.0 <= item.technical_score <= 100.0
        assert 0.0 <= item.fundamental_score <= 100.0
        assert 0.0 <= item.score <= 100.0


def test_threshold_filtering_ready(tmp_path: Path) -> None:
    engine_low, as_of_date = _build_engine(tmp_path, threshold=0)
    engine_high, _ = _build_engine(tmp_path, threshold=85)

    low = engine_low.score_universe(["AAA", "BBB", "CCC"], as_of_date)
    high = engine_high.score_universe(["AAA", "BBB", "CCC"], as_of_date)

    assert len(high) <= len(low)
    assert all(item.score >= 85.0 for item in high)
