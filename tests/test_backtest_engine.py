from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.strategy.regime_detector import Regime
from core.strategy.scoring_engine import StockScore
from src.backtest.backtest_engine import (
    GOLD_ETF_SYMBOL,
    LIQUID_ETF_SYMBOL,
    BacktestEngine,
    BacktestResult,
)
from src.execution.order_simulator import OrderSimulator
from src.risk.portfolio_risk import PortfolioRiskController
from src.risk.trailing_stop import ATRTrailingStop


@dataclass
class _Config:
    max_positions: int = 2
    initial_capital: float = 100_000.0


class _RegimeDetectorStub:
    def __init__(self, regime: Regime) -> None:
        self._regime = regime

    def get_regime(self, as_of_date: pd.Timestamp) -> Regime:
        _ = as_of_date
        return self._regime


class _ScoringEngineStub:
    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking

    def score_universe(self, symbols: list[str], as_of_date: pd.Timestamp) -> list[StockScore]:
        _ = as_of_date
        available = [s for s in self._ranking if s in symbols]
        output: list[StockScore] = []
        score = 100.0
        for symbol in available:
            output.append(
                StockScore(
                    symbol=symbol,
                    score=score,
                    technical_score=score,
                    fundamental_score=score,
                )
            )
            score -= 1.0
        return output


def _price_frame(dates: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(close), 1_000_000.0),
        },
        index=dates,
    )


def _build_price_data(with_drop: bool = False) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    aaa = np.linspace(100.0, 120.0, len(dates))
    if with_drop:
        aaa[-10:] = np.linspace(120.0, 80.0, 10)
    bbb = np.linspace(90.0, 110.0, len(dates))
    etf_l = np.linspace(50.0, 52.0, len(dates))
    etf_g = np.linspace(40.0, 43.0, len(dates))
    return {
        "AAA": _price_frame(dates, aaa),
        "BBB": _price_frame(dates, bbb),
        LIQUID_ETF_SYMBOL: _price_frame(dates, etf_l),
        GOLD_ETF_SYMBOL: _price_frame(dates, etf_g),
    }


def _build_engine(
    price_data: dict[str, pd.DataFrame],
    regime: Regime = Regime.BULL,
    ranking: list[str] | None = None,
) -> BacktestEngine:
    if ranking is None:
        ranking = ["AAA", "BBB"]
    return BacktestEngine(
        price_data=price_data,
        fundamental_data={},
        regime_detector=_RegimeDetectorStub(regime),
        scoring_engine=_ScoringEngineStub(ranking),
        exit_engine=ATRTrailingStop(atr_multiple=2.0),
        risk_controller=PortfolioRiskController(
            max_positions=2, losing_month_tolerance=2, circuit_breaker_enabled=True
        ),
        order_simulator=OrderSimulator(slippage_bps=0.0),
        config=_Config(max_positions=2, initial_capital=100_000.0),
    )


def test_basic_run_completes() -> None:
    engine = _build_engine(_build_price_data())
    result = engine.run_backtest()

    assert isinstance(result, BacktestResult)
    assert not result.equity_curve.empty


def test_monthly_rebalance_occurs() -> None:
    engine = _build_engine(_build_price_data())
    engine.run_backtest()

    assert engine._last_rebalance_month == (2024, 3)


def test_bear_regime_reallocates() -> None:
    engine = _build_engine(_build_price_data(), regime=Regime.BEAR)
    result = engine.run_backtest()

    bought_symbols = {t.symbol for t in result.trades if t.side == "BUY"}
    assert LIQUID_ETF_SYMBOL in bought_symbols
    assert GOLD_ETF_SYMBOL in bought_symbols
    assert "AAA" not in bought_symbols
    assert "BBB" not in bought_symbols


def test_trailing_stops_trigger() -> None:
    engine = _build_engine(_build_price_data(with_drop=True), regime=Regime.BULL, ranking=["AAA"])
    result = engine.run_backtest()

    sells = [t for t in result.trades if t.side == "SELL" and t.symbol == "AAA"]
    assert sells


def test_circuit_breaker_blocks_trades() -> None:
    engine = _build_engine(_build_price_data(), regime=Regime.BULL)
    engine.risk_controller.circuit_breaker_active = True
    result = engine.run_backtest()

    buys = [t for t in result.trades if t.side == "BUY"]
    assert not buys


def test_equity_curve_monotonic_logic_valid() -> None:
    engine = _build_engine(_build_price_data())
    result = engine.run_backtest()

    assert result.equity_curve.index.is_monotonic_increasing
    assert result.equity_curve.notna().all()
    assert (result.equity_curve >= 0.0).all()
