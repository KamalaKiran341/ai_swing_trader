from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from core.strategy.regime_detector import Regime
from core.strategy.scoring_engine import StockScore
from src.execution.order_simulator import OrderSimulator
from src.live.paper_trading_engine import (
    BacktestComponentsBundle,
    JsonFileStateStore,
    PaperTradingConfig,
    PaperTradingEngine,
)
from src.risk.portfolio_risk import PortfolioRiskController
from src.risk.trailing_stop import ATRTrailingStop


class _DataProviderStub:
    def __init__(self) -> None:
        self._prices: dict[date, dict[str, dict[str, float]]] = {}
        self._candidates: dict[date, list[str]] = {}
        self._atr: dict[str, float] = {}

    def set_day(
        self,
        day: date,
        prices: dict[str, dict[str, float]],
        candidates: list[str] | None = None,
    ) -> None:
        self._prices[day] = prices
        if candidates is not None:
            self._candidates[day] = candidates

    def get_prices(self, day: date) -> dict[str, dict[str, float]]:
        return self._prices.get(day, {})

    def get_candidate_symbols(self, day: date) -> list[str]:
        return self._candidates.get(day, [])

    def get_atr(self, symbol: str, as_of: pd.Timestamp) -> float:
        _ = as_of
        return self._atr.get(symbol, 2.0)


class _RegimeDetectorStub:
    def get_regime(self, as_of_date: pd.Timestamp) -> Regime:
        _ = as_of_date
        return Regime.BULL


class _ScoringEngineStub:
    def score_universe(self, symbols: list[str], as_of_date: pd.Timestamp) -> list[StockScore]:
        _ = as_of_date
        ordered = sorted(symbols)
        score = 100.0
        out: list[StockScore] = []
        for symbol in ordered:
            out.append(
                StockScore(
                    symbol=symbol,
                    score=score,
                    technical_score=score,
                    fundamental_score=score,
                )
            )
            score -= 1.0
        return out


@dataclass(frozen=True)
class _BundleFactory:
    @staticmethod
    def create() -> BacktestComponentsBundle:
        return BacktestComponentsBundle(
            regime_detector=_RegimeDetectorStub(),
            scoring_engine=_ScoringEngineStub(),
            exit_engine=ATRTrailingStop(atr_multiple=2.0),
        )


def _engine(
    tmp_path: Path, provider: _DataProviderStub
) -> tuple[PaperTradingEngine, JsonFileStateStore]:
    store = JsonFileStateStore(str(tmp_path / "paper_state.json"))
    engine = PaperTradingEngine(
        data_provider=provider,
        backtest_components_bundle=_BundleFactory.create(),
        order_simulator=OrderSimulator(slippage_bps=0.0),
        risk_controller=PortfolioRiskController(
            max_positions=3, losing_month_tolerance=2, circuit_breaker_enabled=True
        ),
        config=PaperTradingConfig(starting_capital=500_000.0, max_positions=2),
        state_store=store,
    )
    return engine, store


def test_daily_run_updates_equity(tmp_path: Path) -> None:
    provider = _DataProviderStub()
    d1 = date(2024, 1, 2)
    provider.set_day(
        d1,
        prices={"AAA": {"open": 100.0, "close": 101.0}, "BBB": {"open": 200.0, "close": 202.0}},
        candidates=["AAA", "BBB"],
    )
    engine, _ = _engine(tmp_path, provider)
    state = engine.run_daily(d1)

    assert len(state.equity_curve) == 1
    assert state.equity_curve[-1]["equity"] > 0.0


def test_state_persists_correctly(tmp_path: Path) -> None:
    provider = _DataProviderStub()
    d1 = date(2024, 1, 2)
    provider.set_day(d1, prices={"AAA": {"open": 100.0, "close": 101.0}}, candidates=["AAA"])
    engine, store = _engine(tmp_path, provider)
    state = engine.run_daily(d1)

    loaded = store.load_state()
    assert loaded is not None
    assert loaded.current_capital == state.current_capital
    assert loaded.equity_curve == state.equity_curve


def test_idempotent_same_day_run(tmp_path: Path) -> None:
    provider = _DataProviderStub()
    d1 = date(2024, 1, 2)
    provider.set_day(d1, prices={"AAA": {"open": 100.0, "close": 101.0}}, candidates=["AAA"])
    engine, _ = _engine(tmp_path, provider)

    first = engine.run_daily(d1)
    second = engine.run_daily(d1)

    assert len(first.equity_curve) == 1
    assert len(second.equity_curve) == 1
    assert first.open_positions == second.open_positions


def test_monthly_rebalance_fires(tmp_path: Path) -> None:
    provider = _DataProviderStub()
    jan = date(2024, 1, 2)
    feb = date(2024, 2, 1)
    provider.set_day(jan, prices={"AAA": {"open": 100.0, "close": 101.0}}, candidates=["AAA"])
    provider.set_day(feb, prices={"BBB": {"open": 150.0, "close": 151.0}}, candidates=["BBB"])
    engine, _ = _engine(tmp_path, provider)

    state_jan = engine.run_daily(jan)
    state_feb = engine.run_daily(feb)

    assert state_jan.last_rebalance_date == jan.isoformat()
    assert state_feb.last_rebalance_date == feb.isoformat()


def test_circuit_breaker_respected(tmp_path: Path) -> None:
    provider = _DataProviderStub()
    d1 = date(2024, 1, 2)
    provider.set_day(d1, prices={"AAA": {"open": 100.0, "close": 101.0}}, candidates=["AAA"])
    engine, _ = _engine(tmp_path, provider)
    engine.risk_controller.circuit_breaker_active = True

    state = engine.run_daily(d1)
    assert state.circuit_breaker_active is True
    assert not state.open_positions


def test_no_crash_on_missing_data(tmp_path: Path) -> None:
    provider = _DataProviderStub()
    d1 = date(2024, 1, 2)
    engine, _ = _engine(tmp_path, provider)

    state = engine.run_daily(d1)
    assert len(state.equity_curve) == 1
    assert state.equity_curve[-1]["equity"] == 500_000.0
