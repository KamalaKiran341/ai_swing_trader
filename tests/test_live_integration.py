from __future__ import annotations

from datetime import date
from pathlib import Path

from core.strategy.regime_detector import Regime
from core.strategy.scoring_engine import StockScore
from src.automation.daily_runner import DailyStrategyRunner, RunnerConfig
from src.execution.order_simulator import OrderSimulator
from src.live.paper_trading_engine import (
    BacktestComponentsBundle,
    JsonFileStateStore,
    PaperTradingConfig,
    PaperTradingEngine,
)
from src.live.zerodha_data_provider import ZerodhaLiveDataProvider
from src.risk.portfolio_risk import PortfolioRiskController
from src.risk.trailing_stop import ATRTrailingStop


class _Calendar:
    def is_holiday(self, day: date) -> bool:
        _ = day
        return False


class _RegimeDetector:
    def get_regime(self, as_of_date):  # type: ignore[no-untyped-def]
        _ = as_of_date
        return Regime.BULL


class _ScoringEngine:
    def score_universe(self, symbols, as_of_date):  # type: ignore[no-untyped-def]
        _ = as_of_date
        return [
            StockScore(symbol=s, score=100.0 - i, technical_score=90.0, fundamental_score=80.0)
            for i, s in enumerate(sorted(symbols))
        ]


class _Adapter:
    def get_watchlist_symbols(self):  # type: ignore[no-untyped-def]
        return ["INFY", "TCS"]

    def get_ltp(self, symbols):  # type: ignore[no-untyped-def]
        return {s: 100.0 + i for i, s in enumerate(symbols)}


def test_provider_with_paper_engine_and_runner(tmp_path: Path) -> None:
    provider = ZerodhaLiveDataProvider(_Adapter(), cache_enabled=False)
    engine = PaperTradingEngine(
        data_provider=provider,
        backtest_components_bundle=BacktestComponentsBundle(
            regime_detector=_RegimeDetector(),
            scoring_engine=_ScoringEngine(),
            exit_engine=ATRTrailingStop(2.0),
        ),
        order_simulator=OrderSimulator(slippage_bps=0.0),
        risk_controller=PortfolioRiskController(max_positions=2),
        config=PaperTradingConfig(starting_capital=500_000.0, max_positions=2),
        state_store=JsonFileStateStore(str(tmp_path / "state.json")),
    )
    runner = DailyStrategyRunner(engine, _Calendar(), RunnerConfig(retry_backoff_seconds=0.0))
    status = runner.run(date(2024, 1, 8))

    assert status.success is True
    assert status.skipped is False
    assert status.summary is not None
    assert status.summary["open_positions_count"] >= 0
