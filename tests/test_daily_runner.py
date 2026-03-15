from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest
from src.automation.daily_runner import DailyStrategyRunner, RunnerConfig
from scripts import run_daily


class _Calendar:
    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays = holidays or set()

    def is_holiday(self, day: date) -> bool:
        return day in self.holidays


@dataclass
class _State:
    current_capital: float = 100_000.0
    open_positions: dict = None
    equity_curve: list = None
    today_trades: list = None
    last_regime: str = "BULL"

    def __post_init__(self) -> None:
        if self.open_positions is None:
            self.open_positions = {}
        if self.equity_curve is None:
            self.equity_curve = [{"date": "2024-01-02", "equity": 100_000.0}]
        if self.today_trades is None:
            self.today_trades = []


class _PaperEngine:
    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.state = _State()

    def run_daily(self, current_date: date) -> _State:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient failure")
        self.state.equity_curve.append(
            {"date": current_date.isoformat(), "equity": 100_000.0 + self.calls}
        )
        self.state.today_trades = [{"symbol": "AAA"}]
        return self.state


def _runner(engine: _PaperEngine, holidays: set[date] | None = None) -> DailyStrategyRunner:
    return DailyStrategyRunner(
        paper_engine=engine,
        calendar_provider=_Calendar(holidays=holidays),
        config=RunnerConfig(retry_backoff_seconds=0.0),
    )


def test_skips_weekends() -> None:
    runner = _runner(_PaperEngine())
    status = runner.run(date(2024, 1, 6))  # Saturday
    assert status.skipped is True
    assert status.success is True


def test_runs_on_trading_day() -> None:
    engine = _PaperEngine()
    runner = _runner(engine)
    status = runner.run(date(2024, 1, 8))  # Monday
    assert status.success is True
    assert status.skipped is False
    assert engine.calls == 1


def test_retry_works(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _PaperEngine(fail_times=1)
    runner = _runner(engine)
    monkeypatch.setattr("src.automation.daily_runner.time.sleep", lambda _: None)
    status = runner.run_with_retry(date(2024, 1, 8), max_retries=2)
    assert status.success is True
    assert status.attempts == 2


def test_summary_generated() -> None:
    runner = _runner(_PaperEngine())
    summary = runner.generate_daily_summary(
        _State(
            current_capital=90_000.0,
            open_positions={"AAA": {"quantity": 10}},
            equity_curve=[
                {"date": "2024-01-01", "equity": 100_000.0},
                {"date": "2024-01-02", "equity": 95_000.0},
            ],
            today_trades=[{"symbol": "AAA"}],
            last_regime="BULL",
        )
    )
    assert summary["current_equity"] == 95_000.0
    assert summary["open_positions_count"] == 1
    assert summary["today_trades"] == 1
    assert summary["drawdown"] > 0.0
    assert summary["regime"] == "BULL"


def test_cli_returns_proper_code(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _PaperEngine()
    runner = _runner(engine)
    monkeypatch.setattr(run_daily, "create_runner", lambda: runner)
    code = run_daily.main(["--date", "2024-01-08", "--retries", "0"])
    assert code == 0


def test_failure_handled_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _PaperEngine(fail_times=5)
    runner = _runner(engine)
    monkeypatch.setattr(run_daily, "create_runner", lambda: runner)
    monkeypatch.setattr("src.automation.daily_runner.time.sleep", lambda _: None)
    code = run_daily.main(["--date", "2024-01-08", "--retries", "1"])
    assert code == 1
