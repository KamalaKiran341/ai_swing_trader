"""Scheduler-friendly daily runner for paper trading."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    """Operational settings for the daily strategy runner."""

    retry_backoff_seconds: float = 1.0


@dataclass(frozen=True)
class RunStatus:
    """Result of one daily-run invocation."""

    success: bool
    skipped: bool
    message: str
    run_date: str
    attempts: int
    summary: dict[str, Any] | None = None


class CalendarProvider(Protocol):
    """Trading calendar interface."""

    def is_holiday(self, day: date) -> bool:
        """Return True when exchange is closed for holiday."""


class DailyStrategyRunner:
    """Execute paper trading in a scheduler-safe and deterministic way."""

    def __init__(
        self,
        paper_engine: Any,
        calendar_provider: CalendarProvider,
        config: RunnerConfig,
    ) -> None:
        self.paper_engine = paper_engine
        self.calendar_provider = calendar_provider
        self.config = config

    def should_run(self, current_date: date | pd.Timestamp) -> bool:
        """Return whether strategy should run for ``current_date``."""
        day = pd.Timestamp(current_date).date()
        if day.weekday() >= 5:
            return False
        if self.calendar_provider.is_holiday(day):
            return False
        return True

    def run(self, current_date: date | pd.Timestamp) -> RunStatus:
        """Execute one daily run and return status."""
        day = pd.Timestamp(current_date).date()
        LOGGER.info("event=run_started date=%s", day.isoformat())

        if not self.should_run(day):
            LOGGER.info("event=run_skipped date=%s reason=non_trading_day", day.isoformat())
            return RunStatus(
                success=True,
                skipped=True,
                message="Skipped non-trading day.",
                run_date=day.isoformat(),
                attempts=1,
            )

        try:
            state = self.paper_engine.run_daily(day)
            summary = self.generate_daily_summary(state)
            LOGGER.info("event=run_success date=%s summary=%s", day.isoformat(), summary)
            return RunStatus(
                success=True,
                skipped=False,
                message="Run completed.",
                run_date=day.isoformat(),
                attempts=1,
                summary=summary,
            )
        except Exception as exc:
            LOGGER.exception("event=run_failure date=%s error=%s", day.isoformat(), str(exc))
            return RunStatus(
                success=False,
                skipped=False,
                message=f"Run failed: {exc}",
                run_date=day.isoformat(),
                attempts=1,
            )

    def run_with_retry(self, current_date: date | pd.Timestamp, max_retries: int = 2) -> RunStatus:
        """Execute run with retry on failures using exponential backoff."""
        total_attempts = max_retries + 1
        day = pd.Timestamp(current_date).date()

        for attempt in range(1, total_attempts + 1):
            status = self.run(day)
            if status.success or status.skipped:
                return RunStatus(
                    success=status.success,
                    skipped=status.skipped,
                    message=status.message,
                    run_date=status.run_date,
                    attempts=attempt,
                    summary=status.summary,
                )
            if attempt < total_attempts:
                wait_seconds = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "event=retry_scheduled date=%s attempt=%s wait_seconds=%.2f",
                    day.isoformat(),
                    attempt,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        LOGGER.error(
            "event=run_retries_exhausted date=%s attempts=%s", day.isoformat(), total_attempts
        )
        return RunStatus(
            success=False,
            skipped=False,
            message="Run failed after retries.",
            run_date=day.isoformat(),
            attempts=total_attempts,
        )

    def generate_daily_summary(self, state: Any) -> dict[str, Any]:
        """Generate structured summary from paper portfolio state."""
        equity_curve = list(getattr(state, "equity_curve", []))
        current_equity = (
            float(equity_curve[-1]["equity"])
            if equity_curve
            else float(getattr(state, "current_capital", 0.0))
        )
        open_positions = dict(getattr(state, "open_positions", {}))
        today_trades = list(getattr(state, "today_trades", []))
        regime = getattr(state, "last_regime", "UNKNOWN")

        drawdown = 0.0
        if equity_curve:
            values = pd.Series([float(x["equity"]) for x in equity_curve], dtype=float)
            peak = float(values.cummax().iloc[-1]) if not values.empty else 0.0
            if peak > 0:
                drawdown = float((peak - values.iloc[-1]) / peak)

        return {
            "current_equity": current_equity,
            "open_positions_count": len(open_positions),
            "today_trades": len(today_trades),
            "drawdown": drawdown,
            "regime": str(regime),
        }
