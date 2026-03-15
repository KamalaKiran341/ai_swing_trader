"""Daily paper-trading engine with persisted state and deterministic behavior."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperTradingConfig:
    """Runtime configuration for paper trading."""

    starting_capital: float = 500_000.0
    execution_mode: str = "next_open"
    rebalance_frequency: str = "monthly"
    regime_frequency: str = "weekly"
    max_positions: int = 10


@dataclass
class PaperPortfolioState:
    """Persisted paper-portfolio state (JSON serializable)."""

    current_capital: float
    open_positions: dict[str, dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    last_rebalance_date: str | None
    circuit_breaker_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_capital": float(self.current_capital),
            "open_positions": self.open_positions,
            "equity_curve": self.equity_curve,
            "last_rebalance_date": self.last_rebalance_date,
            "circuit_breaker_active": bool(self.circuit_breaker_active),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PaperPortfolioState":
        return cls(
            current_capital=float(payload.get("current_capital", 0.0)),
            open_positions=dict(payload.get("open_positions", {})),
            equity_curve=list(payload.get("equity_curve", [])),
            last_rebalance_date=payload.get("last_rebalance_date"),
            circuit_breaker_active=bool(payload.get("circuit_breaker_active", False)),
        )


class StateStore(Protocol):
    """Persistence interface for paper-trading state."""

    def load_state(self) -> PaperPortfolioState | None:
        """Return stored state or ``None`` if no state exists."""

    def save_state(self, state: PaperPortfolioState) -> None:
        """Persist updated state."""


class JsonFileStateStore:
    """JSON-file implementation of ``StateStore``."""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)

    def load_state(self) -> PaperPortfolioState | None:
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return PaperPortfolioState.from_dict(payload)

    def save_state(self, state: PaperPortfolioState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )


@dataclass(frozen=True)
class BacktestComponentsBundle:
    """Bundle of strategy components reused in paper mode."""

    regime_detector: Any
    scoring_engine: Any
    exit_engine: Any


class PaperTradingEngine:
    """Forward-running paper trading orchestrator."""

    def __init__(
        self,
        data_provider: Any,
        backtest_components_bundle: BacktestComponentsBundle,
        order_simulator: Any,
        risk_controller: Any,
        config: PaperTradingConfig,
        state_store: StateStore,
    ) -> None:
        self.data_provider = data_provider
        self.components = backtest_components_bundle
        self.order_simulator = order_simulator
        self.risk_controller = risk_controller
        self.config = config
        self.state_store = state_store

    def run_daily(self, current_date: date | pd.Timestamp) -> PaperPortfolioState:
        """Run one deterministic daily paper-trading cycle."""
        day = pd.Timestamp(current_date).date()
        state = self.state_store.load_state() or self._initial_state()

        if self._already_processed(state, day):
            LOGGER.info("event=daily_skip reason=idempotent date=%s", day.isoformat())
            return state

        prices = self._get_prices(day)
        if not prices:
            self._record_equity_snapshot(state, day, prices)
            self.state_store.save_state(state)
            LOGGER.info("event=daily_skip reason=no_market_data date=%s", day.isoformat())
            return state

        regime = self._get_regime(day)
        self._update_trailing_stops(state, prices)
        self._process_exits(state, prices, day)

        if self._is_rebalance_day(state, day):
            if not self.risk_controller.is_trading_allowed():
                state.circuit_breaker_active = True
                LOGGER.info("event=circuit_breaker_active date=%s", day.isoformat())
            else:
                state.circuit_breaker_active = False
                self._process_entries(state, prices, day)
            state.last_rebalance_date = day.isoformat()
            LOGGER.info("event=rebalance date=%s regime=%s", day.isoformat(), regime)

        self._record_equity_snapshot(state, day, prices)
        self.state_store.save_state(state)
        LOGGER.info(
            "event=daily_summary date=%s equity=%.2f cash=%.2f open_positions=%s",
            day.isoformat(),
            state.equity_curve[-1]["equity"],
            state.current_capital,
            len(state.open_positions),
        )
        return state

    def _initial_state(self) -> PaperPortfolioState:
        return PaperPortfolioState(
            current_capital=float(self.config.starting_capital),
            open_positions={},
            equity_curve=[],
            last_rebalance_date=None,
            circuit_breaker_active=False,
        )

    def _already_processed(self, state: PaperPortfolioState, day: date) -> bool:
        if not state.equity_curve:
            return False
        return str(state.equity_curve[-1].get("date")) == day.isoformat()

    def _get_prices(self, day: date) -> dict[str, dict[str, float]]:
        try:
            raw = self.data_provider.get_prices(day)
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw

    def _get_regime(self, day: date) -> str:
        if self.config.regime_frequency == "weekly" and pd.Timestamp(day).weekday() not in {0, 4}:
            return "UNCHANGED"
        try:
            regime = self.components.regime_detector.get_regime(pd.Timestamp(day))
            LOGGER.info("event=regime_check date=%s regime=%s", day.isoformat(), regime)
            return str(regime)
        except Exception:
            return "UNKNOWN"

    def _update_trailing_stops(
        self, state: PaperPortfolioState, prices: dict[str, dict[str, float]]
    ) -> None:
        for symbol, position in list(state.open_positions.items()):
            price_data = prices.get(symbol)
            if not price_data:
                continue
            close = float(price_data.get("close", 0.0))
            if close <= 0:
                continue
            atr = self._get_atr(symbol, pd.Timestamp(price_data.get("date", pd.Timestamp.today())))
            if atr is None:
                atr = max(close * 0.02, 0.01)
            updated = self.components.exit_engine.update_stop(
                float(position["stop_price"]), close, float(atr)
            )
            position["stop_price"] = float(updated)

    def _process_exits(
        self,
        state: PaperPortfolioState,
        prices: dict[str, dict[str, float]],
        day: date,
    ) -> None:
        for symbol in list(state.open_positions):
            position = state.open_positions[symbol]
            px = prices.get(symbol)
            if not px:
                continue
            close = float(px.get("close", 0.0))
            if close <= 0:
                continue
            if self.components.exit_engine.check_exit(close, float(position["stop_price"])):
                qty = int(position["quantity"])
                fill = self.order_simulator.simulate_order(symbol, "SELL", qty, close)
                state.current_capital += float(fill.net_amount)
                LOGGER.info(
                    "event=trade side=SELL date=%s symbol=%s qty=%s price=%.4f",
                    day.isoformat(),
                    symbol,
                    qty,
                    fill.price,
                )
                del state.open_positions[symbol]

    def _process_entries(
        self,
        state: PaperPortfolioState,
        prices: dict[str, dict[str, float]],
        day: date,
    ) -> None:
        candidates = self._candidate_symbols(day, prices)
        if not candidates:
            return

        scored = self.components.scoring_engine.score_universe(candidates, pd.Timestamp(day))
        ranked = [item.symbol for item in scored]
        available_slots = max(self.config.max_positions - len(state.open_positions), 0)
        selected = [s for s in ranked if s not in state.open_positions][:available_slots]
        if not selected:
            return

        budget = self.risk_controller.compute_equal_position_size(
            state.current_capital, len(selected)
        )
        if budget <= 0:
            return

        for symbol in selected:
            px = prices.get(symbol)
            if not px:
                continue
            order_price = self._execution_price(px)
            if order_price <= 0:
                continue
            quantity = int(budget // order_price)
            if quantity <= 0:
                continue
            fill = self.order_simulator.simulate_order(symbol, "BUY", quantity, order_price)
            new_cash = state.current_capital + float(fill.net_amount)
            if new_cash < 0:
                continue
            state.current_capital = new_cash
            atr = self._get_atr(symbol, pd.Timestamp(day))
            if atr is None:
                atr = max(fill.price * 0.02, 0.01)
            stop = self.components.exit_engine.initialize_stop(fill.price, float(atr))
            state.open_positions[symbol] = {
                "quantity": int(quantity),
                "avg_price": float(fill.price),
                "stop_price": float(stop),
                "entry_date": day.isoformat(),
            }
            LOGGER.info(
                "event=trade side=BUY date=%s symbol=%s qty=%s price=%.4f",
                day.isoformat(),
                symbol,
                quantity,
                fill.price,
            )

    def _candidate_symbols(self, day: date, prices: dict[str, dict[str, float]]) -> list[str]:
        try:
            symbols = self.data_provider.get_candidate_symbols(day)
        except Exception:
            symbols = []
        if not symbols:
            symbols = sorted(prices)
        return [s for s in symbols if s in prices]

    def _execution_price(self, price_row: dict[str, float]) -> float:
        if self.config.execution_mode == "next_open":
            value = float(price_row.get("open", price_row.get("close", 0.0)))
            return value
        return float(price_row.get("close", 0.0))

    def _is_rebalance_day(self, state: PaperPortfolioState, day: date) -> bool:
        if self.config.rebalance_frequency != "monthly":
            return True
        if state.last_rebalance_date is None:
            return True
        last = pd.Timestamp(state.last_rebalance_date).date()
        return (last.year, last.month) != (day.year, day.month)

    def _get_atr(self, symbol: str, as_of: pd.Timestamp) -> float | None:
        try:
            atr = self.data_provider.get_atr(symbol, as_of)
        except Exception:
            return None
        if atr is None:
            return None
        value = float(atr)
        if value <= 0 or pd.isna(value):
            return None
        return value

    def _record_equity_snapshot(
        self, state: PaperPortfolioState, day: date, prices: dict[str, dict[str, float]]
    ) -> None:
        equity = float(state.current_capital)
        for symbol, pos in state.open_positions.items():
            px = prices.get(symbol)
            if not px:
                continue
            close = float(px.get("close", 0.0))
            if close <= 0:
                continue
            equity += int(pos["quantity"]) * close
        state.equity_curve.append({"date": day.isoformat(), "equity": float(equity)})
        LOGGER.info("event=equity_snapshot date=%s equity=%.2f", day.isoformat(), equity)
