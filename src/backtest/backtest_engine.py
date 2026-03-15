"""Deterministic portfolio backtest engine orchestration."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.strategy.regime_detector import Regime
from core.strategy.scoring_engine import StockScore
from src.risk.trailing_stop import compute_atr

LOGGER = logging.getLogger(__name__)

LIQUID_ETF_SYMBOL = "LIQUID_ETF"
GOLD_ETF_SYMBOL = "GOLD_ETF"


@dataclass(frozen=True)
class TradeRecord:
    """Single executed trade event."""

    date: pd.Timestamp
    symbol: str
    side: str
    quantity: int
    price: float
    net_amount: float
    realized_pnl: float | None = None


@dataclass(frozen=True)
class BacktestResult:
    """Backtest output bundle."""

    equity_curve: pd.Series
    trades: list[TradeRecord]
    performance_metrics: dict[str, Any]


@dataclass
class _Position:
    symbol: str
    quantity: int
    avg_price: float
    stop_price: float
    entry_date: pd.Timestamp


class BacktestEngine:
    """Master daily loop engine for deterministic swing backtests."""

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        fundamental_data: Any,
        regime_detector: Any,
        scoring_engine: Any,
        exit_engine: Any,
        risk_controller: Any,
        order_simulator: Any,
        config: Any,
    ) -> None:
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.regime_detector = regime_detector
        self.scoring_engine = scoring_engine
        self.exit_engine = exit_engine
        self.risk_controller = risk_controller
        self.order_simulator = order_simulator
        self.config = config

        self._atr_cache: dict[str, pd.Series] = {}
        self._last_rebalance_month: tuple[int, int] | None = None
        self._last_regime: str | None = None
        self._last_month_end_equity: float | None = None

        self._positions: dict[str, _Position] = {}
        self._trades: list[TradeRecord] = []
        self._cash: float = float(getattr(config, "initial_capital", 1_000_000.0))

    def run_backtest(self) -> BacktestResult:
        """Run end-to-end backtest simulation and return result object."""
        dates = self._trading_dates()
        equity_curve_values: dict[pd.Timestamp, float] = {}
        drawdown_values: dict[pd.Timestamp, float] = {}

        for date in dates:
            regime = self._weekly_regime(date)

            self._update_trailing_stops(date)
            self._process_exits(date)

            if self.is_rebalance_day(date):
                LOGGER.info("event=rebalance date=%s", date.date())
                if str(regime) == str(Regime.BEAR):
                    self._rebalance_bear(date)
                else:
                    self._rebalance_bull(date)

            equity = self._portfolio_equity(date)
            drawdown = self.risk_controller.update_drawdown(equity)
            equity_curve_values[date] = equity
            drawdown_values[date] = drawdown

            if self._month_changed(date):
                self._update_monthly_risk_state(equity)

        equity_curve = pd.Series(equity_curve_values, dtype=float).sort_index()
        drawdown_series = pd.Series(drawdown_values, dtype=float).sort_index()
        metrics = self._compute_performance_metrics(equity_curve, drawdown_series, self._trades)
        return BacktestResult(
            equity_curve=equity_curve, trades=self._trades, performance_metrics=metrics
        )

    def is_rebalance_day(self, date: pd.Timestamp) -> bool:
        """Return True once per month on first encountered trading day."""
        month_key = (date.year, date.month)
        if self._last_rebalance_month == month_key:
            return False
        self._last_rebalance_month = month_key
        return True

    def _rebalance_bull(self, date: pd.Timestamp) -> None:
        if not self.risk_controller.is_trading_allowed():
            LOGGER.info("event=circuit_breaker_active date=%s", date.date())
            return

        symbols = self._candidate_universe(date)
        scored: list[StockScore] = self.scoring_engine.score_universe(symbols, date)
        selected = [
            item.symbol for item in scored[: int(getattr(self.config, "max_positions", 10))]
        ]

        for symbol in list(self._positions):
            if symbol not in selected:
                self._sell_position(symbol, date)

        if not selected:
            return

        total_equity = self._portfolio_equity(date)
        position_budget = self.risk_controller.compute_equal_position_size(
            total_equity, len(selected)
        )
        if position_budget <= 0:
            return

        for symbol in selected:
            if symbol in self._positions:
                continue
            self._buy_new_position(symbol, date, position_budget)

    def _rebalance_bear(self, date: pd.Timestamp) -> None:
        for symbol in list(self._positions):
            self._sell_position(symbol, date)

        if not self.risk_controller.is_trading_allowed():
            LOGGER.info("event=circuit_breaker_active date=%s", date.date())
            return

        total_equity = self._portfolio_equity(date)
        allocations = {LIQUID_ETF_SYMBOL: 0.60, GOLD_ETF_SYMBOL: 0.40}
        for symbol, weight in allocations.items():
            self._buy_new_position(symbol, date, total_equity * weight)

    def _weekly_regime(self, date: pd.Timestamp) -> Any:
        if date.weekday() == 0 or self._last_regime is None:
            current = self.regime_detector.get_regime(date)
            regime_text = str(current)
            if regime_text != self._last_regime:
                LOGGER.info("event=regime_switch date=%s regime=%s", date.date(), regime_text)
            self._last_regime = regime_text
            return current
        return self._last_regime

    def _update_trailing_stops(self, date: pd.Timestamp) -> None:
        for symbol, position in list(self._positions.items()):
            close = self._close_price(symbol, date)
            if close is None:
                continue
            atr_value = self._atr_value(symbol, date)
            if atr_value is None or math.isnan(atr_value):
                continue
            new_stop = self.exit_engine.update_stop(position.stop_price, close, atr_value)
            position.stop_price = float(new_stop)

    def _process_exits(self, date: pd.Timestamp) -> None:
        for symbol in list(self._positions):
            position = self._positions[symbol]
            close = self._close_price(symbol, date)
            if close is None:
                continue
            if self.exit_engine.check_exit(close, position.stop_price):
                self._sell_position(symbol, date)

    def _buy_new_position(self, symbol: str, date: pd.Timestamp, budget: float) -> None:
        price = self._close_price(symbol, date)
        if price is None or price <= 0:
            return
        qty = int(budget // price)
        if qty <= 0:
            return

        fill = self.order_simulator.simulate_order(symbol, "BUY", qty, price)
        cash_after = self._cash + fill.net_amount
        if cash_after < 0:
            return
        self._cash = cash_after

        atr_value = self._atr_value(symbol, date)
        if atr_value is None or math.isnan(atr_value):
            atr_value = max(price * 0.02, 0.01)
        stop = self.exit_engine.initialize_stop(fill.price, atr_value)
        self._positions[symbol] = _Position(
            symbol=symbol,
            quantity=qty,
            avg_price=fill.price,
            stop_price=float(stop),
            entry_date=date,
        )
        self._trades.append(
            TradeRecord(
                date=date,
                symbol=symbol,
                side="BUY",
                quantity=qty,
                price=fill.price,
                net_amount=fill.net_amount,
            )
        )
        LOGGER.info("event=trade side=BUY symbol=%s qty=%s date=%s", symbol, qty, date.date())

    def _sell_position(self, symbol: str, date: pd.Timestamp) -> None:
        position = self._positions.get(symbol)
        if position is None:
            return
        price = self._close_price(symbol, date)
        if price is None:
            return

        fill = self.order_simulator.simulate_order(symbol, "SELL", position.quantity, price)
        self._cash += fill.net_amount
        realized = (fill.price - position.avg_price) * position.quantity - fill.total_cost
        self._trades.append(
            TradeRecord(
                date=date,
                symbol=symbol,
                side="SELL",
                quantity=position.quantity,
                price=fill.price,
                net_amount=fill.net_amount,
                realized_pnl=float(realized),
            )
        )
        del self._positions[symbol]
        LOGGER.info(
            "event=trade side=SELL symbol=%s qty=%s date=%s", symbol, position.quantity, date.date()
        )

    def _candidate_universe(self, date: pd.Timestamp) -> list[str]:
        excluded = {LIQUID_ETF_SYMBOL, GOLD_ETF_SYMBOL}
        symbols: list[str] = []
        for symbol in sorted(self.price_data):
            if symbol in excluded:
                continue
            if self._close_price(symbol, date) is not None:
                symbols.append(symbol)
        return symbols

    def _trading_dates(self) -> list[pd.Timestamp]:
        dates = set()
        for frame in self.price_data.values():
            index = (
                frame.index
                if isinstance(frame.index, pd.DatetimeIndex)
                else pd.to_datetime(frame["date"])
            )
            for value in index:
                dates.add(pd.Timestamp(value).normalize())
        return sorted(dates)

    def _close_price(self, symbol: str, date: pd.Timestamp) -> float | None:
        frame = self.price_data.get(symbol)
        if frame is None:
            return None
        ts = pd.Timestamp(date).normalize()
        if ts not in frame.index:
            return None
        value = frame.at[ts, "close"]
        if pd.isna(value):
            return None
        return float(value)

    def _atr_value(self, symbol: str, date: pd.Timestamp) -> float | None:
        frame = self.price_data.get(symbol)
        if frame is None or frame.empty:
            return None
        if symbol not in self._atr_cache:
            if {"high", "low", "close"}.issubset(frame.columns):
                self._atr_cache[symbol] = compute_atr(frame, period=14)
            else:
                self._atr_cache[symbol] = pd.Series(np.nan, index=frame.index)
        series = self._atr_cache[symbol]
        ts = pd.Timestamp(date).normalize()
        if ts not in series.index:
            return None
        value = series.at[ts]
        if pd.isna(value):
            return None
        return float(value)

    def _portfolio_equity(self, date: pd.Timestamp) -> float:
        total = self._cash
        for symbol, position in self._positions.items():
            price = self._close_price(symbol, date)
            if price is None:
                continue
            total += position.quantity * price
        return float(max(total, 0.0))

    def _month_changed(self, date: pd.Timestamp) -> bool:
        next_day = date + pd.Timedelta(days=1)
        return date.month != next_day.month

    def _update_monthly_risk_state(self, month_end_equity: float) -> None:
        if self._last_month_end_equity is None or self._last_month_end_equity <= 0:
            self._last_month_end_equity = month_end_equity
            return
        monthly_return = (month_end_equity / self._last_month_end_equity) - 1.0
        self.risk_controller.update_monthly_performance(monthly_return)
        self.risk_controller.check_recovery(monthly_return)
        self._last_month_end_equity = month_end_equity

    def _compute_performance_metrics(
        self,
        equity_curve: pd.Series,
        drawdown_series: pd.Series,
        trades: list[TradeRecord],
    ) -> dict[str, Any]:
        if equity_curve.empty:
            return {
                "cagr": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "monthly_returns": {},
                "sharpe": 0.0,
            }

        start = float(equity_curve.iloc[0])
        end = float(equity_curve.iloc[-1])
        days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
        years = days / 365.25
        cagr = ((end / start) ** (1 / years) - 1.0) if start > 0 and years > 0 else 0.0

        realized = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        wins = [x for x in realized if x > 0]
        losses = [x for x in realized if x < 0]
        win_rate = len(wins) / len(realized) if realized else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        daily_returns = equity_curve.pct_change().dropna()
        sharpe = 0.0
        if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
            sharpe = float((daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252))

        monthly_returns = (
            equity_curve.resample("ME").last().pct_change().dropna().to_dict()
            if len(equity_curve) > 1
            else {}
        )
        monthly_returns = {k.strftime("%Y-%m-%d"): float(v) for k, v in monthly_returns.items()}

        return {
            "cagr": float(cagr),
            "max_drawdown": float(drawdown_series.max() if not drawdown_series.empty else 0.0),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "monthly_returns": monthly_returns,
            "sharpe": float(sharpe),
        }
