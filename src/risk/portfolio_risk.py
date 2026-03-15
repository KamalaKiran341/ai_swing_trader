"""Portfolio-level risk controls for capital allocation and trading eligibility."""

from __future__ import annotations

import math


class PortfolioRiskController:
    """Manage portfolio risk state for sizing, drawdown, and circuit breaking."""

    def __init__(
        self,
        max_positions: int,
        losing_month_tolerance: int = 2,
        circuit_breaker_enabled: bool = True,
    ) -> None:
        """Initialize risk controller with portfolio constraints."""
        if max_positions <= 0:
            raise ValueError("max_positions must be greater than 0.")
        if losing_month_tolerance < 0:
            raise ValueError("losing_month_tolerance must be non-negative.")

        self.max_positions = int(max_positions)
        self.losing_month_tolerance = int(losing_month_tolerance)
        self.circuit_breaker_enabled = bool(circuit_breaker_enabled)

        self.equity_peak: float = 0.0
        self.current_drawdown: float = 0.0
        self.losing_months_counter: int = 0
        self.circuit_breaker_active: bool = False

    def compute_equal_position_size(self, total_capital: float, num_positions: int) -> float:
        """Return equal-weight capital allocation per position.

        Returns 0.0 when capital is non-positive or position count is invalid.
        """
        if math.isnan(total_capital):
            raise ValueError("total_capital must be a valid number.")
        if total_capital <= 0 or num_positions <= 0:
            return 0.0

        effective_positions = min(int(num_positions), self.max_positions)
        if effective_positions <= 0:
            return 0.0
        return float(total_capital) / float(effective_positions)

    def update_drawdown(self, current_equity: float) -> float:
        """Update and return current drawdown percentage as a value in [0, 1]."""
        if math.isnan(current_equity):
            raise ValueError("current_equity must be a valid number.")
        if current_equity < 0:
            raise ValueError("current_equity cannot be negative.")

        equity = float(current_equity)
        if equity > self.equity_peak:
            self.equity_peak = equity

        if self.equity_peak <= 0:
            self.current_drawdown = 0.0
        else:
            drawdown = (self.equity_peak - equity) / self.equity_peak
            self.current_drawdown = max(0.0, drawdown)

        return self.current_drawdown

    def update_monthly_performance(self, monthly_return: float) -> None:
        """Update losing-months streak and trigger circuit breaker when breached."""
        if math.isnan(monthly_return):
            return

        if monthly_return < 0:
            self.losing_months_counter += 1
        else:
            self.losing_months_counter = 0

        if (
            self.circuit_breaker_enabled
            and self.losing_months_counter > self.losing_month_tolerance
        ):
            self.circuit_breaker_active = True

    def is_trading_allowed(self) -> bool:
        """Return whether new trades are currently allowed."""
        if not self.circuit_breaker_enabled:
            return True
        return not self.circuit_breaker_active

    def check_recovery(self, monthly_return: float) -> None:
        """Disable active breaker on positive recovery month."""
        if math.isnan(monthly_return):
            return

        if self.circuit_breaker_active and monthly_return > 0:
            self.losing_months_counter = 0
            self.circuit_breaker_active = False

    def reset_state(self) -> None:
        """Reset runtime risk state to clean initial values."""
        self.equity_peak = 0.0
        self.current_drawdown = 0.0
        self.losing_months_counter = 0
        self.circuit_breaker_active = False
