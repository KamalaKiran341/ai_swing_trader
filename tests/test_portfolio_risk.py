from __future__ import annotations

import math

import pytest

from src.risk.portfolio_risk import PortfolioRiskController


def test_equal_allocation_correct() -> None:
    controller = PortfolioRiskController(max_positions=10)
    size = controller.compute_equal_position_size(total_capital=100_000.0, num_positions=5)
    assert size == pytest.approx(20_000.0)


def test_drawdown_computed_correctly() -> None:
    controller = PortfolioRiskController(max_positions=10)
    assert controller.update_drawdown(100_000.0) == pytest.approx(0.0)
    assert controller.update_drawdown(90_000.0) == pytest.approx(0.10)
    assert controller.update_drawdown(120_000.0) == pytest.approx(0.0)


def test_losing_month_counter_works() -> None:
    controller = PortfolioRiskController(max_positions=10, losing_month_tolerance=2)
    controller.update_monthly_performance(-0.01)
    controller.update_monthly_performance(-0.02)
    assert controller.losing_months_counter == 2
    controller.update_monthly_performance(0.01)
    assert controller.losing_months_counter == 0


def test_circuit_breaker_triggers() -> None:
    controller = PortfolioRiskController(max_positions=10, losing_month_tolerance=2)
    controller.update_monthly_performance(-0.01)
    controller.update_monthly_performance(-0.02)
    controller.update_monthly_performance(-0.03)
    assert controller.circuit_breaker_active is True


def test_automatic_recovery_works() -> None:
    controller = PortfolioRiskController(max_positions=10, losing_month_tolerance=1)
    controller.update_monthly_performance(-0.01)
    controller.update_monthly_performance(-0.02)
    assert controller.circuit_breaker_active is True

    controller.check_recovery(0.03)
    assert controller.circuit_breaker_active is False
    assert controller.losing_months_counter == 0


def test_trading_gate_behaves_correctly() -> None:
    enabled = PortfolioRiskController(max_positions=10, circuit_breaker_enabled=True)
    disabled = PortfolioRiskController(max_positions=10, circuit_breaker_enabled=False)

    enabled.circuit_breaker_active = True
    disabled.circuit_breaker_active = True

    assert enabled.is_trading_allowed() is False
    assert disabled.is_trading_allowed() is True


def test_edge_cases_and_state_reset() -> None:
    controller = PortfolioRiskController(max_positions=10)

    assert controller.compute_equal_position_size(total_capital=0.0, num_positions=5) == 0.0
    assert controller.compute_equal_position_size(total_capital=100_000.0, num_positions=0) == 0.0

    controller.update_monthly_performance(float("nan"))
    assert controller.losing_months_counter == 0

    controller.check_recovery(float("nan"))
    assert controller.circuit_breaker_active is False

    with pytest.raises(ValueError, match="current_equity cannot be negative"):
        controller.update_drawdown(-1.0)

    with pytest.raises(ValueError, match="current_equity must be a valid number"):
        controller.update_drawdown(float("nan"))

    with pytest.raises(ValueError, match="total_capital must be a valid number"):
        controller.compute_equal_position_size(float("nan"), 5)

    controller.update_drawdown(100_000.0)
    controller.update_monthly_performance(-0.02)
    controller.circuit_breaker_active = True
    controller.reset_state()

    assert controller.equity_peak == 0.0
    assert controller.current_drawdown == 0.0
    assert controller.losing_months_counter == 0
    assert controller.circuit_breaker_active is False


def test_first_month_behavior() -> None:
    controller = PortfolioRiskController(max_positions=10)
    assert controller.losing_months_counter == 0
    assert controller.current_drawdown == 0.0
    assert controller.is_trading_allowed() is True
    assert math.isclose(controller.update_drawdown(0.0), 0.0)
