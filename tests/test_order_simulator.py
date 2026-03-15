from __future__ import annotations

import pytest

from src.execution.order_simulator import (
    OrderSimulator,
    apply_slippage,
    calculate_zerodha_costs,
)


def test_buy_slippage_increases_price() -> None:
    slipped = apply_slippage(price=100.0, side="BUY", slippage_bps=5)
    assert slipped == pytest.approx(100.05)


def test_sell_slippage_decreases_price() -> None:
    slipped = apply_slippage(price=100.0, side="SELL", slippage_bps=5)
    assert slipped == pytest.approx(99.95)


def test_zerodha_costs_computed() -> None:
    costs = calculate_zerodha_costs(turnover=100_000.0, side="SELL")
    assert costs["brokerage"] == 0.0
    assert costs["stt"] == pytest.approx(100.0)
    assert costs["exchange_txn"] > 0.0
    assert costs["gst"] > 0.0
    assert costs["sebi_charges"] > 0.0
    assert costs["stamp_duty"] == 0.0
    assert costs["total_cost"] == pytest.approx(
        costs["brokerage"]
        + costs["stt"]
        + costs["exchange_txn"]
        + costs["gst"]
        + costs["sebi_charges"]
        + costs["stamp_duty"]
    )


def test_net_amount_correct() -> None:
    simulator = OrderSimulator(slippage_bps=0.0)

    buy = simulator.simulate_order("INFY", "BUY", quantity=10, market_price=100.0)
    sell = simulator.simulate_order("INFY", "SELL", quantity=10, market_price=100.0)

    assert buy.turnover == 1_000.0
    assert buy.net_amount == pytest.approx(-(buy.turnover + buy.total_cost))
    assert sell.net_amount == pytest.approx(sell.turnover - sell.total_cost)


def test_invalid_inputs_raise_errors() -> None:
    simulator = OrderSimulator()

    with pytest.raises(ValueError, match="quantity must be greater than 0"):
        simulator.simulate_order("INFY", "BUY", quantity=0, market_price=100.0)
    with pytest.raises(ValueError, match="market_price must be greater than 0"):
        simulator.simulate_order("INFY", "BUY", quantity=1, market_price=0.0)
    with pytest.raises(ValueError, match="side must be one of"):
        simulator.simulate_order("INFY", "HOLD", quantity=1, market_price=100.0)
    with pytest.raises(ValueError, match="slippage_bps must be non-negative"):
        apply_slippage(price=100.0, side="BUY", slippage_bps=-1)
    with pytest.raises(ValueError, match="turnover must be greater than 0"):
        calculate_zerodha_costs(turnover=0.0, side="BUY")


def test_deterministic_output() -> None:
    simulator = OrderSimulator(slippage_bps=5.0)
    first = simulator.simulate_order("RELIANCE", "SELL", quantity=25, market_price=2500.0)
    second = simulator.simulate_order("RELIANCE", "SELL", quantity=25, market_price=2500.0)
    assert first == second
