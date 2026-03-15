"""Deterministic backtest order simulation with Zerodha-equity cost modeling."""

from __future__ import annotations

from dataclasses import dataclass

VALID_SIDES = {"BUY", "SELL"}


@dataclass(frozen=True)
class FillResult:
    """Final simulated fill details and cost breakdown."""

    symbol: str
    side: str
    quantity: int
    price: float
    turnover: float
    brokerage: float
    stt: float
    exchange_txn: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    total_cost: float
    net_amount: float


def calculate_zerodha_costs(turnover: float, side: str) -> dict[str, float]:
    """Calculate approximate Zerodha equity-cash delivery charges."""
    if turnover <= 0:
        raise ValueError("turnover must be greater than 0.")
    normalized_side = side.upper()
    if normalized_side not in VALID_SIDES:
        raise ValueError("side must be one of {'BUY', 'SELL'}.")

    brokerage = 0.0  # Delivery brokerage is effectively zero.
    stt = turnover * 0.001 if normalized_side == "SELL" else 0.0
    exchange_txn = turnover * 0.0000345
    gst = 0.18 * (brokerage + exchange_txn)
    sebi_charges = turnover * (10.0 / 10_000_000.0)
    stamp_duty = turnover * 0.00015 if normalized_side == "BUY" else 0.0

    costs = {
        "brokerage": _round_money(brokerage),
        "stt": _round_money(stt),
        "exchange_txn": _round_money(exchange_txn),
        "gst": _round_money(gst),
        "sebi_charges": _round_money(sebi_charges),
        "stamp_duty": _round_money(stamp_duty),
    }
    costs["total_cost"] = _round_money(sum(costs.values()))
    return costs


def apply_slippage(price: float, side: str, slippage_bps: float = 5) -> float:
    """Apply deterministic basis-point slippage to market price."""
    if price <= 0:
        raise ValueError("price must be greater than 0.")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative.")

    normalized_side = side.upper()
    if normalized_side not in VALID_SIDES:
        raise ValueError("side must be one of {'BUY', 'SELL'}.")

    slip_ratio = slippage_bps / 10_000.0
    slipped = price * (1.0 + slip_ratio) if normalized_side == "BUY" else price * (1.0 - slip_ratio)
    return round(slipped, 4)


class OrderSimulator:
    """Simulate backtest fills with slippage and brokerage-cost application."""

    def __init__(self, slippage_bps: float = 5.0) -> None:
        """Create an order simulator instance."""
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative.")
        self._slippage_bps = float(slippage_bps)

    def simulate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        market_price: float,
    ) -> FillResult:
        """Simulate a single order fill and return full cost-adjusted details."""
        if quantity <= 0:
            raise ValueError("quantity must be greater than 0.")
        if market_price <= 0:
            raise ValueError("market_price must be greater than 0.")

        normalized_side = side.upper()
        if normalized_side not in VALID_SIDES:
            raise ValueError("side must be one of {'BUY', 'SELL'}.")

        fill_price = apply_slippage(market_price, normalized_side, self._slippage_bps)
        turnover = _round_money(fill_price * quantity)
        costs = calculate_zerodha_costs(turnover=turnover, side=normalized_side)
        total_cost = costs["total_cost"]

        if normalized_side == "BUY":
            net_amount = _round_money(-(turnover + total_cost))
        else:
            net_amount = _round_money(turnover - total_cost)

        return FillResult(
            symbol=symbol,
            side=normalized_side,
            quantity=quantity,
            price=fill_price,
            turnover=turnover,
            brokerage=costs["brokerage"],
            stt=costs["stt"],
            exchange_txn=costs["exchange_txn"],
            gst=costs["gst"],
            sebi_charges=costs["sebi_charges"],
            stamp_duty=costs["stamp_duty"],
            total_cost=total_cost,
            net_amount=net_amount,
        )


def _round_money(value: float) -> float:
    return round(float(value), 2)
