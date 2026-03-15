from __future__ import annotations

import pytest

from src.broker.zerodha_adapter import ZerodhaAdapter, ZerodhaConfig


class _MockKite:
    def __init__(self) -> None:
        self.access_token: str | None = None
        self.place_order_calls = 0
        self.ltp_calls = 0

    def set_access_token(self, token: str) -> None:
        self.access_token = token

    def ltp(self, symbols: list[str]) -> dict:
        self.ltp_calls += 1
        return {s: {"last_price": 100.0 + idx} for idx, s in enumerate(symbols)}

    def holdings(self) -> list[dict]:
        return [
            {
                "tradingsymbol": "INFY",
                "quantity": 10,
                "average_price": 1200.0,
                "last_price": 1250.0,
                "pnl": 500.0,
            }
        ]

    def positions(self) -> dict:
        return {
            "net": [
                {
                    "tradingsymbol": "TCS",
                    "quantity": 5,
                    "average_price": 3000.0,
                    "last_price": 3050.0,
                    "pnl": 250.0,
                }
            ]
        }

    def place_order(self, **kwargs) -> str:  # type: ignore[no-untyped-def]
        self.place_order_calls += 1
        return f"LIVE-{kwargs['tradingsymbol']}"


def _config(paper_mode: bool = True) -> ZerodhaConfig:
    return ZerodhaConfig(
        api_key="key",
        api_secret="secret",
        access_token="token",
        paper_mode=paper_mode,
    )


def test_paper_mode_does_not_hit_api() -> None:
    kite = _MockKite()
    adapter = ZerodhaAdapter(_config(paper_mode=True), kite_client=kite)

    result = adapter.place_order(symbol="INFY", side="BUY", quantity=10)

    assert result["mode"] == "paper"
    assert kite.place_order_calls == 0


def test_ltp_fetch_works_mocked() -> None:
    kite = _MockKite()
    adapter = ZerodhaAdapter(_config(), kite_client=kite)

    ltp = adapter.get_ltp(["NSE:INFY", "NSE:TCS"])

    assert ltp == {"NSE:INFY": 100.0, "NSE:TCS": 101.0}
    assert kite.ltp_calls == 1


def test_holdings_normalized() -> None:
    adapter = ZerodhaAdapter(_config(), kite_client=_MockKite())
    holdings = adapter.get_holdings()

    assert holdings == [
        {
            "symbol": "INFY",
            "quantity": 10,
            "average_price": 1200.0,
            "last_price": 1250.0,
            "pnl": 500.0,
        }
    ]


def test_order_validation_works() -> None:
    adapter = ZerodhaAdapter(_config(), kite_client=_MockKite())
    with pytest.raises(ValueError, match="symbol must be a non-empty string"):
        adapter.place_order(symbol="", side="BUY", quantity=1)
    with pytest.raises(ValueError, match="side must be one of"):
        adapter.place_order(symbol="INFY", side="HOLD", quantity=1)
    with pytest.raises(ValueError, match="quantity must be greater than 0"):
        adapter.place_order(symbol="INFY", side="BUY", quantity=0)


def test_live_flag_respected() -> None:
    kite = _MockKite()
    adapter = ZerodhaAdapter(_config(paper_mode=False), kite_client=kite)

    result = adapter.place_order(symbol="INFY", side="SELL", quantity=2)

    assert result["mode"] == "live"
    assert result["order_id"] == "LIVE-INFY"
    assert kite.place_order_calls == 1


def test_deterministic_behavior() -> None:
    kite = _MockKite()
    adapter = ZerodhaAdapter(_config(), kite_client=kite)

    one = adapter.place_order(symbol="INFY", side="BUY", quantity=5)
    two = adapter.place_order(symbol="INFY", side="BUY", quantity=5)

    assert one == two
