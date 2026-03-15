from __future__ import annotations

import pandas as pd

from src.live.zerodha_data_provider import DefaultSymbolMapper, ZerodhaLiveDataProvider


class _Adapter:
    def __init__(self) -> None:
        self.calls = 0
        self.should_fail = False

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("timeout")
        out: dict[str, float] = {}
        for idx, sym in enumerate(symbols):
            if "MISSING" in sym:
                continue
            out[sym] = 100.0 + idx
        return out

    def get_watchlist_symbols(self) -> list[str]:
        return ["INFY", "TCS", "MISSING"]

    def get_daily_bar(self, symbol: str, bar_date: pd.Timestamp) -> dict[str, float]:
        _ = (symbol, bar_date)
        return {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}


def test_symbol_mapping_works() -> None:
    mapper = DefaultSymbolMapper()
    assert mapper.map("infy") == "NSE:INFY"


def test_ltp_fetch_normalized() -> None:
    adapter = _Adapter()
    provider = ZerodhaLiveDataProvider(adapter, cache_enabled=False)
    prices = provider.get_latest_prices(["INFY", "TCS"])
    assert prices == {"INFY": 100.0, "TCS": 101.0}


def test_missing_symbols_handled() -> None:
    adapter = _Adapter()
    provider = ZerodhaLiveDataProvider(adapter, cache_enabled=False)
    prices = provider.get_latest_prices(["INFY", "MISSING"])
    assert prices == {"INFY": 100.0}


def test_cache_works() -> None:
    adapter = _Adapter()
    provider = ZerodhaLiveDataProvider(adapter, cache_enabled=True)
    first = provider.get_latest_prices(["INFY"])
    second = provider.get_latest_prices(["INFY"])
    assert first == second
    assert adapter.calls == 1


def test_adapter_errors_handled() -> None:
    adapter = _Adapter()
    adapter.should_fail = True
    provider = ZerodhaLiveDataProvider(adapter, cache_enabled=False)
    prices = provider.get_latest_prices(["INFY"])
    assert prices == {}


def test_deterministic_behavior() -> None:
    adapter = _Adapter()
    provider = ZerodhaLiveDataProvider(adapter, cache_enabled=False)
    one = provider.get_latest_prices(["TCS", "INFY"])
    two = provider.get_latest_prices(["INFY", "TCS"])
    assert one == two
