"""Live market-data provider wiring Zerodha adapter into paper workflow."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Protocol

import pandas as pd

LOGGER = logging.getLogger(__name__)


class SymbolMapper(Protocol):
    """Pluggable symbol-mapping interface."""

    def map(self, internal_symbol: str) -> str:
        """Map internal symbol to broker symbol."""


class DefaultSymbolMapper:
    """Default mapper for NSE cash symbols and ETFs."""

    def map(self, internal_symbol: str) -> str:
        clean = internal_symbol.strip().upper()
        if not clean:
            raise ValueError("internal_symbol must be non-empty.")
        # Default future-safe convention: NSE:<SYMBOL> for cash/ETF.
        return f"NSE:{clean}"


class ZerodhaLiveDataProvider:
    """Fetch normalized live prices using Zerodha adapter."""

    def __init__(
        self,
        zerodha_adapter: Any,
        symbol_mapper: SymbolMapper | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self.zerodha_adapter = zerodha_adapter
        self.symbol_mapper = symbol_mapper or DefaultSymbolMapper()
        self.cache_enabled = cache_enabled
        self._cache: dict[str, float] = {}
        self._reverse_map: dict[str, str] = {}

    def map_to_broker_symbol(self, internal_symbol: str) -> str:
        """Convert internal symbol namespace to Zerodha broker symbol."""
        return self.symbol_mapper.map(internal_symbol)

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch LTP for symbols with partial-safe normalization and caching."""
        unique_symbols = sorted(
            {s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()}
        )
        if not unique_symbols:
            return {}

        result: dict[str, float] = {}
        to_fetch: list[str] = []
        self._reverse_map = {}

        for symbol in unique_symbols:
            if self.cache_enabled and symbol in self._cache:
                result[symbol] = self._cache[symbol]
                continue
            try:
                broker_symbol = self.map_to_broker_symbol(symbol)
            except Exception as exc:
                LOGGER.warning("event=data_symbol_map_failure symbol=%s error=%s", symbol, str(exc))
                continue
            self._reverse_map[broker_symbol] = symbol
            to_fetch.append(broker_symbol)

        if to_fetch:
            try:
                ltp_data = self.zerodha_adapter.get_ltp(to_fetch)
            except Exception as exc:
                LOGGER.error("event=data_ltp_fetch_failure error=%s", str(exc))
                ltp_data = {}

            for broker_symbol, price in (ltp_data or {}).items():
                internal = self._reverse_map.get(broker_symbol)
                if internal is None:
                    continue
                try:
                    value = float(price)
                except (TypeError, ValueError):
                    continue
                result[internal] = value
                if self.cache_enabled:
                    self._cache[internal] = value

        return result

    def get_daily_bar(self, symbol: str, bar_date: date | pd.Timestamp) -> dict[str, float]:
        """Fallback daily OHLC retrieval when live LTP is unavailable."""
        ts = pd.Timestamp(bar_date)
        try:
            if hasattr(self.zerodha_adapter, "get_daily_bar"):
                payload = self.zerodha_adapter.get_daily_bar(self.map_to_broker_symbol(symbol), ts)
                if isinstance(payload, dict):
                    return self._normalize_bar(payload)
        except Exception as exc:
            LOGGER.warning(
                "event=data_daily_bar_failure symbol=%s date=%s error=%s",
                symbol,
                ts.date(),
                str(exc),
            )
        return {}

    def get_prices(self, day: date | pd.Timestamp) -> dict[str, dict[str, float]]:
        """PaperEngine-compatible market data payload for the given day."""
        _ = day
        symbols = self._discover_symbols()
        latest = self.get_latest_prices(symbols)
        payload: dict[str, dict[str, float]] = {}
        for symbol, price in latest.items():
            payload[symbol] = {"open": price, "close": price}
        return payload

    def get_candidate_symbols(self, day: date | pd.Timestamp) -> list[str]:
        """Return candidate symbols, defaulting to all discoverable symbols."""
        _ = day
        return self._discover_symbols()

    def get_atr(self, symbol: str, as_of: pd.Timestamp) -> float:
        """Optional ATR passthrough for paper engine (safe default)."""
        _ = (symbol, as_of)
        return 2.0

    def clear_cache(self) -> None:
        """Clear in-memory per-run price cache."""
        self._cache.clear()

    def _discover_symbols(self) -> list[str]:
        if hasattr(self.zerodha_adapter, "get_watchlist_symbols"):
            try:
                symbols = self.zerodha_adapter.get_watchlist_symbols()
                if isinstance(symbols, list):
                    return sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
            except Exception as exc:
                LOGGER.warning("event=data_watchlist_failure error=%s", str(exc))
        return []

    def _normalize_bar(self, payload: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in ("open", "high", "low", "close"):
            value = payload.get(key)
            if value is None:
                return {}
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                return {}
        return out
