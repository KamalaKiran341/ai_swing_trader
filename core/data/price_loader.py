"""Efficient parquet-backed loader for OHLCV price history."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
)


class PriceDataLoader:
    """Load and serve price history with a deterministic symbol-level LRU cache."""

    def __init__(self, base_path: str, use_adjusted: bool = True, cache_size: int = 100) -> None:
        """Initialize a new price loader.

        Args:
            base_path: Root data directory containing ``prices/{SYMBOL}.parquet``.
            use_adjusted: If ``True``, pricing methods use ``adj_close``.
            cache_size: Maximum number of symbol frames retained in memory.
        """
        if cache_size <= 0:
            raise ValueError("cache_size must be greater than 0.")

        self._base_path = Path(base_path)
        self._use_adjusted = use_adjusted
        self._cache_size = cache_size
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get_history(self, symbol: str, end_date: pd.Timestamp, lookback: int) -> pd.DataFrame:
        """Return the last ``lookback`` rows up to and including ``end_date``.

        The returned frame is indexed by date and sorted ascending.
        If insufficient history is available, all available rows are returned.
        """
        if lookback <= 0:
            raise ValueError("lookback must be greater than 0.")

        end_ts = _to_naive_timestamp(end_date)
        frame = self._load_symbol(symbol)
        window = frame.loc[:end_ts].tail(lookback)

        if self._use_adjusted:
            result = window.loc[:, ["open", "high", "low", "adj_close", "volume"]].rename(
                columns={"adj_close": "close"}
            )
            return result

        return window.loc[:, ["open", "high", "low", "close", "volume"]]

    def get_price(self, symbol: str, date: pd.Timestamp) -> float:
        """Return the exact closing price for ``date``.

        Forward fill is intentionally not supported.
        """
        ts = _to_naive_timestamp(date)
        frame = self._load_symbol(symbol)

        if ts not in frame.index:
            raise ValueError(f"Price for symbol '{symbol}' is missing on {ts.date()}.")

        price_column = "adj_close" if self._use_adjusted else "close"
        return float(frame.at[ts, price_column])

    def preload(self, symbols: list[str]) -> None:
        """Warm the cache for a list of symbols using LRU semantics."""
        for symbol in symbols:
            self._load_symbol(symbol)

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        normalized_symbol = _normalize_symbol(symbol)

        cached = self._cache.get(normalized_symbol)
        if cached is not None:
            self._cache.move_to_end(normalized_symbol)
            return cached

        frame = self._read_symbol_file(normalized_symbol)
        self._cache[normalized_symbol] = frame
        self._cache.move_to_end(normalized_symbol)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def _read_symbol_file(self, symbol: str) -> pd.DataFrame:
        file_path = self._base_path / "prices" / f"{symbol}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Price file not found for symbol '{symbol}': {file_path}")

        frame = pd.read_parquet(file_path)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Price file for symbol '{symbol}' is missing required columns: {missing_columns}"
            )

        price_frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        price_frame["date"] = pd.to_datetime(price_frame["date"], errors="coerce")
        if price_frame["date"].isna().any():
            raise ValueError(f"Price file for symbol '{symbol}' contains invalid date values.")

        if not price_frame["date"].is_monotonic_increasing:
            price_frame = price_frame.sort_values("date", kind="mergesort")

        if price_frame["date"].duplicated().any():
            raise ValueError(f"Price file for symbol '{symbol}' contains duplicate date values.")

        return price_frame.set_index("date", drop=True)


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must be a non-empty string.")
    return normalized


def _to_naive_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts
