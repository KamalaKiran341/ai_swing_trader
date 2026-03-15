"""Composite stock scoring engine using technical and fundamental factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from core.analytics.indicators import momentum, rsi, sma
from core.config.config_loader import StrategyConfig
from core.data.price_loader import PriceDataLoader


@dataclass(frozen=True)
class StockScore:
    """Final score payload for a symbol."""

    symbol: str
    score: float
    technical_score: float
    fundamental_score: float


class FundamentalDataProvider(Protocol):
    """Protocol for fundamental data retrieval."""

    def get_fundamentals(self, symbol: str, as_of_date: pd.Timestamp) -> dict:
        """Return symbol fundamentals with expected keys.

        Expected keys:
        - earnings_growth
        - roe
        - dividend_yield
        """


class ScoringEngine:
    """Compute weighted stock scores from technical and fundamental inputs."""

    _TECHNICAL_WEIGHT = 0.60
    _FUNDAMENTAL_WEIGHT = 0.40
    _MOMENTUM_WINDOW = 90
    _RSI_WINDOW = 14
    _SMA_WINDOW = 200
    _LOOKBACK = 260
    _FUNDAMENTAL_KEYS = ("earnings_growth", "roe", "dividend_yield")

    def __init__(
        self,
        price_loader: PriceDataLoader,
        fundamentals: FundamentalDataProvider,
        config: StrategyConfig,
    ) -> None:
        """Initialize the scoring engine."""
        self._price_loader = price_loader
        self._fundamentals = fundamentals
        self._config = config

    def score_universe(self, symbols: list[str], as_of_date: pd.Timestamp) -> list[StockScore]:
        """Score symbols and return deterministic ranked results.

        Symbols failing the liquidity and cyclical-sector hooks are excluded.
        Missing fundamentals are handled safely with neutral scoring.
        """
        as_of_ts = pd.Timestamp(as_of_date)
        technical_rows: list[dict[str, float | str]] = []

        for symbol in symbols:
            if not self._passes_liquidity_filter(symbol, as_of_ts):
                continue
            if self._is_excluded_cyclical_sector(symbol):
                continue

            row = self._compute_technical_inputs(symbol, as_of_ts)
            if row is not None:
                technical_rows.append(row)

        if not technical_rows:
            return []

        technical_df = pd.DataFrame(technical_rows).set_index("symbol", drop=True)
        technical_df["momentum_score"] = _percentile_score(technical_df["momentum"])
        technical_df["rsi_score"] = _rsi_health_score(technical_df["rsi"])
        technical_df["trend_score"] = technical_df["bullish_trend"] * 100.0
        technical_df["technical_score"] = (
            technical_df["momentum_score"] + technical_df["rsi_score"] + technical_df["trend_score"]
        ) / 3.0

        fundamentals_df = self._collect_fundamentals(technical_df.index.tolist(), as_of_ts)
        combined = technical_df.join(fundamentals_df, how="left")
        combined["fundamental_score"] = combined.loc[:, list(self._FUNDAMENTAL_KEYS)].mean(axis=1)
        combined["score"] = (
            self._TECHNICAL_WEIGHT * combined["technical_score"]
            + self._FUNDAMENTAL_WEIGHT * combined["fundamental_score"]
        ).clip(lower=0.0, upper=100.0)

        filtered = combined[combined["score"] >= float(self._config.score_threshold)]
        ranked = (
            filtered.assign(symbol_key=filtered.index)
            .sort_values(by=["score", "symbol_key"], ascending=[False, True], kind="mergesort")
            .drop(columns=["symbol_key"])
        )

        return [
            StockScore(
                symbol=str(symbol),
                score=float(row["score"]),
                technical_score=float(row["technical_score"]),
                fundamental_score=float(row["fundamental_score"]),
            )
            for symbol, row in ranked.iterrows()
        ]

    def _compute_technical_inputs(
        self, symbol: str, as_of_date: pd.Timestamp
    ) -> dict[str, float | str] | None:
        try:
            history = self._price_loader.get_history(
                symbol, end_date=as_of_date, lookback=self._LOOKBACK
            )
        except (FileNotFoundError, ValueError):
            return None

        if len(history) < self._SMA_WINDOW:
            return None

        close = history["close"]
        if close.isna().any():
            return None

        momentum_value = momentum(close, window=self._MOMENTUM_WINDOW).iloc[-1]
        rsi_value = rsi(close, window=self._RSI_WINDOW).iloc[-1]
        sma_value = sma(close, window=self._SMA_WINDOW).iloc[-1]
        if pd.isna(momentum_value) or pd.isna(rsi_value) or pd.isna(sma_value):
            return None

        bullish_trend = 1.0 if float(close.iloc[-1]) > float(sma_value) else 0.0
        return {
            "symbol": symbol,
            "momentum": float(momentum_value),
            "rsi": float(rsi_value),
            "bullish_trend": bullish_trend,
        }

    def _collect_fundamentals(self, symbols: list[str], as_of_date: pd.Timestamp) -> pd.DataFrame:
        raw = pd.DataFrame(
            index=pd.Index(symbols, name="symbol"), columns=self._FUNDAMENTAL_KEYS, dtype=float
        )

        for symbol in symbols:
            try:
                values = self._fundamentals.get_fundamentals(symbol, as_of_date)
            except Exception:
                values = {}

            for key in self._FUNDAMENTAL_KEYS:
                value = values.get(key)
                raw.at[symbol, key] = _to_float_or_nan(value)

        scored = pd.DataFrame(index=raw.index, dtype=float)
        for key in self._FUNDAMENTAL_KEYS:
            scored[key] = _percentile_score(raw[key]).fillna(50.0)
        return scored

    def _passes_liquidity_filter(self, symbol: str, as_of_date: pd.Timestamp) -> bool:
        """Placeholder hook for liquidity filtering."""
        _ = (symbol, as_of_date)
        return True

    def _is_excluded_cyclical_sector(self, symbol: str) -> bool:
        """Placeholder hook for cyclical-sector exclusion."""
        _ = symbol
        return False


def _percentile_score(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", pct=True)
    return ranks * 100.0


def _rsi_health_score(values: pd.Series) -> pd.Series:
    lower, upper = 45.0, 65.0
    distance = np.where(
        values < lower, lower - values, np.where(values > upper, values - upper, 0.0)
    )
    return pd.Series(100.0 * np.exp(-((distance / 20.0) ** 2)), index=values.index)


def _to_float_or_nan(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
