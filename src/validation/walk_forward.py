"""Walk-forward validation engine for out-of-sample robustness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for rolling walk-forward windows."""

    train_years: int = 3
    test_months: int = 6
    step_months: int = 6
    min_history_years: int = 4


@dataclass(frozen=True)
class WindowSpec:
    """Single train/test window boundaries."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardResult:
    """Result bundle for walk-forward execution."""

    window_results: list[dict[str, Any]]
    aggregate_metrics: dict[str, float]
    equity_curve_combined: pd.Series


class WalkForwardEngine:
    """Run deterministic rolling out-of-sample backtest validation."""

    def __init__(
        self,
        backtest_engine_factory: Callable[..., Any],
        full_price_data: dict[str, pd.DataFrame],
        full_fundamental_data: Any,
        config: WalkForwardConfig,
    ) -> None:
        self.backtest_engine_factory = backtest_engine_factory
        self.full_price_data = full_price_data
        self.full_fundamental_data = full_fundamental_data
        self.config = config

    def generate_windows(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> Iterator[WindowSpec]:
        """Yield deterministic rolling windows without overlap leakage."""
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            return

        test_start = start + pd.DateOffset(years=self.config.min_history_years)
        while test_start <= end:
            train_end = test_start - pd.Timedelta(days=1)
            train_start = test_start - pd.DateOffset(years=self.config.train_years)
            test_end = min(
                (test_start + pd.DateOffset(months=self.config.test_months) - pd.Timedelta(days=1)),
                end,
            )
            if train_start < start:
                test_start = test_start + pd.DateOffset(months=self.config.step_months)
                continue
            if test_start > test_end:
                break

            yield WindowSpec(
                train_start=train_start.normalize(),
                train_end=train_end.normalize(),
                test_start=test_start.normalize(),
                test_end=test_end.normalize(),
            )
            test_start = test_start + pd.DateOffset(months=self.config.step_months)

    def run(self) -> WalkForwardResult:
        """Execute walk-forward windows and aggregate OOS performance."""
        start_date, end_date = self._data_bounds()
        if start_date is None or end_date is None:
            return WalkForwardResult([], self._empty_metrics(), pd.Series(dtype=float))

        window_results: list[dict[str, Any]] = []
        combined_curves: list[pd.Series] = []

        for window in self.generate_windows(start_date, end_date):
            train_prices = self._slice_price_data(window.train_start, window.train_end)
            test_prices = self._slice_price_data(window.test_start, window.test_end)
            if not test_prices:
                continue

            train_fund = self._slice_fundamental_data(
                self.full_fundamental_data, window.train_start, window.train_end
            )
            test_fund = self._slice_fundamental_data(
                self.full_fundamental_data, window.test_start, window.test_end
            )

            engine = self._build_engine(
                train_price_data=train_prices,
                test_price_data=test_prices,
                train_fundamental_data=train_fund,
                test_fundamental_data=test_fund,
                window=window,
            )
            result = engine.run_backtest()
            equity_curve = pd.Series(result.equity_curve, dtype=float).sort_index()
            if equity_curve.empty:
                continue

            window_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)
            cagr = float(result.performance_metrics.get("cagr", 0.0))

            window_results.append(
                {
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "window_return": window_return,
                    "cagr": cagr,
                    "metrics": result.performance_metrics,
                }
            )
            combined_curves.append(equity_curve)

        combined = self._combine_equity_curves(combined_curves)
        aggregate = self._aggregate_metrics(window_results)
        return WalkForwardResult(
            window_results=window_results,
            aggregate_metrics=aggregate,
            equity_curve_combined=combined,
        )

    def _build_engine(
        self,
        train_price_data: dict[str, pd.DataFrame],
        test_price_data: dict[str, pd.DataFrame],
        train_fundamental_data: Any,
        test_fundamental_data: Any,
        window: WindowSpec,
    ) -> Any:
        return self.backtest_engine_factory(
            train_price_data=train_price_data,
            test_price_data=test_price_data,
            train_fundamental_data=train_fundamental_data,
            test_fundamental_data=test_fundamental_data,
            window=window,
        )

    def _data_bounds(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        min_date: pd.Timestamp | None = None
        max_date: pd.Timestamp | None = None
        for frame in self.full_price_data.values():
            if frame.empty:
                continue
            idx = (
                frame.index
                if isinstance(frame.index, pd.DatetimeIndex)
                else pd.to_datetime(frame["date"])
            )
            frame_min = pd.Timestamp(idx.min()).normalize()
            frame_max = pd.Timestamp(idx.max()).normalize()
            min_date = frame_min if min_date is None else min(min_date, frame_min)
            max_date = frame_max if max_date is None else max(max_date, frame_max)
        return min_date, max_date

    def _slice_price_data(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        sliced: dict[str, pd.DataFrame] = {}
        for symbol, frame in self.full_price_data.items():
            index = (
                frame.index
                if isinstance(frame.index, pd.DatetimeIndex)
                else pd.to_datetime(frame["date"])
            )
            mask = (index >= start_date) & (index <= end_date)
            sub = frame.loc[mask]
            if not sub.empty:
                sliced[symbol] = sub.copy()
        return sliced

    def _slice_fundamental_data(
        self, data: Any, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> Any:
        if isinstance(data, pd.DataFrame):
            index = (
                data.index
                if isinstance(data.index, pd.DatetimeIndex)
                else pd.to_datetime(data["date"])
            )
            mask = (index >= start_date) & (index <= end_date)
            return data.loc[mask].copy()
        if isinstance(data, dict):
            out: dict[Any, Any] = {}
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    index = (
                        value.index
                        if isinstance(value.index, pd.DatetimeIndex)
                        else pd.to_datetime(value["date"])
                    )
                    mask = (index >= start_date) & (index <= end_date)
                    subset = value.loc[mask]
                    if not subset.empty:
                        out[key] = subset.copy()
                else:
                    out[key] = value
            return out
        return data

    def _combine_equity_curves(self, curves: list[pd.Series]) -> pd.Series:
        if not curves:
            return pd.Series(dtype=float)
        combined = pd.concat(curves).sort_index()
        return combined[~combined.index.duplicated(keep="last")]

    def _aggregate_metrics(self, window_results: list[dict[str, Any]]) -> dict[str, float]:
        if not window_results:
            return self._empty_metrics()

        window_returns = np.array(
            [float(item["window_return"]) for item in window_results], dtype=float
        )
        cagr_values = np.array([float(item["cagr"]) for item in window_results], dtype=float)
        profitable = float(np.sum(window_returns > 0))
        total = float(len(window_returns))
        stability = profitable / total if total > 0 else 0.0

        return {
            "median_cagr": float(np.median(cagr_values)),
            "worst_window_return": float(np.min(window_returns)),
            "best_window_return": float(np.max(window_returns)),
            "win_rate_windows": float(stability),
            "stability_score": float(stability),
        }

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "median_cagr": 0.0,
            "worst_window_return": 0.0,
            "best_window_return": 0.0,
            "win_rate_windows": 0.0,
            "stability_score": 0.0,
        }
