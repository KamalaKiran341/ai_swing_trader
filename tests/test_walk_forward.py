from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.validation.walk_forward import WalkForwardConfig, WalkForwardEngine


@dataclass(frozen=True)
class _StubBacktestResult:
    equity_curve: pd.Series
    performance_metrics: dict[str, float]


class _StubBacktestEngine:
    def __init__(
        self, test_price_data: dict[str, pd.DataFrame], cagr: float, window_return: float
    ) -> None:
        self._test_price_data = test_price_data
        self._cagr = cagr
        self._window_return = window_return

    def run_backtest(self) -> _StubBacktestResult:
        first_symbol = sorted(self._test_price_data)[0]
        dates = self._test_price_data[first_symbol].index
        if len(dates) == 0:
            return _StubBacktestResult(
                equity_curve=pd.Series(dtype=float), performance_metrics={"cagr": 0.0}
            )
        start = 100.0
        end = start * (1.0 + self._window_return)
        values = np.linspace(start, end, len(dates))
        curve = pd.Series(values, index=dates, dtype=float)
        return _StubBacktestResult(equity_curve=curve, performance_metrics={"cagr": self._cagr})


class _Factory:
    def __init__(self) -> None:
        self.call_count = 0
        self.window_checks: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        self.engine_ids: list[int] = []

    def __call__(self, **kwargs: Any) -> _StubBacktestEngine:
        window = kwargs["window"]
        self.window_checks.append(
            (window.train_start, window.train_end, window.test_start, window.test_end)
        )
        self.call_count += 1

        # Alternate profitable/unprofitable windows to validate stability metrics.
        window_return = 0.05 if self.call_count % 2 == 1 else -0.02
        cagr = 0.12 if self.call_count % 2 == 1 else -0.03
        engine = _StubBacktestEngine(
            test_price_data=kwargs["test_price_data"],
            cagr=cagr,
            window_return=window_return,
        )
        self.engine_ids.append(id(engine))
        return engine


def _price_data() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    close = np.linspace(100.0, 200.0, len(dates))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(dates), 1_000_000.0),
        },
        index=dates,
    )
    return {"AAA": frame}


def test_window_generation_correct() -> None:
    config = WalkForwardConfig(train_years=3, test_months=6, step_months=6, min_history_years=4)
    engine = WalkForwardEngine(_Factory(), _price_data(), {}, config)
    windows = list(engine.generate_windows(pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31")))

    assert windows
    first = windows[0]
    assert first.train_start == pd.Timestamp("2016-01-01")
    assert first.train_end == pd.Timestamp("2018-12-31")
    assert first.test_start == pd.Timestamp("2019-01-01")
    assert first.test_end == pd.Timestamp("2019-06-30")


def test_no_overlap_leakage() -> None:
    config = WalkForwardConfig()
    engine = WalkForwardEngine(_Factory(), _price_data(), {}, config)
    windows = list(engine.generate_windows(pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31")))

    for window in windows:
        assert window.train_end < window.test_start
    for prev, curr in zip(windows, windows[1:]):
        assert prev.test_end < curr.test_start


def test_multiple_windows_produced() -> None:
    config = WalkForwardConfig()
    engine = WalkForwardEngine(_Factory(), _price_data(), {}, config)
    windows = list(engine.generate_windows(pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31")))
    assert len(windows) > 2


def test_aggregate_metrics_computed() -> None:
    factory = _Factory()
    engine = WalkForwardEngine(factory, _price_data(), {}, WalkForwardConfig())
    result = engine.run()

    assert result.window_results
    assert "median_cagr" in result.aggregate_metrics
    assert "best_window_return" in result.aggregate_metrics
    assert "worst_window_return" in result.aggregate_metrics
    assert not result.equity_curve_combined.empty
    assert factory.call_count == len(result.window_results)


def test_stability_score_correct() -> None:
    factory = _Factory()
    engine = WalkForwardEngine(factory, _price_data(), {}, WalkForwardConfig())
    result = engine.run()

    profitable = sum(1 for item in result.window_results if float(item["window_return"]) > 0)
    expected = profitable / len(result.window_results)
    assert result.aggregate_metrics["stability_score"] == expected
    assert 0.0 <= result.aggregate_metrics["stability_score"] <= 1.0


def test_deterministic_behavior() -> None:
    config = WalkForwardConfig()
    engine_a = WalkForwardEngine(_Factory(), _price_data(), {}, config)
    engine_b = WalkForwardEngine(_Factory(), _price_data(), {}, config)

    result_a = engine_a.run()
    result_b = engine_b.run()

    assert result_a.aggregate_metrics == result_b.aggregate_metrics
    assert result_a.window_results == result_b.window_results
    pd.testing.assert_series_equal(result_a.equity_curve_combined, result_b.equity_curve_combined)
