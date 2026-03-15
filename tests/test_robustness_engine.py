from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.validation.robustness_engine import RobustnessConfig, RobustnessEngine, RobustnessResult


@dataclass(frozen=True)
class _BaseConfig:
    score_threshold: int = 70
    atr_multiple: float = 2.0
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class _BacktestResult:
    equity_curve: pd.Series
    performance_metrics: dict[str, float]


class _StubBacktestEngine:
    def __init__(self, cagr: float, drawdown: float, seed: int) -> None:
        self._cagr = cagr
        self._drawdown = drawdown
        self._seed = seed

    def run_backtest(self) -> _BacktestResult:
        rng = np.random.default_rng(self._seed)
        returns = rng.normal(loc=0.0004 + (self._cagr / 252.0), scale=0.002, size=252)
        curve = pd.Series(
            np.cumprod(1.0 + returns), index=pd.bdate_range("2024-01-01", periods=252)
        )
        return _BacktestResult(
            equity_curve=curve,
            performance_metrics={"cagr": self._cagr, "max_drawdown": self._drawdown},
        )


class _Factory:
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def __call__(self, **kwargs: Any) -> _StubBacktestEngine:
        score = float(kwargs["score_threshold"])
        atr = float(kwargs["atr_multiple"])
        slippage = float(kwargs["slippage_bps"])
        cost = float(kwargs["cost_multiplier"])
        self.calls.append(
            {
                "score_threshold": score,
                "atr_multiple": atr,
                "slippage_bps": slippage,
                "cost_multiplier": cost,
            }
        )
        cagr = (
            0.20
            - 0.001 * abs(score - 70.0)
            - 0.02 * abs(atr - 2.0)
            - 0.003 * slippage
            - 0.05 * (cost - 1.0)
        )
        drawdown = 0.10 + 0.01 * abs(atr - 2.0) + 0.005 * (cost - 1.0) + 0.001 * slippage
        seed = int(score * 10 + atr * 100 + slippage * 5 + cost * 100)
        return _StubBacktestEngine(cagr=cagr, drawdown=drawdown, seed=seed)


def _price_data() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    close = np.linspace(100.0, 160.0, len(dates))
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


def _robustness_config() -> RobustnessConfig:
    return RobustnessConfig(
        score_threshold_range=[60, 70, 80],
        atr_multiplier_range=[1.5, 2.0, 2.5],
        slippage_bps_range=[5.0, 10.0, 20.0],
        cost_multiplier_range=[1.0, 1.5, 2.0],
        monte_carlo_runs=20,
        random_seed=7,
    )


def test_parameter_sweep_runs() -> None:
    factory = _Factory()
    engine = RobustnessEngine(factory, _BaseConfig(), _robustness_config(), _price_data(), {})
    results = engine.run_parameter_sensitivity()
    assert len(results) == 9


def test_cost_stress_runs() -> None:
    factory = _Factory()
    engine = RobustnessEngine(factory, _BaseConfig(), _robustness_config(), _price_data(), {})
    results = engine.run_cost_stress()
    assert len(results) == 3
    assert [r["cost_multiplier"] for r in results] == [1.0, 1.5, 2.0]


def test_slippage_stress_runs() -> None:
    factory = _Factory()
    engine = RobustnessEngine(factory, _BaseConfig(), _robustness_config(), _price_data(), {})
    results = engine.run_slippage_stress()
    assert len(results) == 3
    assert [r["slippage_bps"] for r in results] == [5.0, 10.0, 20.0]


def test_monte_carlo_produces_distribution() -> None:
    factory = _Factory()
    engine = RobustnessEngine(factory, _BaseConfig(), _robustness_config(), _price_data(), {})
    summary = engine.run_monte_carlo()
    assert set(summary) == {"median_cagr", "worst_case_drawdown", "return_5pct"}


def test_robustness_score_computed() -> None:
    factory = _Factory()
    engine = RobustnessEngine(factory, _BaseConfig(), _robustness_config(), _price_data(), {})
    result = engine.run()
    assert isinstance(result, RobustnessResult)
    assert 0.0 <= result.robustness_score <= 100.0


def test_deterministic_behavior() -> None:
    cfg = _robustness_config()
    result_a = RobustnessEngine(_Factory(), _BaseConfig(), cfg, _price_data(), {}).run()
    result_b = RobustnessEngine(_Factory(), _BaseConfig(), cfg, _price_data(), {}).run()
    assert result_a.parameter_results == result_b.parameter_results
    assert result_a.cost_results == result_b.cost_results
    assert result_a.slippage_results == result_b.slippage_results
    assert result_a.monte_carlo_summary == result_b.monte_carlo_summary
    assert result_a.robustness_score == result_b.robustness_score
