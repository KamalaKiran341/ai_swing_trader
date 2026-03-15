"""Robustness validation framework for backtest stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustnessConfig:
    """Configuration for robustness sweeps and stress tests."""

    score_threshold_range: list[int]
    atr_multiplier_range: list[float]
    slippage_bps_range: list[float]
    cost_multiplier_range: list[float]
    monte_carlo_runs: int = 50
    random_seed: int = 42

    @classmethod
    def default(cls) -> "RobustnessConfig":
        return cls(
            score_threshold_range=[60, 70, 80],
            atr_multiplier_range=[1.5, 2.0, 2.5],
            slippage_bps_range=[5.0, 10.0, 20.0],
            cost_multiplier_range=[1.0, 1.5, 2.0],
            monte_carlo_runs=50,
            random_seed=42,
        )


@dataclass(frozen=True)
class RobustnessResult:
    """Container for full robustness outputs."""

    parameter_results: list[dict[str, float]]
    cost_results: list[dict[str, float]]
    slippage_results: list[dict[str, float]]
    monte_carlo_summary: dict[str, float]
    robustness_score: float


class RobustnessEngine:
    """Run institutional-style robustness checks on strategy behavior."""

    def __init__(
        self,
        backtest_engine_factory: Callable[..., Any],
        base_config: Any,
        robustness_config: RobustnessConfig,
        price_data: dict[str, pd.DataFrame],
        fundamental_data: Any,
    ) -> None:
        self.backtest_engine_factory = backtest_engine_factory
        self.base_config = base_config
        self.robustness_config = robustness_config
        self.price_data = price_data
        self.fundamental_data = fundamental_data

    def run_parameter_sensitivity(self) -> list[dict[str, float]]:
        """Sweep score threshold and ATR multiplier combinations."""
        results: list[dict[str, float]] = []
        thresholds = sorted(self.robustness_config.score_threshold_range)
        atr_values = sorted(self.robustness_config.atr_multiplier_range)

        for score_threshold, atr_multiple in product(thresholds, atr_values):
            run = self._run_backtest(
                score_threshold=score_threshold,
                atr_multiple=atr_multiple,
                slippage_bps=float(getattr(self.base_config, "slippage_bps", 5.0)),
                cost_multiplier=1.0,
            )
            metrics = run.performance_metrics
            results.append(
                {
                    "score_threshold": float(score_threshold),
                    "atr_multiple": float(atr_multiple),
                    "cagr": float(metrics.get("cagr", 0.0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                }
            )
        return results

    def run_cost_stress(self) -> list[dict[str, float]]:
        """Run strategy under increased transaction cost multipliers."""
        results: list[dict[str, float]] = []
        for cost_multiplier in sorted(self.robustness_config.cost_multiplier_range):
            run = self._run_backtest(
                score_threshold=int(getattr(self.base_config, "score_threshold", 70)),
                atr_multiple=float(getattr(self.base_config, "atr_multiple", 2.0)),
                slippage_bps=float(getattr(self.base_config, "slippage_bps", 5.0)),
                cost_multiplier=cost_multiplier,
            )
            metrics = run.performance_metrics
            results.append(
                {
                    "cost_multiplier": float(cost_multiplier),
                    "cagr": float(metrics.get("cagr", 0.0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                }
            )
        return results

    def run_slippage_stress(self) -> list[dict[str, float]]:
        """Run strategy under different deterministic slippage assumptions."""
        results: list[dict[str, float]] = []
        for slippage_bps in sorted(self.robustness_config.slippage_bps_range):
            run = self._run_backtest(
                score_threshold=int(getattr(self.base_config, "score_threshold", 70)),
                atr_multiple=float(getattr(self.base_config, "atr_multiple", 2.0)),
                slippage_bps=slippage_bps,
                cost_multiplier=1.0,
            )
            metrics = run.performance_metrics
            results.append(
                {
                    "slippage_bps": float(slippage_bps),
                    "cagr": float(metrics.get("cagr", 0.0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                }
            )
        return results

    def run_monte_carlo(self) -> dict[str, float]:
        """Run lightweight Monte Carlo with shuffled returns and small noise."""
        base_run = self._run_backtest(
            score_threshold=int(getattr(self.base_config, "score_threshold", 70)),
            atr_multiple=float(getattr(self.base_config, "atr_multiple", 2.0)),
            slippage_bps=float(getattr(self.base_config, "slippage_bps", 5.0)),
            cost_multiplier=1.0,
        )
        equity = pd.Series(base_run.equity_curve, dtype=float).dropna()
        returns = equity.pct_change().dropna().to_numpy(dtype=float)
        if returns.size == 0:
            return {
                "median_cagr": 0.0,
                "worst_case_drawdown": 0.0,
                "return_5pct": 0.0,
            }

        rng = np.random.default_rng(self.robustness_config.random_seed)
        cagr_samples: list[float] = []
        drawdown_samples: list[float] = []
        final_return_samples: list[float] = []

        for _ in range(self.robustness_config.monte_carlo_runs):
            shuffled = rng.permutation(returns)
            noisy = shuffled + rng.normal(loc=0.0, scale=0.001, size=shuffled.size)
            curve = np.cumprod(1.0 + noisy)
            if curve.size == 0:
                continue
            final_return = float(curve[-1] - 1.0)
            running_peak = np.maximum.accumulate(curve)
            drawdown = float(np.max((running_peak - curve) / running_peak))
            years = max(curve.size / 252.0, 1 / 252.0)
            cagr = float((curve[-1] ** (1.0 / years)) - 1.0)

            final_return_samples.append(final_return)
            drawdown_samples.append(drawdown)
            cagr_samples.append(cagr)

        if not cagr_samples:
            return {
                "median_cagr": 0.0,
                "worst_case_drawdown": 0.0,
                "return_5pct": 0.0,
            }

        return {
            "median_cagr": float(np.median(cagr_samples)),
            "worst_case_drawdown": float(np.max(drawdown_samples)),
            "return_5pct": float(np.percentile(final_return_samples, 5)),
        }

    def run(self) -> RobustnessResult:
        """Run full robustness suite and compute composite robustness score."""
        parameter_results = self.run_parameter_sensitivity()
        cost_results = self.run_cost_stress()
        slippage_results = self.run_slippage_stress()
        monte_carlo_summary = self.run_monte_carlo()
        robustness_score = self._compute_robustness_score(
            parameter_results=parameter_results,
            cost_results=cost_results,
            slippage_results=slippage_results,
            monte_carlo_summary=monte_carlo_summary,
        )
        return RobustnessResult(
            parameter_results=parameter_results,
            cost_results=cost_results,
            slippage_results=slippage_results,
            monte_carlo_summary=monte_carlo_summary,
            robustness_score=robustness_score,
        )

    def _run_backtest(
        self,
        score_threshold: int,
        atr_multiple: float,
        slippage_bps: float,
        cost_multiplier: float,
    ) -> Any:
        engine = self.backtest_engine_factory(
            price_data=self.price_data,
            fundamental_data=self.fundamental_data,
            base_config=self.base_config,
            score_threshold=score_threshold,
            atr_multiple=atr_multiple,
            slippage_bps=slippage_bps,
            cost_multiplier=cost_multiplier,
        )
        return engine.run_backtest()

    def _compute_robustness_score(
        self,
        parameter_results: list[dict[str, float]],
        cost_results: list[dict[str, float]],
        slippage_results: list[dict[str, float]],
        monte_carlo_summary: dict[str, float],
    ) -> float:
        param_score = self._parameter_stability_score(parameter_results)
        cost_score = self._stress_resilience_score(cost_results)
        slippage_score = self._stress_resilience_score(slippage_results)
        monte_score = self._monte_score(monte_carlo_summary)
        return float(
            np.clip((param_score + cost_score + slippage_score + monte_score) / 4.0, 0.0, 100.0)
        )

    def _parameter_stability_score(self, results: list[dict[str, float]]) -> float:
        if not results:
            return 0.0
        cagr = np.array([float(x["cagr"]) for x in results], dtype=float)
        mean_abs = float(np.mean(np.abs(cagr)))
        std = float(np.std(cagr))
        if mean_abs == 0.0:
            return 0.0
        score = 100.0 * max(0.0, 1.0 - (std / (mean_abs + 1e-12)))
        return float(np.clip(score, 0.0, 100.0))

    def _stress_resilience_score(self, results: list[dict[str, float]]) -> float:
        if not results:
            return 0.0
        cagr = np.array([float(x["cagr"]) for x in results], dtype=float)
        baseline = float(cagr[0])
        if baseline <= 0:
            return float(np.clip(50.0 + 10.0 * np.mean(cagr), 0.0, 100.0))
        worst_ratio = float(np.min(cagr) / baseline)
        return float(np.clip(worst_ratio * 100.0, 0.0, 100.0))

    def _monte_score(self, summary: dict[str, float]) -> float:
        p5 = float(summary.get("return_5pct", 0.0))
        worst_dd = float(summary.get("worst_case_drawdown", 1.0))
        return_component = float(np.clip((p5 + 0.20) / 0.40, 0.0, 1.0))
        drawdown_component = float(np.clip((0.60 - worst_dd) / 0.60, 0.0, 1.0))
        return (0.6 * return_component + 0.4 * drawdown_component) * 100.0
