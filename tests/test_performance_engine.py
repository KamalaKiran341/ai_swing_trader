from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.performance_engine import PerformanceEngine


def test_cagr_correct() -> None:
    dates = pd.bdate_range("2024-01-01", periods=253)
    equity = pd.Series(np.linspace(100.0, 200.0, len(dates)), index=dates)
    report = PerformanceEngine(equity).generate_report()
    assert report.core_metrics["cagr"] == pytest.approx(1.0, rel=0.05)


def test_drawdown_correct() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0], index=dates)
    report = PerformanceEngine(equity).generate_report()
    assert report.core_metrics["max_drawdown"] == pytest.approx(0.25)


def test_sharpe_stable_with_zero_volatility() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    equity = pd.Series([100.0] * len(dates), index=dates)
    report = PerformanceEngine(equity).generate_report()
    assert report.core_metrics["sharpe_ratio"] == 0.0


def test_trade_metrics_correct() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    equity = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)
    trades = pd.DataFrame(
        {
            "pnl": [100.0, -50.0, 150.0, -25.0],
            "holding_period_days": [10, 8, 12, 7],
        }
    )
    report = PerformanceEngine(equity, trades=trades).generate_report()

    assert report.trade_metrics["win_rate"] == pytest.approx(0.5)
    assert report.trade_metrics["avg_win"] == pytest.approx(125.0)
    assert report.trade_metrics["avg_loss"] == pytest.approx(-37.5)
    assert report.trade_metrics["profit_factor"] == pytest.approx(250.0 / 75.0)
    assert report.trade_metrics["expectancy"] == pytest.approx(43.75)
    assert report.trade_metrics["avg_holding_period_days"] == pytest.approx(9.25)


def test_rolling_sharpe_computed() -> None:
    dates = pd.bdate_range("2023-01-01", periods=260)
    equity = pd.Series(np.cumprod(np.full(len(dates), 1.001)), index=dates)
    report = PerformanceEngine(equity).generate_report()
    assert not report.rolling_sharpe.dropna().empty


def test_handles_empty_trades() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    equity = pd.Series(np.linspace(100.0, 101.0, len(dates)), index=dates)
    report = PerformanceEngine(equity, trades=pd.DataFrame()).generate_report()
    assert report.trade_metrics["win_rate"] == 0.0
    assert report.trade_metrics["expectancy"] == 0.0


def test_deterministic_behavior() -> None:
    dates = pd.bdate_range("2023-01-01", periods=260)
    equity = pd.Series(np.cumprod(np.linspace(1.0001, 1.001, len(dates))), index=dates)
    trades = pd.DataFrame({"pnl": [10.0, -3.0, 5.0], "holding_period_days": [4, 5, 6]})

    report_a = PerformanceEngine(equity, trades=trades).generate_report()
    report_b = PerformanceEngine(equity, trades=trades).generate_report()

    assert report_a.core_metrics == report_b.core_metrics
    assert report_a.trade_metrics == report_b.trade_metrics
    assert report_a.risk_metrics == report_b.risk_metrics
    assert report_a.stability_metrics == report_b.stability_metrics
    pd.testing.assert_frame_equal(report_a.monthly_returns, report_b.monthly_returns)
    pd.testing.assert_series_equal(report_a.rolling_sharpe, report_b.rolling_sharpe)
