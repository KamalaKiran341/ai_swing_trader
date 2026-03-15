"""Institutional-grade performance analytics for strategy equity curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
ROLLING_SHARPE_WINDOW = 126  # ~6 months


@dataclass(frozen=True)
class PerformanceReport:
    """Structured analytics output."""

    core_metrics: dict[str, float]
    trade_metrics: dict[str, float]
    risk_metrics: dict[str, Any]
    stability_metrics: dict[str, Any]
    monthly_returns: pd.DataFrame
    rolling_sharpe: pd.Series


class PerformanceEngine:
    """Compute robust performance metrics from equity and trades."""

    def __init__(self, equity_curve: pd.Series, trades: pd.DataFrame | None = None) -> None:
        self.equity_curve = pd.Series(equity_curve, dtype=float).sort_index()
        if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
            self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
        self.equity_curve = self.equity_curve.dropna()
        self.trades = trades.copy() if trades is not None else pd.DataFrame()

    def generate_report(self, regime_series: pd.Series | None = None) -> PerformanceReport:
        """Generate full performance report."""
        returns = self.equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        monthly_df = self._monthly_returns_table()
        rolling_sharpe = self._rolling_sharpe(returns)
        max_drawdown, drawdown_series = self._max_drawdown(self.equity_curve)

        core_metrics = self._core_metrics(returns, max_drawdown)
        trade_metrics = self._trade_metrics(self.trades)
        risk_metrics = {
            "max_drawdown": float(max_drawdown),
            "recovery_time_days": float(self._recovery_time_days(drawdown_series)),
        }
        stability_metrics = self._stability_metrics(monthly_df, rolling_sharpe, regime_series)

        return PerformanceReport(
            core_metrics=core_metrics,
            trade_metrics=trade_metrics,
            risk_metrics=risk_metrics,
            stability_metrics=stability_metrics,
            monthly_returns=monthly_df,
            rolling_sharpe=rolling_sharpe,
        )

    def _core_metrics(self, returns: pd.Series, max_drawdown: float) -> dict[str, float]:
        if self.equity_curve.empty or returns.empty:
            return {
                "cagr": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": float(max_drawdown),
                "recovery_time_days": 0.0,
            }

        start = float(self.equity_curve.iloc[0])
        end = float(self.equity_curve.iloc[-1])
        years = max(len(returns) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
        cagr = (end / start) ** (1.0 / years) - 1.0 if start > 0 else 0.0

        vol = float(returns.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (
            0.0
            if vol == 0
            else float(returns.mean() / returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        )

        downside = returns[returns < 0]
        downside_std = (
            float(downside.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if not downside.empty
            else 0.0
        )
        sortino = (
            0.0
            if downside_std == 0
            else float(returns.mean() * TRADING_DAYS_PER_YEAR / downside_std)
        )
        calmar = 0.0 if max_drawdown <= 0 else float(cagr / max_drawdown)

        return {
            "cagr": float(cagr),
            "annualized_volatility": float(vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_drawdown),
            "recovery_time_days": float(
                self._recovery_time_days(self._max_drawdown(self.equity_curve)[1])
            ),
        }

    def _trade_metrics(self, trades: pd.DataFrame) -> dict[str, float]:
        if trades is None or trades.empty:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_holding_period_days": 0.0,
            }

        pnl_series = self._extract_pnl_series(trades)
        if pnl_series.empty:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_holding_period_days": 0.0,
            }

        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        win_rate = len(wins) / len(pnl_series)
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
        expectancy = float(pnl_series.mean())
        avg_holding = self._avg_holding_period_days(trades)

        return {
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "avg_holding_period_days": float(avg_holding),
        }

    def _stability_metrics(
        self,
        monthly_df: pd.DataFrame,
        rolling_sharpe: pd.Series,
        regime_series: pd.Series | None,
    ) -> dict[str, Any]:
        if monthly_df.empty:
            base = {
                "worst_month": 0.0,
                "consecutive_losing_months": 0.0,
                "monthly_return_5pct": 0.0,
                "equity_smoothness_score": 0.0,
                "rolling_sharpe_latest": 0.0,
            }
        else:
            values = monthly_df["return"]
            base = {
                "worst_month": float(values.min()),
                "consecutive_losing_months": float(_max_consecutive_losses(values)),
                "monthly_return_5pct": float(np.percentile(values, 5)),
                "equity_smoothness_score": float(self._equity_smoothness_score()),
                "rolling_sharpe_latest": (
                    float(rolling_sharpe.dropna().iloc[-1])
                    if not rolling_sharpe.dropna().empty
                    else 0.0
                ),
            }

        if regime_series is not None:
            base["regime_attribution"] = self._regime_attribution(regime_series)
        return base

    def _monthly_returns_table(self) -> pd.DataFrame:
        if self.equity_curve.empty:
            return pd.DataFrame(columns=["return"])
        monthly = self.equity_curve.resample("ME").last().pct_change().dropna()
        return monthly.to_frame(name="return")

    def _rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        if returns.empty:
            return pd.Series(dtype=float)
        rolling_mean = returns.rolling(ROLLING_SHARPE_WINDOW, min_periods=20).mean()
        rolling_std = returns.rolling(ROLLING_SHARPE_WINDOW, min_periods=20).std(ddof=0)
        sharpe = pd.Series(0.0, index=rolling_mean.index, dtype=float)
        valid = rolling_std > 0
        sharpe.loc[valid] = (
            rolling_mean.loc[valid] / rolling_std.loc[valid] * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        sharpe.loc[rolling_mean.isna()] = np.nan
        return sharpe

    def _max_drawdown(self, equity: pd.Series) -> tuple[float, pd.Series]:
        if equity.empty:
            return 0.0, pd.Series(dtype=float)
        running_peak = equity.cummax()
        drawdown = (running_peak - equity) / running_peak.replace(0, np.nan)
        drawdown = drawdown.fillna(0.0)
        return float(drawdown.max()), drawdown

    def _recovery_time_days(self, drawdown: pd.Series) -> int:
        if drawdown.empty:
            return 0
        max_duration = 0
        current_duration = 0
        for value in drawdown.to_numpy():
            if value > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return int(max_duration)

    def _extract_pnl_series(self, trades: pd.DataFrame) -> pd.Series:
        if "pnl" in trades.columns:
            pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
            return pnl
        if "realized_pnl" in trades.columns:
            pnl = pd.to_numeric(trades["realized_pnl"], errors="coerce").dropna()
            return pnl
        return pd.Series(dtype=float)

    def _avg_holding_period_days(self, trades: pd.DataFrame) -> float:
        if "holding_period_days" in trades.columns:
            values = pd.to_numeric(trades["holding_period_days"], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else 0.0
        if {"entry_date", "exit_date"}.issubset(trades.columns):
            entry = pd.to_datetime(trades["entry_date"], errors="coerce")
            exit_ = pd.to_datetime(trades["exit_date"], errors="coerce")
            diff = (exit_ - entry).dt.days.dropna()
            return float(diff.mean()) if not diff.empty else 0.0
        return 0.0

    def _equity_smoothness_score(self) -> float:
        if len(self.equity_curve) < 3:
            return 0.0
        y = self.equity_curve.to_numpy(dtype=float)
        if float(np.std(y)) == 0.0:
            return 0.0
        x = np.linspace(0.0, 1.0, len(y))
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def _regime_attribution(self, regime_series: pd.Series) -> dict[str, float]:
        if self.equity_curve.empty:
            return {"bull": 0.0, "bear": 0.0, "sideways": 0.0}
        aligned_regime = regime_series.reindex(self.equity_curve.index).ffill().fillna("sideways")
        returns = self.equity_curve.pct_change().fillna(0.0)
        out = {"bull": 0.0, "bear": 0.0, "sideways": 0.0}
        for name in out:
            mask = aligned_regime.astype(str).str.lower() == name
            if mask.any():
                out[name] = float((1.0 + returns[mask]).prod() - 1.0)
        return out


def _max_consecutive_losses(values: pd.Series) -> int:
    max_streak = 0
    current = 0
    for val in values.to_numpy(dtype=float):
        if val < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)
