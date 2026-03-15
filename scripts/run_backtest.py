"""CLI entrypoint to run a deterministic backtest and export reports."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Support `python scripts/run_backtest.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy.regime_detector import Regime  # noqa: E402
from core.strategy.scoring_engine import StockScore  # noqa: E402
from src.analytics.performance_engine import PerformanceEngine  # noqa: E402
from src.backtest.backtest_engine import (  # noqa: E402
    GOLD_ETF_SYMBOL,
    LIQUID_ETF_SYMBOL,
    BacktestEngine,
)
from src.execution.order_simulator import OrderSimulator  # noqa: E402
from src.risk.portfolio_risk import PortfolioRiskController  # noqa: E402
from src.risk.trailing_stop import ATRTrailingStop  # noqa: E402

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOGGER = logging.getLogger(__name__)


@dataclass
class _Config:
    max_positions: int
    initial_capital: float


class _RegimeDetectorStub:
    def __init__(self, regime: Regime = Regime.BULL) -> None:
        self._regime = regime

    def get_regime(self, as_of_date: pd.Timestamp) -> Regime:
        _ = as_of_date
        return self._regime


class _MomentumScoringEngine:
    def __init__(self, price_data: dict[str, pd.DataFrame]) -> None:
        self.price_data = price_data

    def score_universe(self, symbols: list[str], as_of_date: pd.Timestamp) -> list[StockScore]:
        scored: list[StockScore] = []
        date = pd.Timestamp(as_of_date).normalize()

        for symbol in symbols:
            frame = self.price_data.get(symbol)
            if frame is None or frame.empty or date not in frame.index:
                continue
            closes = frame["close"].loc[:date].tail(22)
            if len(closes) < 2:
                continue
            momentum = float((closes.iloc[-1] / closes.iloc[0] - 1.0) * 100.0)
            score = float(np.clip(50.0 + momentum * 2.0, 0.0, 100.0))
            scored.append(
                StockScore(
                    symbol=symbol,
                    score=score,
                    technical_score=score,
                    fundamental_score=50.0,
                )
            )
        return sorted(scored, key=lambda x: x.score, reverse=True)


def _flatten_metrics(
    report: object,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    monthly_returns: dict[str, float],
) -> dict[str, object]:
    return {
        "start_date": equity_df["date"].iloc[0].strftime("%Y-%m-%d") if not equity_df.empty else "",
        "end_date": equity_df["date"].iloc[-1].strftime("%Y-%m-%d") if not equity_df.empty else "",
        "equity_points": int(len(equity_df)),
        "trades_count": int(len(trades_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 0.0,
        "cagr": float(report.core_metrics.get("cagr", 0.0)),
        "annualized_volatility": float(report.core_metrics.get("annualized_volatility", 0.0)),
        "sharpe_ratio": float(report.core_metrics.get("sharpe_ratio", 0.0)),
        "sortino_ratio": float(report.core_metrics.get("sortino_ratio", 0.0)),
        "calmar_ratio": float(report.core_metrics.get("calmar_ratio", 0.0)),
        "max_drawdown": float(report.core_metrics.get("max_drawdown", 0.0)),
        "recovery_time_days": float(report.core_metrics.get("recovery_time_days", 0.0)),
        "win_rate": float(report.trade_metrics.get("win_rate", 0.0)),
        "avg_win": float(report.trade_metrics.get("avg_win", 0.0)),
        "avg_loss": float(report.trade_metrics.get("avg_loss", 0.0)),
        "profit_factor": float(report.trade_metrics.get("profit_factor", 0.0)),
        "expectancy": float(report.trade_metrics.get("expectancy", 0.0)),
        "avg_holding_period_days": float(report.trade_metrics.get("avg_holding_period_days", 0.0)),
        "worst_month": float(report.stability_metrics.get("worst_month", 0.0)),
        "consecutive_losing_months": float(
            report.stability_metrics.get("consecutive_losing_months", 0.0)
        ),
        "monthly_return_5pct": float(report.stability_metrics.get("monthly_return_5pct", 0.0)),
        "equity_smoothness_score": float(
            report.stability_metrics.get("equity_smoothness_score", 0.0)
        ),
        "rolling_sharpe_latest": float(report.stability_metrics.get("rolling_sharpe_latest", 0.0)),
        "monthly_returns": monthly_returns,
    }


def _load_single_price_file(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported price file extension for {path}")

    if "date" not in frame.columns:
        raise ValueError(f"Missing 'date' column in {path}")
    if "close" not in frame.columns and "adj_close" in frame.columns:
        frame = frame.rename(columns={"adj_close": "close"})

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))]
    out = out.sort_values("date", kind="mergesort")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])

    return out.set_index("date")[["open", "high", "low", "close", "volume"]]


def load_price_data_from_dir(
    base_dir: str,
    start_date: str,
    end_date: str,
    symbols: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    base = Path(base_dir)
    prices_dir = base / "prices" if (base / "prices").exists() else base
    files = sorted(
        list(prices_dir.glob("*.parquet")) + list(prices_dir.glob("*.csv")),
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"No .parquet/.csv files found in {prices_dir}")

    symbol_filter = {s.strip().upper() for s in symbols or [] if s and s.strip()}
    output: dict[str, pd.DataFrame] = {}
    for file_path in files:
        symbol = file_path.stem.strip().upper()
        if symbol_filter and symbol not in symbol_filter:
            continue
        frame = _load_single_price_file(file_path, start_date, end_date)
        if frame.empty:
            continue
        output[symbol] = frame

    if not output:
        raise ValueError("No usable price data loaded for the selected date range/symbols.")
    return output


def _price_frame(dates: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1_000_000.0),
        },
        index=dates,
    )


def build_demo_price_data(start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(start_date, end_date)
    n = len(dates)
    if n < 30:
        raise ValueError("Date range too short. Use at least 30 business days.")

    aaa = np.linspace(100.0, 135.0, n) + np.sin(np.linspace(0, 8, n)) * 2.0
    bbb = np.linspace(90.0, 120.0, n) + np.cos(np.linspace(0, 7, n)) * 1.5
    ccc = np.linspace(120.0, 95.0, n) + np.sin(np.linspace(0, 6, n)) * 2.5
    etf_l = np.linspace(50.0, 52.0, n)
    etf_g = np.linspace(40.0, 43.0, n)

    return {
        "AAA": _price_frame(dates, aaa),
        "BBB": _price_frame(dates, bbb),
        "CCC": _price_frame(dates, ccc),
        LIQUID_ETF_SYMBOL: _price_frame(dates, etf_l),
        GOLD_ETF_SYMBOL: _price_frame(dates, etf_g),
    }


def create_engine(
    price_data: dict[str, pd.DataFrame],
    initial_capital: float,
    max_positions: int,
) -> BacktestEngine:
    return BacktestEngine(
        price_data=price_data,
        fundamental_data={},
        regime_detector=_RegimeDetectorStub(Regime.BULL),
        scoring_engine=_MomentumScoringEngine(price_data),
        exit_engine=ATRTrailingStop(atr_multiple=2.0),
        risk_controller=PortfolioRiskController(
            max_positions=max_positions,
            losing_month_tolerance=2,
            circuit_breaker_enabled=True,
        ),
        order_simulator=OrderSimulator(slippage_bps=5.0),
        config=_Config(max_positions=max_positions, initial_capital=initial_capital),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic swing backtest.")
    parser.add_argument("--start-date", default="2024-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument(
        "--price-data-dir",
        default=None,
        help="Directory with per-symbol OHLCV files (.parquet/.csv). Supports <dir>/prices/*.parquet.",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated symbols to include (e.g. RELIANCE,TCS,HDFCBANK).",
    )
    parser.add_argument("--output-dir", default="reports/backtest")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]

    try:
        if args.price_data_dir:
            LOGGER.info("event=load_price_data mode=real path=%s", args.price_data_dir)
            price_data = load_price_data_from_dir(
                base_dir=args.price_data_dir,
                start_date=args.start_date,
                end_date=args.end_date,
                symbols=symbols or None,
            )
        else:
            LOGGER.info("event=load_price_data mode=demo")
            price_data = build_demo_price_data(start_date=args.start_date, end_date=args.end_date)
    except Exception as exc:
        LOGGER.error("event=price_data_load_failure error=%s", exc)
        return 2

    try:
        engine = create_engine(
            price_data=price_data,
            initial_capital=args.initial_capital,
            max_positions=max(args.max_positions, 1),
        )
    except Exception as exc:
        LOGGER.error("event=backtest_init_failure error=%s", exc)
        return 2

    result = engine.run_backtest()
    equity_curve = pd.Series(result.equity_curve, dtype=float).sort_index()
    equity_df = equity_curve.rename("equity").reset_index().rename(columns={"index": "date"})
    trades_df = pd.DataFrame([t.__dict__ for t in result.trades])

    report = PerformanceEngine(equity_curve=equity_curve, trades=trades_df).generate_report()
    monthly_returns = {
        ts.strftime("%Y-%m-%d"): float(value)
        for ts, value in report.monthly_returns["return"].items()
    }
    metrics = _flatten_metrics(
        report=report,
        equity_df=equity_df,
        trades_df=trades_df,
        monthly_returns=monthly_returns,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "equity_curve.csv"
    trades_path = output_dir / "trades.csv"
    metrics_path = output_dir / "metrics.json"
    metrics_flat_path = output_dir / "metrics_flat.csv"

    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_metrics = dict(metrics)
    csv_metrics["monthly_returns_json"] = json.dumps(
        metrics.get("monthly_returns", {}), sort_keys=True
    )
    csv_metrics.pop("monthly_returns", None)
    pd.DataFrame([csv_metrics]).to_csv(metrics_flat_path, index=False)

    print(f"Backtest completed. Equity points: {len(equity_df)}")
    print(f"Final Equity: {equity_df['equity'].iloc[-1]:.2f}")
    print(f"Reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
