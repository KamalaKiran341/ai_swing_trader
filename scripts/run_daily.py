"""Cron-friendly CLI to execute one daily paper-trading run."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from datetime import date

import pandas as pd

from src.automation.daily_runner import DailyStrategyRunner

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def create_runner() -> DailyStrategyRunner:
    """Create a configured ``DailyStrategyRunner`` from a factory path.

    Set environment variable ``DAILY_RUNNER_FACTORY`` to
    ``module_path:function_name`` where the function returns
    a fully initialized ``DailyStrategyRunner``.
    """
    factory_path = os.getenv("DAILY_RUNNER_FACTORY")
    if not factory_path:
        raise RuntimeError("DAILY_RUNNER_FACTORY is not set.")

    if ":" not in factory_path:
        raise RuntimeError("DAILY_RUNNER_FACTORY must be in 'module:function' format.")
    module_name, func_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    runner = factory()
    if not isinstance(runner, DailyStrategyRunner):
        raise RuntimeError("Factory did not return DailyStrategyRunner.")
    return runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run daily paper trading workflow.")
    parser.add_argument(
        "--date", dest="run_date", default=None, help="Run date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--retries", dest="retries", default=2, type=int, help="Number of retries on failure."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    try:
        run_date = pd.Timestamp(args.run_date).date() if args.run_date else date.today()
    except Exception:
        logging.getLogger(__name__).error("event=cli_invalid_date value=%s", args.run_date)
        return 2

    try:
        runner = create_runner()
    except Exception as exc:
        logging.getLogger(__name__).error("event=cli_init_failure error=%s", str(exc))
        return 2

    status = runner.run_with_retry(run_date, max_retries=max(args.retries, 0))
    if status.success:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
