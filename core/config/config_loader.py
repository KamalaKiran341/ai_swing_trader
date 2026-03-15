"""Typed configuration loader for trading strategy settings."""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BearAllocationConfig:
    """Portfolio allocation used in bear-market conditions."""

    liquid_etf: float
    gold_etf: float


@dataclass(frozen=True)
class LiquidityConfig:
    """Liquidity filters applied to instrument selection."""

    min_avg_turnover: float


@dataclass(frozen=True)
class StrategyConfig:
    """Top-level strategy configuration."""

    score_threshold: int
    max_positions: int
    rebalance_frequency: str
    atr_multiple: float
    bear_allocation: BearAllocationConfig
    liquidity: LiquidityConfig


def load_strategy_config(path: str) -> StrategyConfig:
    """Load and validate strategy configuration from a YAML file.

    Args:
        path: Filesystem path to the YAML strategy configuration.

    Returns:
        A validated and typed ``StrategyConfig`` instance.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the YAML cannot be parsed or validation fails.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Strategy config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in strategy config '{config_path}': {exc}") from exc

    if raw is None:
        raise ValueError("Strategy config is empty.")
    if not isinstance(raw, dict):
        raise ValueError("Strategy config must be a mapping at the top level.")

    score_threshold = _get_required_int(raw, "score_threshold")
    if not 0 <= score_threshold <= 100:
        raise ValueError("score_threshold must be between 0 and 100.")

    max_positions = _get_required_int(raw, "max_positions")
    if max_positions <= 0:
        raise ValueError("max_positions must be greater than 0.")

    rebalance_frequency = _get_required_str(raw, "rebalance_frequency")
    allowed_rebalance = {"monthly"}
    if rebalance_frequency not in allowed_rebalance:
        raise ValueError("rebalance_frequency must be one of {'monthly'}.")

    atr_multiple = _get_required_float(raw, "atr_multiple")
    if atr_multiple <= 0:
        raise ValueError("atr_multiple must be greater than 0.")

    bear_raw = _get_required_dict(raw, "bear_allocation")
    liquid_etf = _get_required_float(bear_raw, "liquid_etf", parent="bear_allocation")
    gold_etf = _get_required_float(bear_raw, "gold_etf", parent="bear_allocation")
    total_allocation = liquid_etf + gold_etf
    if not math.isclose(total_allocation, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("bear_allocation values must sum to 1.0.")
    bear_allocation = BearAllocationConfig(liquid_etf=liquid_etf, gold_etf=gold_etf)

    liquidity_raw = _get_required_dict(raw, "liquidity")
    min_avg_turnover = _get_required_float(liquidity_raw, "min_avg_turnover", parent="liquidity")
    if min_avg_turnover <= 0:
        raise ValueError("liquidity.min_avg_turnover must be greater than 0.")
    liquidity = LiquidityConfig(min_avg_turnover=min_avg_turnover)

    return StrategyConfig(
        score_threshold=score_threshold,
        max_positions=max_positions,
        rebalance_frequency=rebalance_frequency,
        atr_multiple=atr_multiple,
        bear_allocation=bear_allocation,
        liquidity=liquidity,
    )


@lru_cache(maxsize=1)
def load_default_strategy_config() -> StrategyConfig:
    """Load the default strategy config from ``config/strategy.yaml``.

    The path is resolved relative to the project root, making the loader
    independent from the current working directory.
    """
    default_path = _project_root() / "config" / "strategy.yaml"
    return load_strategy_config(str(default_path))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _field_name(name: str, parent: str | None = None) -> str:
    if parent:
        return f"{parent}.{name}"
    return name


def _get_required_dict(raw: dict[str, Any], key: str, parent: str | None = None) -> dict[str, Any]:
    value = _get_required(raw, key, parent=parent)
    if not isinstance(value, dict):
        raise ValueError(f"{_field_name(key, parent)} must be a mapping.")
    return value


def _get_required_str(raw: dict[str, Any], key: str, parent: str | None = None) -> str:
    value = _get_required(raw, key, parent=parent)
    if not isinstance(value, str):
        raise ValueError(f"{_field_name(key, parent)} must be a string.")
    return value


def _get_required_int(raw: dict[str, Any], key: str, parent: str | None = None) -> int:
    value = _get_required(raw, key, parent=parent)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_field_name(key, parent)} must be an integer.")
    return value


def _get_required_float(raw: dict[str, Any], key: str, parent: str | None = None) -> float:
    value = _get_required(raw, key, parent=parent)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{_field_name(key, parent)} must be a number.")
    return float(value)


def _get_required(raw: dict[str, Any], key: str, parent: str | None = None) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required field: {_field_name(key, parent)}")
    return raw[key]
