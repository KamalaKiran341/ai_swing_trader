from __future__ import annotations

from pathlib import Path

import pytest

from core.config.config_loader import (
    StrategyConfig,
    load_default_strategy_config,
    load_strategy_config,
)


def test_load_strategy_config_valid_default_file() -> None:
    config = load_strategy_config("config/strategy.yaml")

    assert isinstance(config, StrategyConfig)
    assert config.score_threshold == 70
    assert config.max_positions == 10
    assert config.rebalance_frequency == "monthly"
    assert config.atr_multiple == 2.0
    assert config.bear_allocation.liquid_etf == pytest.approx(0.6)
    assert config.bear_allocation.gold_etf == pytest.approx(0.4)
    assert config.liquidity.min_avg_turnover == pytest.approx(50_000_000)


def test_load_strategy_config_missing_required_field_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "strategy_missing.yaml"
    file_path.write_text(
        "\n".join(
            [
                "score_threshold: 70",
                "rebalance_frequency: monthly",
                "atr_multiple: 2.0",
                "bear_allocation:",
                "  liquid_etf: 0.6",
                "  gold_etf: 0.4",
                "liquidity:",
                "  min_avg_turnover: 50000000",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required field: max_positions"):
        load_strategy_config(str(file_path))


def test_load_strategy_config_invalid_range_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "strategy_bad_range.yaml"
    file_path.write_text(
        "\n".join(
            [
                "score_threshold: 101",
                "max_positions: 10",
                "rebalance_frequency: monthly",
                "atr_multiple: 2.0",
                "bear_allocation:",
                "  liquid_etf: 0.6",
                "  gold_etf: 0.4",
                "liquidity:",
                "  min_avg_turnover: 50000000",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="score_threshold must be between 0 and 100"):
        load_strategy_config(str(file_path))


def test_load_strategy_config_bear_allocation_sum_validation(tmp_path: Path) -> None:
    file_path = tmp_path / "strategy_bad_allocation.yaml"
    file_path.write_text(
        "\n".join(
            [
                "score_threshold: 70",
                "max_positions: 10",
                "rebalance_frequency: monthly",
                "atr_multiple: 2.0",
                "bear_allocation:",
                "  liquid_etf: 0.7",
                "  gold_etf: 0.4",
                "liquidity:",
                "  min_avg_turnover: 50000000",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bear_allocation values must sum to 1.0"):
        load_strategy_config(str(file_path))


def test_load_strategy_config_type_safety(tmp_path: Path) -> None:
    file_path = tmp_path / "strategy_bad_type.yaml"
    file_path.write_text(
        "\n".join(
            [
                "score_threshold: seventy",
                "max_positions: 10",
                "rebalance_frequency: monthly",
                "atr_multiple: 2.0",
                "bear_allocation:",
                "  liquid_etf: 0.6",
                "  gold_etf: 0.4",
                "liquidity:",
                "  min_avg_turnover: 50000000",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="score_threshold must be an integer"):
        load_strategy_config(str(file_path))


def test_load_default_strategy_config_works_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_default_strategy_config.cache_clear()
    monkeypatch.chdir(tmp_path)

    config = load_default_strategy_config()

    assert isinstance(config, StrategyConfig)
    assert config.rebalance_frequency == "monthly"


def test_load_default_strategy_config_is_cached() -> None:
    load_default_strategy_config.cache_clear()

    first = load_default_strategy_config()
    second = load_default_strategy_config()

    assert first is second


def test_load_strategy_config_invalid_path_handled() -> None:
    with pytest.raises(FileNotFoundError, match="Strategy config file not found"):
        load_strategy_config("config/does_not_exist.yaml")
