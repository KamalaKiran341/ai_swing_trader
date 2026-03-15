from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.data.price_loader import PriceDataLoader


def _write_symbol_parquet(base_path: Path, symbol: str, frame: pd.DataFrame) -> None:
    prices_dir = base_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(prices_dir / f"{symbol}.parquet", index=False)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1_000, 1_100, 1_200, 1_300, 1_400],
            "adj_close": [98.0, 99.0, 100.0, 101.0, 102.0],
        }
    )


def test_load_valid_parquet(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAPL", _sample_frame())
    loader = PriceDataLoader(base_path=str(base_path))

    history = loader.get_history("AAPL", pd.Timestamp("2024-01-05"), lookback=5)

    assert len(history) == 5
    assert list(history.columns) == ["open", "high", "low", "close", "volume"]


def test_get_history_normal_case(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAPL", _sample_frame())
    loader = PriceDataLoader(base_path=str(base_path))

    history = loader.get_history("AAPL", pd.Timestamp("2024-01-05"), lookback=3)

    assert len(history) == 3
    assert history.index[0] == pd.Timestamp("2024-01-03")
    assert history.index[-1] == pd.Timestamp("2024-01-05")


def test_get_history_insufficient_lookback(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAPL", _sample_frame())
    loader = PriceDataLoader(base_path=str(base_path))

    history = loader.get_history("AAPL", pd.Timestamp("2024-01-03"), lookback=10)

    assert len(history) == 3
    assert history.index[-1] == pd.Timestamp("2024-01-03")


def test_get_price_exact_date(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAPL", _sample_frame())
    loader = PriceDataLoader(base_path=str(base_path))

    price = loader.get_price("AAPL", pd.Timestamp("2024-01-04"))

    assert price == pytest.approx(101.0)


def test_get_price_missing_date_raises(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAPL", _sample_frame())
    loader = PriceDataLoader(base_path=str(base_path))

    with pytest.raises(ValueError, match="is missing on"):
        loader.get_price("AAPL", pd.Timestamp("2024-02-01"))


def test_adjusted_vs_non_adjusted(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    _write_symbol_parquet(base_path, "AAPL", _sample_frame())
    adjusted_loader = PriceDataLoader(base_path=str(base_path), use_adjusted=True)
    raw_loader = PriceDataLoader(base_path=str(base_path), use_adjusted=False)

    adjusted_price = adjusted_loader.get_price("AAPL", pd.Timestamp("2024-01-05"))
    raw_price = raw_loader.get_price("AAPL", pd.Timestamp("2024-01-05"))

    assert adjusted_price == pytest.approx(102.0)
    assert raw_price == pytest.approx(104.0)


def test_cache_eviction_works(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    frame = _sample_frame()
    _write_symbol_parquet(base_path, "AAPL", frame)
    _write_symbol_parquet(base_path, "MSFT", frame)
    _write_symbol_parquet(base_path, "GOOG", frame)
    loader = PriceDataLoader(base_path=str(base_path), cache_size=2)

    loader.preload(["AAPL", "MSFT"])
    assert list(loader._cache.keys()) == ["AAPL", "MSFT"]

    loader.get_history("GOOG", pd.Timestamp("2024-01-05"), lookback=1)
    assert list(loader._cache.keys()) == ["MSFT", "GOOG"]


def test_validation_failures(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    bad_columns = _sample_frame().drop(columns=["adj_close"])
    _write_symbol_parquet(base_path, "MISS", bad_columns)

    loader = PriceDataLoader(base_path=str(base_path))
    with pytest.raises(ValueError, match="missing required columns"):
        loader.get_history("MISS", pd.Timestamp("2024-01-05"), lookback=1)

    duplicate_dates = _sample_frame()
    duplicate_dates.loc[1, "date"] = duplicate_dates.loc[0, "date"]
    _write_symbol_parquet(base_path, "DUPL", duplicate_dates)
    with pytest.raises(ValueError, match="duplicate date values"):
        loader.get_history("DUPL", pd.Timestamp("2024-01-05"), lookback=1)
