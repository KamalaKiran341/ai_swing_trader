import numpy as np
import pandas as pd

from core.utils.time_utils import get_today_utc


def test_sanity_imports_and_framework() -> None:
    arr = np.array([1, 2, 3], dtype=np.int64)
    series = pd.Series(arr)

    assert arr.sum() == 6
    assert series.iloc[0] == 1


def test_get_today_utc_returns_utc_timestamp() -> None:
    ts = get_today_utc()

    assert isinstance(ts, pd.Timestamp)
    assert ts.tz is not None
    assert ts.tzname() == "UTC"
