"""Time-related utility functions."""

from __future__ import annotations

import pandas as pd


def get_today_utc() -> pd.Timestamp:
    """Return the current UTC timestamp as a timezone-aware pandas Timestamp."""
    return pd.Timestamp.now(tz="UTC")
