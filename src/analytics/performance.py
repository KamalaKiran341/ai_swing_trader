import numpy as np


def calculate_performance(equity_curve):

    df = equity_curve.copy()
    df["returns"] = df["equity"].pct_change().fillna(0)

    # CAGR
    total_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365
    cagr = (df["equity"].iloc[-1] / df["equity"].iloc[0]) ** (1 / total_years) - 1

    # Max Drawdown
    cum_max = df["equity"].cummax()
    drawdown = df["equity"] / cum_max - 1
    max_dd = drawdown.min()

    # Win rate (if trade log exists later)
    win_rate = None

    return {
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Final Equity": df["equity"].iloc[-1],
    }
