import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def compute_spread(laps: pd.DataFrame, driver_a: str, driver_b: str) -> pd.DataFrame:
    # Pivot so each driver is a column, rows aligned by lap number
    # Equivalent to aligning two price series on the same date index before computing spread
    pivot = (
        laps[laps["Driver"].isin([driver_a, driver_b])]
        .pivot(index="LapNumber", columns="Driver", values="LapTime_s")
        .dropna()
    )
    pivot["spread"] = pivot[driver_a] - pivot[driver_b]
    return pivot


def adf_test(spread: pd.Series) -> dict:
    # Augmented Dickey-Fuller test — null hypothesis is that the series has a unit root (non-stationary)
    # p-value < 0.05: reject null, spread is stationary (mean-reverting)
    # finance: exact test used to confirm two stocks are cointegrated before building a pairs strategy
    result = adfuller(spread.dropna(), autolag="AIC")
    return {
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": result[1] < 0.05,
        "critical_values": result[4],
    }


def spread_zscore(spread: pd.Series, window: int = 10) -> pd.Series:
    # Rolling z-score of the spread: how many std devs from the rolling mean
    # finance: this is the entry/exit signal in a pairs trade (enter at ±2, exit at 0)
    # in F1: tells you when one driver is anomalously slow vs their teammate
    mean = spread.rolling(window, min_periods=2).mean()
    std = spread.rolling(window, min_periods=2).std()
    return (spread - mean) / std


def pairs_summary(laps: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    # Full pipeline: compute spread, test stationarity, return summary
    spread_df = compute_spread(laps, driver_a, driver_b)
    spread = spread_df["spread"]
    adf = adf_test(spread)

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "laps_compared": len(spread),
        "spread_mean": round(spread.mean(), 3),
        "spread_std": round(spread.std(), 3),
        "adf_stat": adf["adf_stat"],
        "p_value": adf["p_value"],
        "is_stationary": adf["is_stationary"],
    }
