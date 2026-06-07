import pandas as pd
import numpy as np


def rolling_lap_std(driver_laps: pd.DataFrame, window: int = 5) -> pd.Series:
    # Rolling std of lap times within a stint — equivalent to historical vol window in finance
    # finance: returns.rolling(21).std() * sqrt(252)
    return driver_laps["LapTime_s"].rolling(window, min_periods=2).std()


def stint_volatility(laps: pd.DataFrame) -> pd.DataFrame:
    # Std of lap times grouped by driver and stint
    # Each stint is a regime — new compound, different degradation curve
    # finance equivalent: volatility per market regime (bull/bear state from HMM)
    result = (
        laps.groupby(["Driver", "Stint"])["LapTime_s"]
        .agg(std="std", mean="mean", lap_count="count")
        .reset_index()
    )
    result["cv"] = result["std"] / result["mean"]  # coefficient of variation — normalizes across drivers
    return result


def volatility_by_tyre_age(laps: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    # Does lap time variance increase as tyres degrade?
    # Bins TyreLife into equal-width buckets, computes std per bucket
    # finance equivalent: volatility term structure (how vol changes with time-to-expiry)
    laps = laps.copy()
    laps["TyreAge_bin"] = pd.cut(laps["TyreLife"], bins=bins)
    return (
        laps.groupby("TyreAge_bin", observed=True)["LapTime_s"]
        .agg(std="std", mean="mean", count="count")
        .reset_index()
    )


def driver_volatility_summary(laps: pd.DataFrame) -> pd.DataFrame:
    # Single-number vol summary per driver across the whole race
    # Useful for ranking drivers by consistency — lower std = more consistent
    return (
        laps.groupby("Driver")["LapTime_s"]
        .agg(std="std", mean="mean", lap_count="count")
        .sort_values("std")
        .reset_index()
    )
