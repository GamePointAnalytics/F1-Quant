import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress

# In finance, pairs trading is a market-neutral strategy that involves identifying two historically correlated assets (like stocks) and trading them based on the divergence of their price spread. The idea is to go long on the underperforming asset and short on the outperforming one, betting that the spread will revert to its historical mean.
# In F1, we can think of two drivers on the same team as a "pair" — they should have similar performance since they have the same car. If one driver is consistently faster, we can analyze the "spread" in their lap times to see if it's mean-reverting (stationary). If it is, we can predict that if one driver has a bad lap, they might bounce back, while the other might have a worse lap next time.
# This code computes the lap time spread between two drivers, tests if it's stationary (mean-reverting), and summarizes the results. It's analogous to analyzing the price spread between two stocks in a pairs trading strategy.
def _align_drivers(laps: pd.DataFrame, driver_a: str, driver_b: str) -> pd.DataFrame:
    # Pivot so each driver is a column, rows aligned by lap number
    # Equivalent to aligning two price series on the same date index before computing spread
    return (
        laps[laps["Driver"].isin([driver_a, driver_b])]  #like filtering for two stocks in a dataset, asset prices/yields
        .pivot(index="LapNumber", columns="Driver", values="LapTime_s")
        .dropna()
    )


def estimate_hedge_ratio(laps: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    # Engle-Granger step 1: OLS regression of driver_a on driver_b
    # Finds beta such that driver_a ≈ beta * driver_b + alpha
    # Raw spread assumes beta=1 (A - B), but drivers aren't equally fast — beta corrects for that
    # finance: same step before cointegration test; beta is the hedge ratio (how many units of B to short per unit of A)
    pivot = _align_drivers(laps, driver_a, driver_b)
    slope, intercept, r_value, _, _ = linregress(pivot[driver_b], pivot[driver_a])
    return {
        "hedge_ratio": round(slope, 4),   # beta — multiply driver_b by this before subtracting
        "intercept": round(intercept, 4),
        "r_squared": round(r_value ** 2, 4),  # how well driver_b explains driver_a's lap times
    }


def compute_spread(laps: pd.DataFrame, driver_a: str, driver_b: str, hedge_ratio: float = 1.0) -> pd.DataFrame:
    pivot = _align_drivers(laps, driver_a, driver_b)
    # hedge_ratio=1.0 gives raw spread; pass estimate_hedge_ratio()["hedge_ratio"] for OLS-adjusted spread
    pivot["spread"] = pivot[driver_a] - hedge_ratio * pivot[driver_b]
    return pivot

# ADF test checks if the spread is stationary (mean-reverting) — a key requirement for a pairs strategy to work
# Stationary means spread has a constant mean, variance,and autocorrelation
# Non-stationary spread will drift over time, leading to unbounded losses if you bet on mean reversion
def adf_test(spread: pd.Series) -> dict:
    # Augmented Dickey-Fuller test — null hypothesis is that the series has a unit root (non-stationary)
    # p-value < 0.05: reject null, spread is stationary (mean-reverting)
    # finance: exact test used to confirm two stocks are cointegrated before building a pairs strategy
    result = adfuller(spread.dropna(), autolag="AIC")
    # tries to find a "unit root" in the spread, which would indicate non-stationarity. If it finds one, it returns a high p-value, suggesting we fail to reject the null hypothesis of non-stationarity.
    # ADF test statistic is negative, more negative means stronger evidence against the null (more likely to be stationary). Critical values are thresholds for different confidence levels (1%, 5%, 10%) to compare against the ADF statistic.
    # think about mean reversion and fast/slow
    return {
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": result[1] < 0.05,
        "critical_values": result[4],
    }

# we take into account pit stops by using a rolling window to compute z-scores
# does not take into account hugely variant pitstop strategies like one stop vs two stop, but it does help smooth out the signal and identify when one driver is performing unusually compared to their teammate
# rough laps like outbraking or errors are analogous to liquidity shocks or microstructure noise or news events in finance that can cause temporary divergence in a pairs trade, but we expect the drivers to revert back to their typical performance over time, just like we expect the price spread between two stocks to revert to its mean after a shock.
def spread_zscore(spread: pd.Series, window: int = 10) -> pd.Series:
    # Rolling z-score of the spread: how many std devs from the rolling mean
    # finance: this is the entry/exit signal in a pairs trade (enter at ±2, exit at 0)
    # in F1: tells you when one driver is anomalously slow vs their teammate
    mean = spread.rolling(window, min_periods=2).mean()
    std = spread.rolling(window, min_periods=2).std()
    return (spread - mean) / std

# the lag that is found in normal stocks is sort of negligent here due to radio instructions being almost immediate
def pairs_summary(laps: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    # Full Engle-Granger pipeline: step 1 OLS hedge ratio, step 2 ADF on residuals
    hedge = estimate_hedge_ratio(laps, driver_a, driver_b)
    spread_df = compute_spread(laps, driver_a, driver_b, hedge_ratio=hedge["hedge_ratio"])
    spread = spread_df["spread"]
    adf = adf_test(spread)

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "laps_compared": len(spread),
        "hedge_ratio": hedge["hedge_ratio"],   # beta from OLS — how much faster one driver is relative to the other
        "r_squared": hedge["r_squared"],
        "spread_mean": round(spread.mean(), 3),
        "spread_std": round(spread.std(), 3),
        "adf_stat": adf["adf_stat"],
        "p_value": adf["p_value"],
        "is_stationary": adf["is_stationary"],
    }
