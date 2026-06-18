import pandas as pd
import numpy as np
from scipy.stats import linregress


def fit_stint_deg(stint_df: pd.DataFrame) -> dict:
    # OLS regression: LapTime_s ~ TyreLife for a single stint
    # slope (beta) = seconds lost per additional lap of tyre age — the degradation rate
    # finance equivalent: beta in a factor model — how much does one unit of "age" move the output?
    if len(stint_df) < 3:
        return None

    x = stint_df["TyreLife"].values
    y = stint_df["LapTime_s"].values
    slope, intercept, r_value, _, _ = linregress(x, y)

    return {
        "deg_rate": round(slope, 4),       # seconds lost per lap of tyre age
        "base_pace": round(intercept, 3),  # predicted lap time at TyreLife=0 (fresh tyre)
        "r_squared": round(r_value ** 2, 4),
        "laps": len(stint_df),
    }


def driver_deg_profile(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    # Fit degradation curve for each stint a driver ran
    # Returns one row per stint: compound, deg rate, base pace, R²
    driver_laps = laps[laps["Driver"] == driver].copy()
    rows = []

    for (stint, compound), group in driver_laps.groupby(["Stint", "Compound"]):
        result = fit_stint_deg(group.sort_values("TyreLife"))
        if result is None:
            continue
        rows.append({
            "Driver": driver,
            "Stint": stint,
            "Compound": compound,
            **result,
        })

    return pd.DataFrame(rows)


def field_deg_rates(laps: pd.DataFrame) -> pd.DataFrame:
    # Deg profile for every driver — lets you compare who manages tyres best
    # finance equivalent: cross-sectional factor exposure — who has the highest/lowest beta?
    profiles = []
    for driver in laps["Driver"].unique():
        profile = driver_deg_profile(laps, driver)
        profiles.append(profile)

    all_profiles = pd.concat(profiles, ignore_index=True)
    return all_profiles.sort_values(["Compound", "deg_rate"]).reset_index(drop=True)


def crossover_lap(deg_rate: float, pit_loss_s: float = 22.0) -> float:
    # How many laps until accumulated deg cost exceeds the pit stop time loss?
    # Formula: pit_loss / deg_rate
    # If deg_rate=0.08 and pit_loss=22s → crossover at 275 laps (never pit — tyre is fine)
    # If deg_rate=0.40 and pit_loss=22s → crossover at 55 laps (matches a typical race stint)
    # finance equivalent: break-even holding period — how long until the carry cost exceeds the entry cost?
    if deg_rate <= 0:
        return float("inf")
    return round(pit_loss_s / deg_rate, 1)


def deg_summary(laps: pd.DataFrame, driver: str, pit_loss_s: float = 22.0) -> pd.DataFrame:
    # Full pipeline for one driver: deg rate per stint + crossover lap per stint
    profile = driver_deg_profile(laps, driver)
    if profile.empty:
        return profile
    profile["crossover_lap"] = profile["deg_rate"].apply(
        lambda r: crossover_lap(r, pit_loss_s)
    )
    return profile
