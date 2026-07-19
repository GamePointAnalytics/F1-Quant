import pandas as pd
import numpy as np


def fit_stint_deg(stint_df: pd.DataFrame) -> dict:
    # Two-factor OLS: LapTime_s ~ TyreLife + LapNumber
    # TyreLife captures tyre degradation; LapNumber proxies fuel load (cars lose ~1.6kg/lap)
    # Separating them gives a cleaner deg signal — otherwise fuel effect bleeds into the tyre beta
    # finance equivalent: multi-factor model isolating systematic risk sources
    #
    # Some stints (e.g. around red flags) have TyreLife/Compound recorded as
    # NaN for the whole stint — a design matrix with NaNs makes lstsq's SVD
    # fail to converge rather than just returning a bad fit, so those rows are
    # dropped before checking the minimum-laps guard, not after.
    stint_df = stint_df[stint_df[["TyreLife", "LapNumber", "LapTime_s"]].apply(np.isfinite).all(axis=1)]
    if len(stint_df) < 3:
        return None

    # Design matrix: column of 1s (intercept), TyreLife, LapNumber
    X = np.column_stack([
        np.ones(len(stint_df)),
        stint_df["TyreLife"].values,
        stint_df["LapNumber"].values,
    ])
    y = stint_df["LapTime_s"].values

    # lstsq solves X @ coeffs ≈ y — minimises sum of squared residuals across all factors at once
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta_tyre, beta_fuel = coeffs

    # R² computed manually — lstsq doesn't return it
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "deg_rate": round(beta_tyre, 4),     # seconds lost per lap of tyre age (pure tyre effect)
        "fuel_effect": round(beta_fuel, 4),  # typically negative — car gets faster as fuel burns off
        "base_pace": round(alpha, 3),
        "r_squared": round(r_squared, 4),
        "laps": len(stint_df),
    }


def stint_residuals(stint_df: pd.DataFrame, time_column: str = "LapTime_s") -> pd.Series:
    # What's left of a time column after removing tyre deg + fuel trend — the
    # noise a mean-reversion model (e.g. Ornstein-Uhlenbeck) gets fit to.
    # Same design matrix as fit_stint_deg, sorted chronologically (not by
    # TyreLife) since lag-1 order matters here in a way it didn't for the OLS fit.
    #
    # time_column defaults to the whole lap time, but since the design matrix
    # (TyreLife, LapNumber) is identical for every sector, this same function
    # applied to Sector1Time_s/Sector2Time_s/Sector3Time_s gives residuals that
    # sum exactly to the whole-lap residual (OLS is linear in y) — reused by
    # analysis/sector_noise.py to estimate measurement noise externally.
    stint_df = stint_df[stint_df[["TyreLife", "LapNumber", time_column]].apply(np.isfinite).all(axis=1)]
    if len(stint_df) < 3:
        return pd.Series(dtype=float)

    ordered = stint_df.sort_values("LapNumber")
    X = np.column_stack([
        np.ones(len(ordered)),
        ordered["TyreLife"].values,
        ordered["LapNumber"].values,
    ])
    y = ordered[time_column].values

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeffs
    return pd.Series(residuals, index=ordered["LapNumber"].values)

# Fit degradation profile for a specific driver
def driver_deg_profile(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    # Fit degradation curve for each stint a driver ran
    # Returns one row per stint: compound, deg rate, base pace, R²
    driver_laps = laps[laps["Driver"] == driver].copy()
    rows = []

    # Group by stint and compound — each stint is a separate degradation profile, and different compounds degrade differently
    for (stint, compound), group in driver_laps.groupby(["Stint", "Compound"]):
        #fit_stint_deg can return None if there are fewer than 3 laps in the stint, so we check for that and skip those stints
        result = fit_stint_deg(group.sort_values("TyreLife"))
        if result is None:
            continue
        rows.append({
            "Driver": driver,
            "Stint": stint,
            "Compound": compound,
            #**result unpacks the dictionary returned by fit_stint_deg into individual columns in the output DataFrame
            **result,
        })

    return pd.DataFrame(rows)

# Compute degradation profile for all drivers using a DataFrame of laps
def field_deg_rates(laps: pd.DataFrame) -> pd.DataFrame:
    # Fit degradation curves for all drivers and return a summary DataFrame
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
