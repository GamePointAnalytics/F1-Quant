import numpy as np
import pandas as pd

from analysis.tyre_deg import stint_residuals
from analysis.pairs import adf_test
from analysis.shrinkage import fit_normal_hierarchical_prior, shrink_normal_estimate

MIN_DRIVERS_FOR_SHRINKAGE = 5

# ADF has weak power with only ~15-25 laps per stint (the usual stint length) —
# ~8 laps is a floor below which the test isn't meaningful at all, not a
# guarantee of a reliable result even above it. Treat stationarity fractions
# from this module as directional, not a precise measurement.
MIN_STINT_LAPS_FOR_ADF = 8


def driver_residual_stints(laps: pd.DataFrame, driver: str) -> list[pd.Series]:
    """
    One residual series per stint for a driver — never concatenated across
    stints, since a pit stop is a real discontinuity (fresh tyres, fuel
    correction reset) and treating it as a normal lap-to-lap transition would
    corrupt any lag-1 mean-reversion fit.
    """
    driver_laps = laps[laps["Driver"] == driver]
    stints = []
    for _, group in driver_laps.groupby(["Stint", "Compound"]):
        residuals = stint_residuals(group)
        if len(residuals) >= 3:
            stints.append(residuals)
    return stints


def stint_stationarity(stint_residual_list: list[pd.Series]) -> pd.DataFrame:
    """Per-stint ADF test on the residual series — is there real mean reversion here at all?"""
    rows = []
    for i, series in enumerate(stint_residual_list):
        if len(series) < MIN_STINT_LAPS_FOR_ADF:
            continue
        result = adf_test(series)
        result["stint_index"] = i
        result["laps"] = len(series)
        rows.append(result)
    return pd.DataFrame(rows)


def build_lag_pairs(stint_residual_list: list[pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    x_t, x_t1 = [], []
    for series in stint_residual_list:
        values = series.values
        if len(values) < 2:
            continue
        x_t.append(values[:-1])
        x_t1.append(values[1:])
    if not x_t:
        return np.array([]), np.array([])
    return np.concatenate(x_t), np.concatenate(x_t1)


def fit_ou_params(stint_residual_list: list[pd.Series], min_pairs: int = 10) -> dict | None:
    """
    Pools lag-1 (x_t, x_t+1) pairs across every stint for one driver (never
    across a stint boundary — see driver_residual_stints) and fits the AR(1)
    regression x_t+1 = phi*x_t + c, the same regression structure as the ADF
    test in analysis/pairs.py, just solved for phi directly instead of testing
    whether phi=1.

    Converts the discrete-time AR(1) coefficient to continuous-time OU params:
      theta = -ln(phi)   -> mean-reversion speed (higher = snaps back faster)
      sigma = sqrt(Var(eps) * 2*theta / (1 - phi^2))  -> OU volatility

    theta/sigma are only defined for 0 < phi < 1 (real mean reversion). If phi
    falls outside that range (no reversion, or oscillation), phi/intercept are
    still returned but theta/sigma come back as NaN rather than a nonsense number.
    """
    x_t, x_t1 = build_lag_pairs(stint_residual_list)
    n = len(x_t)
    if n < min_pairs:
        return None

    X = np.column_stack([np.ones(n), x_t])
    coeffs, _, _, _ = np.linalg.lstsq(X, x_t1, rcond=None)
    intercept, phi = coeffs
    resid = x_t1 - X @ coeffs
    resid_var = np.var(resid, ddof=2)

    # Standard OLS coefficient-variance formula: Var(coeffs) = resid_var * (X'X)^-1.
    # phi is the second column of X, so its variance is the [1,1] entry — this is
    # what shrink_phi_across_drivers (analysis/shrinkage.py) needs to know how
    # much to trust this driver's own phi vs. the population.
    xtx_inv = np.linalg.inv(X.T @ X)
    se_phi = np.sqrt(resid_var * xtx_inv[1, 1])

    if 0 < phi < 1:
        theta = -np.log(phi)
        sigma = np.sqrt(resid_var * 2 * theta / (1 - phi ** 2))
    else:
        theta = float("nan")
        sigma = float("nan")

    return {
        "phi": round(float(phi), 4),
        "se_phi": round(float(se_phi), 4),
        "intercept": round(float(intercept), 4),
        "theta": round(float(theta), 4) if theta == theta else theta,
        "sigma": round(float(sigma), 4) if sigma == sigma else sigma,
        "resid_std": round(float(np.sqrt(resid_var)), 4),
        "n_pairs": n,
    }


def season_consistency_profile(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver: OU mean-reversion speed + noise scale, ranked by sigma (lowest = most consistent)."""
    rows = []
    for driver in laps["Driver"].unique():
        stints = driver_residual_stints(laps, driver)
        params = fit_ou_params(stints)
        if params is None:
            continue
        params["driver"] = driver
        params["stints"] = len(stints)
        rows.append(params)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)


def shrink_phi_across_drivers(driver_params: pd.DataFrame) -> pd.DataFrame:
    """
    Takes one training fold's worth of per-driver (phi, se_phi) — e.g. rows
    collected by calling fit_ou_params() per driver on a fold's training
    stints — and pulls each driver's phi toward the population mean by an
    amount proportional to how uncertain that driver's own estimate is
    (analysis.shrinkage's Normal-Normal empirical-Bayes model).

    A driver with few lag-pairs (large se_phi) shrinks hard toward the
    population; a driver with hundreds of pairs (small se_phi) barely moves
    from their own OLS estimate. Requires several drivers' worth of rows to
    fit a meaningful population prior — with too few, the population "mean
    and spread" would just be re-describing 1-2 individual noisy estimates.
    """
    if len(driver_params) < MIN_DRIVERS_FOR_SHRINKAGE:
        result = driver_params.copy()
        result["phi_shrunk"] = result["phi"]
        return result

    prior = fit_normal_hierarchical_prior(
        driver_params["phi"].values, driver_params["se_phi"].values
    )
    result = driver_params.copy()
    result["phi_shrunk"] = result.apply(
        lambda row: shrink_normal_estimate(row["phi"], row["se_phi"], prior["mu"], prior["tau2"]),
        axis=1,
    )
    return result
