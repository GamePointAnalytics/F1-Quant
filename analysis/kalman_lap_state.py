import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis.lap_consistency import driver_residual_stints

# Guards the stationary-variance initialization (Q/(1-phi^2)) from blowing up
# when phi is optimizer-proposed very close to +-1 mid-search.
_VARIANCE_FLOOR = 1e-6


def _stationary_variance(phi: float, Q: float) -> float:
    return Q / max(1 - phi ** 2, _VARIANCE_FLOOR)


def kalman_pass(residual_stints: list[pd.Series], phi: float, Q: float, R: float) -> pd.DataFrame:
    """
    Scalar Kalman filter over:
        state:       x_t = phi*x_t-1 + w_t,  w_t ~ N(0, Q)   -- true pace deviation
        observation: z_t = x_t + v_t,          v_t ~ N(0, R)   -- the tyre_deg residual

    Resets x_hat/P to the process's own stationary variance at the start of
    every stint — a pit stop is a genuine discontinuity (fresh tyres, reset
    fuel-trend baseline), not a real lap-to-lap transition, so nothing about
    a driver's estimated state should carry across it. Same boundary rule as
    analysis.lap_consistency.build_lag_pairs.

    Returns one row per lap: the filtered state estimate (x_hat), its
    posterior variance (P), the innovation (observed - predicted) and its
    variance, and the Kalman gain actually used that lap.
    """
    rows = []
    for stint_index, series in enumerate(residual_stints):
        if len(series) == 0:
            continue

        x_est = 0.0
        P = _stationary_variance(phi, Q)

        for lap_number, z in series.items():
            x_pred = phi * x_est
            P_pred = phi ** 2 * P + Q

            innovation = z - x_pred
            innovation_var = P_pred + R
            gain = P_pred / innovation_var

            x_est = x_pred + gain * innovation
            P = (1 - gain) * P_pred

            rows.append({
                "stint_index": stint_index,
                "lap_number": lap_number,
                "observation": z,
                "x_hat": x_est,
                "P": P,
                "innovation": innovation,
                "innovation_var": innovation_var,
                "kalman_gain": gain,
            })

    return pd.DataFrame(rows)


def neg_log_likelihood(params: np.ndarray, residual_stints: list[pd.Series]) -> float:
    phi, log_Q, log_R = params
    if not (-0.999 < phi < 0.999):
        return 1e10  # keep the optimizer out of the unstable/undefined region without hard bounds

    Q, R = np.exp(log_Q), np.exp(log_R)
    result = kalman_pass(residual_stints, phi, Q, R)
    if result.empty:
        return 1e10

    innovation = result["innovation"].values
    innovation_var = result["innovation_var"].values
    return float(0.5 * np.sum(np.log(2 * np.pi * innovation_var) + innovation ** 2 / innovation_var))


def fit_kalman_params(residual_stints: list[pd.Series], min_laps: int = 30) -> dict | None:
    """
    Fits (phi, Q, R) by maximum likelihood — the standard way state-space
    models are estimated: run the filter forward with candidate parameters,
    score the Gaussian prediction-error (innovation) likelihood, and let
    scipy find the params that maximize it. Q/R are optimized in log-space
    so they can't be proposed negative without needing hard bounds.
    """
    total_laps = sum(len(s) for s in residual_stints)
    if total_laps < min_laps:
        return None

    all_values = np.concatenate([s.values for s in residual_stints if len(s) > 0])
    var_guess = float(np.var(all_values)) if len(all_values) > 1 else 1.0
    x0 = [0.3, np.log(var_guess / 2 + 1e-6), np.log(var_guess / 2 + 1e-6)]

    result = minimize(
        neg_log_likelihood,
        x0=x0,
        args=(residual_stints,),
        method="Nelder-Mead",
    )

    phi, log_Q, log_R = result.x
    return {
        "phi": round(float(phi), 4),
        "Q": round(float(np.exp(log_Q)), 6),
        "R": round(float(np.exp(log_R)), 6),
        "log_likelihood": round(float(-result.fun), 4),
        "converged": bool(result.success),
        "n_laps": total_laps,
    }


def neg_log_likelihood_fixed_R(params: np.ndarray, residual_stints: list[pd.Series], R: float) -> float:
    phi, log_Q = params
    if not (-0.999 < phi < 0.999):
        return 1e10

    Q = np.exp(log_Q)
    result = kalman_pass(residual_stints, phi, Q, R)
    if result.empty:
        return 1e10

    innovation = result["innovation"].values
    innovation_var = result["innovation_var"].values
    return float(0.5 * np.sum(np.log(2 * np.pi * innovation_var) + innovation ** 2 / innovation_var))


def fit_kalman_params_fixed_R(residual_stints: list[pd.Series], R: float, min_laps: int = 30) -> dict | None:
    """
    Same MLE as fit_kalman_params, but R is held fixed at an externally-derived
    value (e.g. analysis.sector_noise.driver_measurement_noise) instead of being
    a free parameter — this is the fix for the Q/R degeneracy: a single noisy
    residual series alone can't separately identify how much of its variance is
    real state movement (Q) vs measurement noise (R), so R has to come from
    outside this likelihood entirely.
    """
    total_laps = sum(len(s) for s in residual_stints)
    if total_laps < min_laps:
        return None

    all_values = np.concatenate([s.values for s in residual_stints if len(s) > 0])
    var_guess = float(np.var(all_values)) if len(all_values) > 1 else 1.0
    remaining_var = max(var_guess - R, 1e-6)
    x0 = [0.3, np.log(remaining_var)]

    result = minimize(
        neg_log_likelihood_fixed_R,
        x0=x0,
        args=(residual_stints, R),
        method="Nelder-Mead",
    )

    phi, log_Q = result.x
    return {
        "phi": round(float(phi), 4),
        "Q": round(float(np.exp(log_Q)), 6),
        "R": round(float(R), 6),
        "log_likelihood": round(float(-result.fun), 4),
        "converged": bool(result.success),
        "n_laps": total_laps,
    }


def driver_kalman_profile(laps: pd.DataFrame, driver: str) -> dict | None:
    stints = driver_residual_stints(laps, driver)
    params = fit_kalman_params(stints)
    if params is None:
        return None
    params["driver"] = driver
    params["stints"] = len(stints)
    return params


def season_kalman_profile(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver: fitted phi/Q/R, ranked by R (lowest = cleanest per-lap pace signal)."""
    rows = []
    for driver in laps["Driver"].unique():
        profile = driver_kalman_profile(laps, driver)
        if profile is not None:
            rows.append(profile)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("R").reset_index(drop=True)
