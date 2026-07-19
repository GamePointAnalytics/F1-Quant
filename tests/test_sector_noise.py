import numpy as np
import pandas as pd

from analysis.sector_noise import estimate_measurement_noise
from analysis.kalman_lap_state import fit_kalman_params_fixed_R
from analysis.tyre_deg import stint_residuals


def simulate_sector_triples(shared_var: float, idio_vars: tuple, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, np.sqrt(shared_var), n)
    s1 = shared + rng.normal(0, np.sqrt(idio_vars[0]), n)
    s2 = shared + rng.normal(0, np.sqrt(idio_vars[1]), n)
    s3 = shared + rng.normal(0, np.sqrt(idio_vars[2]), n)
    return {"s1": pd.Series(s1), "s2": pd.Series(s2), "s3": pd.Series(s3)}


def test_estimate_measurement_noise_recovers_known_idiosyncratic_variance():
    true_shared, true_idio = 0.5, (0.1, 0.15, 0.2)
    stints = [simulate_sector_triples(true_shared, true_idio, 300, seed=i) for i in range(3)]
    result = estimate_measurement_noise(stints)
    assert result is not None
    true_r = sum(true_idio)
    assert abs(result["R_hat"] - true_r) / true_r < 0.3


def test_estimate_measurement_noise_near_zero_when_sectors_perfectly_correlated():
    # no idiosyncratic noise at all -> all variance is shared -> R_hat ~ 0
    stints = [simulate_sector_triples(shared_var=1.0, idio_vars=(0.0, 0.0, 0.0), n=300, seed=1)]
    result = estimate_measurement_noise(stints)
    assert result is not None
    assert result["R_hat"] < 0.05


def test_estimate_measurement_noise_floors_at_zero():
    # tiny sample -> sampling noise can easily push a pairwise covariance above
    # a sector's own variance; idio variances must never go negative
    stints = [simulate_sector_triples(shared_var=0.3, idio_vars=(0.01, 0.01, 0.01), n=12, seed=2)]
    result = estimate_measurement_noise(stints)
    assert result is not None
    assert result["idio_var_s1"] >= 0
    assert result["idio_var_s2"] >= 0
    assert result["idio_var_s3"] >= 0


def test_estimate_measurement_noise_none_below_min_laps():
    stints = [simulate_sector_triples(0.5, (0.1, 0.1, 0.1), n=3, seed=3)]
    assert estimate_measurement_noise(stints) is None


def test_stint_residuals_custom_time_column_matches_default_behavior():
    stint_df = pd.DataFrame({
        "LapNumber": range(1, 8),
        "TyreLife": range(1, 8),
        "LapTime_s": np.linspace(90.0, 91.0, 7),
        "Sector1Time_s": np.linspace(30.0, 30.5, 7),
    })
    default_result = stint_residuals(stint_df)
    custom_result = stint_residuals(stint_df, time_column="Sector1Time_s")
    assert not default_result.empty
    assert not custom_result.empty
    assert list(default_result.index) == list(custom_result.index)


def test_fit_kalman_params_fixed_R_never_moves_R():
    rng = np.random.default_rng(4)
    fixed_R = 0.25
    phi, Q = 0.5, 0.05
    x_prev = 0.0
    z = np.zeros(60)
    for t in range(60):
        x_prev = phi * x_prev + rng.normal(0, np.sqrt(Q))
        z[t] = x_prev + rng.normal(0, np.sqrt(fixed_R))
    stints = [pd.Series(z)]

    params = fit_kalman_params_fixed_R(stints, R=fixed_R)
    assert params is not None
    assert params["R"] == fixed_R
    assert abs(params["phi"] - phi) < 0.3


def test_fit_kalman_params_fixed_R_none_below_min_laps():
    stints = [pd.Series([0.1, 0.2, 0.15])]
    assert fit_kalman_params_fixed_R(stints, R=0.1, min_laps=30) is None
