import numpy as np
import pandas as pd

from analysis.kalman_lap_state import (
    kalman_pass,
    fit_kalman_params,
    _stationary_variance,
)


def simulate_state_space(phi: float, Q: float, R: float, n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x_prev = 0.0
    z = np.zeros(n)
    for t in range(n):
        x_prev = phi * x_prev + rng.normal(0, np.sqrt(Q))
        z[t] = x_prev + rng.normal(0, np.sqrt(R))
    return pd.Series(z)


def test_fit_kalman_params_recovers_known_phi_and_total_variance():
    true_phi, true_Q, true_R = 0.6, 0.05, 0.3
    stints = [simulate_state_space(true_phi, true_Q, true_R, 30, seed=i) for i in range(15)]
    params = fit_kalman_params(stints)
    assert params is not None
    assert abs(params["phi"] - true_phi) < 0.25
    assert params["Q"] > 0 and params["R"] > 0
    # Q and R can trade off against each other somewhat (identification is easier for
    # their sum than for each individually with limited data), so check the
    # combined stationary variance instead of demanding each one lands exactly.
    true_total_var = true_Q / (1 - true_phi ** 2) + true_R
    fitted_total_var = params["Q"] / (1 - params["phi"] ** 2) + params["R"]
    assert abs(fitted_total_var - true_total_var) / true_total_var < 0.6


def test_fit_kalman_params_none_below_min_laps():
    stints = [pd.Series([0.1, 0.2, 0.15])]
    assert fit_kalman_params(stints, min_laps=30) is None


def test_posterior_variance_converges_within_a_long_stint():
    phi, Q, R = 0.7, 0.02, 0.3
    long_stint = simulate_state_space(phi, Q, R, 200, seed=1)
    result = kalman_pass([long_stint], phi, Q, R)
    late_P = result["P"].iloc[-20:]
    assert late_P.std() < 1e-3  # should have settled at the Riccati fixed point by now


def test_gain_approaches_one_when_measurement_noise_negligible():
    # R -> 0: almost no measurement noise -> trust each new observation almost entirely
    phi, Q, R = 0.5, 0.1, 1e-6
    stint = simulate_state_space(phi, Q, R, 50, seed=2)
    result = kalman_pass([stint], phi, Q, R)
    assert result["kalman_gain"].iloc[-1] > 0.99


def test_gain_approaches_zero_when_process_noise_negligible():
    # Q -> 0: the true state barely moves on its own -> distrust noisy new observations
    phi, Q, R = 0.5, 1e-6, 1.0
    stint = simulate_state_space(phi, Q, R, 50, seed=3)
    result = kalman_pass([stint], phi, Q, R)
    assert result["kalman_gain"].iloc[-1] < 0.1


def test_kalman_pass_resets_state_between_stints():
    # stint A ends elevated (around 5); stint B is a fresh stint sitting at 0.
    # If state leaked across the boundary, the filter would predict something
    # pulled toward stint A's level for stint B's first lap instead of 0.
    stint_a = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
    stint_b = pd.Series([0.0, 0.0, 0.0])
    result = kalman_pass([stint_a, stint_b], phi=0.5, Q=0.1, R=0.5)
    first_b_row = result[result["stint_index"] == 1].iloc[0]
    assert abs(first_b_row["x_hat"]) < 0.01


def test_stationary_variance_matches_formula():
    assert abs(_stationary_variance(phi=0.5, Q=1.0) - (1.0 / (1 - 0.25))) < 1e-9
