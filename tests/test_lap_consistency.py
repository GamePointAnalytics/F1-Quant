import numpy as np
import pandas as pd

from analysis.lap_consistency import (
    fit_ou_params,
    stint_stationarity,
    build_lag_pairs,
)
from backtesting.lap_consistency_validation import rmse


def simulate_ar1_series(phi: float, n: int, sigma_eps: float = 0.1, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma_eps)
    return pd.Series(x)


def test_fit_ou_params_recovers_known_phi():
    true_phi = 0.5
    # pool many stints of realistic length, mirroring how a real driver's races get grouped
    stints = [simulate_ar1_series(true_phi, 25, seed=i) for i in range(20)]
    params = fit_ou_params(stints)
    assert params is not None
    assert abs(params["phi"] - true_phi) < 0.15
    assert params["theta"] > 0  # 0 < phi < 1 -> real mean reversion, theta well-defined


def test_fit_ou_params_returns_none_below_min_pairs():
    stints = [pd.Series([0.1, 0.2, 0.15])]  # only 2 lag pairs
    assert fit_ou_params(stints, min_pairs=10) is None


def test_fit_ou_params_theta_sigma_nan_for_random_walk():
    # phi ~ 1 (random walk, no reversion) -> theta/sigma undefined, not a fabricated number
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.normal(0, 0.1, 200)))
    params = fit_ou_params([x])
    assert params is not None
    if not (0 < params["phi"] < 1):
        assert params["theta"] != params["theta"]  # NaN != NaN
        assert params["sigma"] != params["sigma"]


def test_lag_pairs_never_cross_stint_boundary():
    stint_a = pd.Series([1.0, 2.0, 3.0])
    stint_b = pd.Series([100.0, 200.0, 300.0])
    x_t, x_t1 = build_lag_pairs([stint_a, stint_b])
    pairs = set(zip(x_t.tolist(), x_t1.tolist()))
    assert (3.0, 100.0) not in pairs  # the cross-boundary pair that would be wrong
    assert pairs == {(1.0, 2.0), (2.0, 3.0), (100.0, 200.0), (200.0, 300.0)}


def test_stint_stationarity_ranks_mean_reverting_series_below_random_walk():
    stationary = simulate_ar1_series(0.3, 50, seed=2)
    random_walk = pd.Series(np.cumsum(np.random.default_rng(3).normal(0, 0.1, 50)))
    result = stint_stationarity([stationary, random_walk])
    # a genuinely mean-reverting series should score a lower (more significant) p-value
    # than an actual random walk, which is the whole premise this module depends on
    assert result.loc[0, "p_value"] < result.loc[1, "p_value"]


def test_stint_stationarity_skips_short_stints():
    short = pd.Series([0.1, 0.2, 0.15])  # well below MIN_STINT_LAPS_FOR_ADF
    result = stint_stationarity([short])
    assert result.empty


def test_rmse_zero_for_perfect_predictions():
    actual = np.array([1.0, 2.0, 3.0])
    assert rmse(actual, actual) == 0.0


def test_rmse_matches_hand_computed_value():
    predicted = np.array([0.0, 0.0, 0.0])
    actual = np.array([1.0, 2.0, 3.0])
    # sqrt(mean([1, 4, 9])) = sqrt(14/3)
    expected = np.sqrt((1 + 4 + 9) / 3)
    assert abs(rmse(predicted, actual) - expected) < 1e-9
