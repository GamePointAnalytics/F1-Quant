import numpy as np
import pandas as pd
import pytest

from analysis.tyre_deg import driver_deg_profile, fit_race_fuel_effect, fit_stint_deg, stint_residuals


def _stint_with_nan_tyrelife(n_valid: int = 5, n_nan: int = 35) -> pd.DataFrame:
    # Mirrors the real 2023 Canadian GP data (Tsunoda, Stint 3): a long run of
    # laps with TyreLife/Compound never recorded (e.g. around a red flag),
    # mixed in with otherwise-normal laps.
    valid = pd.DataFrame({
        "LapNumber": range(1, n_valid + 1),
        "TyreLife": range(1, n_valid + 1),
        "LapTime_s": np.linspace(90.0, 91.0, n_valid),
    })
    nan_rows = pd.DataFrame({
        "LapNumber": range(n_valid + 1, n_valid + n_nan + 1),
        "TyreLife": [np.nan] * n_nan,
        "LapTime_s": np.linspace(77.0, 77.5, n_nan),
    })
    return pd.concat([valid, nan_rows], ignore_index=True)


def test_fit_stint_deg_does_not_crash_on_nan_tyrelife():
    # Previously raised numpy.linalg.LinAlgError: SVD did not converge
    result = fit_stint_deg(_stint_with_nan_tyrelife())
    assert result is not None
    assert result["laps"] == 5  # only the valid rows should be fit


def test_fit_stint_deg_returns_none_when_all_rows_are_nan():
    all_nan = _stint_with_nan_tyrelife(n_valid=0, n_nan=10)
    assert fit_stint_deg(all_nan) is None


def test_stint_residuals_does_not_crash_on_nan_tyrelife():
    residuals = stint_residuals(_stint_with_nan_tyrelife())
    assert len(residuals) == 5
    assert residuals.notna().all()


def test_stint_residuals_empty_when_all_rows_are_nan():
    all_nan = _stint_with_nan_tyrelife(n_valid=0, n_nan=10)
    assert stint_residuals(all_nan).empty


def _synthetic_multi_stint_laps(true_deg_rate=0.10, true_fuel_effect=-0.06, seed=0) -> pd.DataFrame:
    # TyreLife resets to 1 at each pit stop while LapNumber keeps counting up,
    # so within any one stint TyreLife - LapNumber is a constant — the design
    # matrix [1, TyreLife, LapNumber] used by a single-stint fit has rank 2,
    # not 3, and the split between deg_rate/fuel_effect is arbitrary. Pooling
    # laps across multiple stints (this fixture) is what breaks that
    # collinearity and lets fit_race_fuel_effect recover the true fuel slope.
    rng = np.random.default_rng(seed)
    rows = []
    lap = 1
    for stint_id, stint_len in enumerate([20, 20, 18]):
        for tyre_life in range(1, stint_len + 1):
            laptime = 90.0 + true_deg_rate * tyre_life + true_fuel_effect * lap + rng.normal(0, 0.01)
            rows.append({
                "Driver": "VER",
                "Stint": stint_id,
                "Compound": "MEDIUM",
                "TyreLife": tyre_life,
                "LapNumber": lap,
                "LapTime_s": laptime,
            })
            lap += 1
    return pd.DataFrame(rows)


def test_fit_race_fuel_effect_recovers_true_fuel_slope():
    laps = _synthetic_multi_stint_laps()
    beta_fuel = fit_race_fuel_effect(laps)
    assert beta_fuel == pytest.approx(-0.06, abs=0.01)


def test_fit_race_fuel_effect_none_for_single_stint():
    laps = _synthetic_multi_stint_laps()
    assert fit_race_fuel_effect(laps[laps["Stint"] == 0]) is None


def test_fit_stint_deg_with_fixed_beta_fuel_isolates_deg_rate():
    # Without a fixed beta_fuel, a single stint can't separate tyre deg from
    # fuel burn — this checks that passing the race-level estimate in does.
    laps = _synthetic_multi_stint_laps()
    beta_fuel = fit_race_fuel_effect(laps)
    stint0 = laps[laps["Stint"] == 0].sort_values("TyreLife")
    result = fit_stint_deg(stint0, beta_fuel=beta_fuel)
    assert result["deg_rate"] == pytest.approx(0.10, abs=0.01)
    assert result["fuel_effect"] == pytest.approx(-0.06, abs=0.01)


def test_driver_deg_profile_uses_race_level_fuel_effect():
    laps = _synthetic_multi_stint_laps()
    profile = driver_deg_profile(laps, "VER")
    assert len(profile) == 3
    assert profile["deg_rate"].apply(lambda r: r == pytest.approx(0.10, abs=0.01)).all()
    assert profile["fuel_effect"].nunique() == 1  # same fixed beta_fuel reused across all stints
