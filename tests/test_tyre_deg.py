import numpy as np
import pandas as pd

from analysis.tyre_deg import fit_stint_deg, stint_residuals


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
