"""
Fixes the Q/R degeneracy found in backtesting/kalman_lap_validation.py (R
collapsed to ~0 for 18/21 drivers because a single noisy residual series can't
separately identify process vs measurement noise) by estimating R externally
from sector-time cross-correlation (analysis/sector_noise.py) instead of
letting the Kalman MLE guess it from the same series it's already fitting.

Walk-forward across several expanding folds (not one fixed split) over the
season, all cached locally — no new downloads. Five one-step-ahead forecasts
of the next lap's residual, scored by RMSE:
  kalman_fixed_R -> phi * x_hat_t, with R fixed from sector times
  kalman_free_R  -> phi * x_hat_t, with R freely (mis-)estimated (prior pass)
  static_ou      -> phi_static * x_t (single-OLS-pass model, no filtering)
  persistence    -> x_t   (naive: no reversion)
  zero           -> 0     (naive: full/instant reversion)

Usage:
    python -m backtesting.kalman_sector_validation
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fastf1_loader import get_season_races, get_accurate_laps
from analysis.lap_consistency import fit_ou_params, build_lag_pairs
from analysis.kalman_lap_state import fit_kalman_params, fit_kalman_params_fixed_R
from analysis.sector_noise import driver_sector_residual_stints, estimate_measurement_noise
from backtesting.lap_consistency_validation import rmse, collect_driver_stints
from backtesting.kalman_lap_validation import build_kalman_forecast_pairs
from backtesting.rolling_cv import expanding_folds

RESULTS_DIR = Path(__file__).parent / "results"


def collect_driver_sector_stints(races: list[pd.DataFrame], driver: str) -> list[dict]:
    stints = []
    for laps in races:
        stints.extend(driver_sector_residual_stints(laps, driver))
    return stints


def main():
    year = 2023
    min_train_races = 10
    test_size = 3

    schedule = get_season_races(year)
    print(f"Season {year}: {len(schedule)} races\n")

    print("Loading cached race laps...")
    all_laps = []
    for race in schedule:
        try:
            all_laps.append(get_accurate_laps(year, race["event"]))
        except Exception as e:
            print(f"  [{race['event']}] SKIPPED — load failed: {e}")
    all_laps = [l for l in all_laps if l is not None]

    folds = expanding_folds(all_laps, min_train=min_train_races, test_size=test_size)
    print(f"{len(folds)} walk-forward folds (min_train={min_train_races} races, test_size={test_size} races)\n")

    all_rows = []
    for fold_index, (train_laps, test_laps) in enumerate(folds):
        drivers = sorted(set().union(*[set(l["Driver"].unique()) for l in train_laps]))
        for driver in drivers:
            train_stints = collect_driver_stints(train_laps, driver)
            test_stints = collect_driver_stints(test_laps, driver)

            train_sector_stints = collect_driver_sector_stints(train_laps, driver)
            noise_estimate = estimate_measurement_noise(train_sector_stints)

            static_params = fit_ou_params(train_stints)
            free_params = fit_kalman_params(train_stints)
            if static_params is None or free_params is None or noise_estimate is None:
                continue

            r_hat = noise_estimate["R_hat"]
            fixed_params = fit_kalman_params_fixed_R(train_stints, R=r_hat)
            if fixed_params is None:
                continue

            x_t, x_t1 = build_lag_pairs(test_stints)
            fixed_x_hat, fixed_actual = build_kalman_forecast_pairs(
                test_stints, fixed_params["phi"], fixed_params["Q"], fixed_params["R"]
            )
            free_x_hat, free_actual = build_kalman_forecast_pairs(
                test_stints, free_params["phi"], free_params["Q"], free_params["R"]
            )
            if len(x_t) < 5 or len(fixed_x_hat) < 5:
                continue

            all_rows.append({
                "fold": fold_index,
                "driver": driver,
                "test_pairs": len(x_t),
                "R_sector": r_hat,
                "R_free_mle": free_params["R"],
                "phi_fixed_R": fixed_params["phi"],
                "Q_fixed_R": fixed_params["Q"],
                "rmse_kalman_fixed_R": round(rmse(fixed_params["phi"] * fixed_x_hat, fixed_actual), 4),
                "rmse_kalman_free_R": round(rmse(free_params["phi"] * free_x_hat, free_actual), 4),
                "rmse_static_ou": round(rmse(static_params["phi"] * x_t, x_t1), 4),
                "rmse_persistence": round(rmse(x_t, x_t1), 4),
                "rmse_zero": round(rmse(np.zeros_like(x_t), x_t1), 4),
            })

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("No results produced — not enough held-out data.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"kalman_sector_validation_{year}.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved per-fold, per-driver results to {out_path}\n")

    print("=== R: sector-derived vs free-MLE-estimated, mean across folds (the headline comparison) ===")
    print(results.groupby("driver")[["R_sector", "R_free_mle"]].mean().round(4).to_string())

    print("\n=== Per-fold field-wide mean RMSE ===")
    cols = ["rmse_kalman_fixed_R", "rmse_kalman_free_R", "rmse_static_ou", "rmse_persistence", "rmse_zero"]
    print(results.groupby("fold")[cols].mean().round(4).to_string())

    print("\n=== Across-fold summary: mean ± std of the per-fold means (lower is better) ===")
    for col in cols:
        per_fold_mean = results.groupby("fold")[col].mean()
        print(f"  {col}: {per_fold_mean.mean():.4f} ± {per_fold_mean.std():.4f}  "
              f"(per-fold: {per_fold_mean.round(4).tolist()})")

    n = len(results)
    print(f"\n(driver, fold) pairs where kalman_fixed_R beats kalman_free_R: {(results['rmse_kalman_fixed_R'] < results['rmse_kalman_free_R']).sum()}/{n}")
    print(f"(driver, fold) pairs where kalman_fixed_R beats static_ou:      {(results['rmse_kalman_fixed_R'] < results['rmse_static_ou']).sum()}/{n}")
    print(f"(driver, fold) pairs where kalman_fixed_R beats persistence:    {(results['rmse_kalman_fixed_R'] < results['rmse_persistence']).sum()}/{n}")
    print(f"(driver, fold) pairs where kalman_fixed_R beats zero:           {(results['rmse_kalman_fixed_R'] < results['rmse_zero']).sum()}/{n}")


if __name__ == "__main__":
    main()
