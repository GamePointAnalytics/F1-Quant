"""
Fixes the Q/R degeneracy found in backtesting/kalman_lap_validation.py (R
collapsed to ~0 for 18/21 drivers because a single noisy residual series can't
separately identify process vs measurement noise) by estimating R externally
from sector-time cross-correlation (analysis/sector_noise.py) instead of
letting the Kalman MLE guess it from the same series it's already fitting.

Same 2023 season split as the prior two backtests (first 15 races train, last
7 test, all cached locally — no new downloads). Five one-step-ahead forecasts
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

RESULTS_DIR = Path(__file__).parent / "results"


def collect_driver_sector_stints(races: list[pd.DataFrame], driver: str) -> list[dict]:
    stints = []
    for laps in races:
        stints.extend(driver_sector_residual_stints(laps, driver))
    return stints


def main():
    year = 2023
    train_races = 15

    schedule = get_season_races(year)
    print(f"Season {year}: {len(schedule)} races, train on first {train_races}, "
          f"test on remaining {len(schedule) - train_races}\n")

    print("Loading cached race laps...")
    all_laps = []
    for race in schedule:
        try:
            all_laps.append(get_accurate_laps(year, race["event"]))
        except Exception as e:
            print(f"  [{race['event']}] SKIPPED — load failed: {e}")
            all_laps.append(None)

    train_laps = [l for l in all_laps[:train_races] if l is not None]
    test_laps = [l for l in all_laps[train_races:] if l is not None]
    drivers = sorted(set().union(*[set(l["Driver"].unique()) for l in train_laps]))

    rows = []
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

        rows.append({
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

    results = pd.DataFrame(rows).sort_values("driver").reset_index(drop=True)
    if results.empty:
        print("No results produced — not enough held-out data.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"kalman_sector_validation_{year}.csv"
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")

    print("\n=== R: sector-derived vs free-MLE-estimated (the headline comparison) ===")
    print(results[["driver", "R_sector", "R_free_mle"]].to_string(index=False))

    print("\n=== Field-wide mean RMSE (lower is better) ===")
    for col in ["rmse_kalman_fixed_R", "rmse_kalman_free_R", "rmse_static_ou", "rmse_persistence", "rmse_zero"]:
        print(f"  {col}: {results[col].mean():.4f}")

    n = len(results)
    print(f"\nDrivers where kalman_fixed_R beats kalman_free_R: {(results['rmse_kalman_fixed_R'] < results['rmse_kalman_free_R']).sum()}/{n}")
    print(f"Drivers where kalman_fixed_R beats static_ou:      {(results['rmse_kalman_fixed_R'] < results['rmse_static_ou']).sum()}/{n}")
    print(f"Drivers where kalman_fixed_R beats persistence:    {(results['rmse_kalman_fixed_R'] < results['rmse_persistence']).sum()}/{n}")
    print(f"Drivers where kalman_fixed_R beats zero:           {(results['rmse_kalman_fixed_R'] < results['rmse_zero']).sum()}/{n}")


if __name__ == "__main__":
    main()
