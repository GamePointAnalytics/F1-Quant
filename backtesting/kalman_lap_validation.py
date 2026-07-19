"""
Extends backtesting/lap_consistency_validation.py's train/test comparison
with the Kalman-filtered version of the OU model, to test whether explicitly
separating process noise (Q) from measurement noise (R) closes the gap that
made the static OU model lose to the trivial "predict 0" baseline for almost
every driver.

Same 2023 season split as before (first 15 races train, last 7 test, all
cached locally already). Four forecasts of the next lap's residual, scored
by RMSE:
  kalman      -> phi * x_hat_t   (x_hat_t is the FILTERED state estimate,
                                   not the raw noisy observation)
  static_ou   -> phi_static * x_t  (the original single-OLS-pass model)
  persistence -> x_t             (naive: no reversion)
  zero        -> 0               (naive: full/instant reversion)

Usage:
    python -m backtesting.kalman_lap_validation
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fastf1_loader import get_season_races, get_accurate_laps
from analysis.lap_consistency import driver_residual_stints, fit_ou_params, build_lag_pairs
from analysis.kalman_lap_state import fit_kalman_params, kalman_pass
from backtesting.lap_consistency_validation import rmse, collect_driver_stints

RESULTS_DIR = Path(__file__).parent / "results"


def build_kalman_forecast_pairs(residual_stints: list[pd.Series], phi: float, Q: float, R: float):
    """Like lap_consistency.build_lag_pairs, but the 't' side is the filtered
    state estimate x_hat_t instead of the raw noisy observation x_t."""
    filtered = kalman_pass(residual_stints, phi, Q, R)
    if filtered.empty:
        return np.array([]), np.array([])

    x_hat_t, x_actual_t1 = [], []
    for _, group in filtered.groupby("stint_index"):
        if len(group) < 2:
            continue
        x_hat_t.append(group["x_hat"].values[:-1])
        x_actual_t1.append(group["observation"].values[1:])
    if not x_hat_t:
        return np.array([]), np.array([])
    return np.concatenate(x_hat_t), np.concatenate(x_actual_t1)


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

        kalman_params = fit_kalman_params(train_stints)
        static_params = fit_ou_params(train_stints)
        if kalman_params is None or static_params is None:
            continue

        x_t, x_t1 = build_lag_pairs(test_stints)
        kalman_x_hat, kalman_actual = build_kalman_forecast_pairs(
            test_stints, kalman_params["phi"], kalman_params["Q"], kalman_params["R"]
        )
        if len(x_t) < 5 or len(kalman_x_hat) < 5:
            continue

        rows.append({
            "driver": driver,
            "train_pairs": static_params["n_pairs"],
            "test_pairs": len(x_t),
            "phi_kalman": kalman_params["phi"],
            "Q": kalman_params["Q"],
            "R": kalman_params["R"],
            "phi_static": static_params["phi"],
            "rmse_kalman": round(rmse(kalman_params["phi"] * kalman_x_hat, kalman_actual), 4),
            "rmse_static_ou": round(rmse(static_params["phi"] * x_t, x_t1), 4),
            "rmse_persistence": round(rmse(x_t, x_t1), 4),
            "rmse_zero": round(rmse(np.zeros_like(x_t), x_t1), 4),
        })

    results = pd.DataFrame(rows).sort_values("driver").reset_index(drop=True)
    if results.empty:
        print("No results produced — not enough held-out data.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"kalman_lap_validation_{year}.csv"
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")

    print("\n=== Field-wide mean RMSE (lower is better) ===")
    print(f"  kalman:      {results['rmse_kalman'].mean():.4f}")
    print(f"  static_ou:   {results['rmse_static_ou'].mean():.4f}")
    print(f"  persistence: {results['rmse_persistence'].mean():.4f}")
    print(f"  zero:        {results['rmse_zero'].mean():.4f}")

    n = len(results)
    print(f"\nDrivers where kalman beats static_ou:   {(results['rmse_kalman'] < results['rmse_static_ou']).sum()}/{n}")
    print(f"Drivers where kalman beats persistence: {(results['rmse_kalman'] < results['rmse_persistence']).sum()}/{n}")
    print(f"Drivers where kalman beats zero:        {(results['rmse_kalman'] < results['rmse_zero']).sum()}/{n}")


if __name__ == "__main__":
    main()
