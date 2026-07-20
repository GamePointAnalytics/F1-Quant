"""
Walk-forward version of the free-R Kalman filter comparison: refit on an
expanding training window, evaluate on the next block of held-out races,
across several folds instead of one fixed split — to test whether explicitly
separating process noise (Q) from measurement noise (R) closes the gap that
made the static OU model lose to the trivial "predict 0" baseline, and how
much that result varies fold to fold rather than trusting a single split.

Four forecasts of the next lap's residual, scored by RMSE:
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
from backtesting.rolling_cv import expanding_folds

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

            all_rows.append({
                "fold": fold_index,
                "driver": driver,
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

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("No results produced — not enough held-out data.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"kalman_lap_validation_{year}.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved per-fold, per-driver results to {out_path}\n")

    print("=== Per-fold field-wide mean RMSE ===")
    cols = ["rmse_kalman", "rmse_static_ou", "rmse_persistence", "rmse_zero"]
    print(results.groupby("fold")[cols].mean().round(4).to_string())

    print("\n=== Across-fold summary: mean ± std of the per-fold means (lower is better) ===")
    for col in cols:
        per_fold_mean = results.groupby("fold")[col].mean()
        print(f"  {col}: {per_fold_mean.mean():.4f} ± {per_fold_mean.std():.4f}  "
              f"(per-fold: {per_fold_mean.round(4).tolist()})")

    n = len(results)
    print(f"\n(driver, fold) pairs where kalman beats static_ou:   {(results['rmse_kalman'] < results['rmse_static_ou']).sum()}/{n}")
    print(f"(driver, fold) pairs where kalman beats persistence: {(results['rmse_kalman'] < results['rmse_persistence']).sum()}/{n}")
    print(f"(driver, fold) pairs where kalman beats zero:        {(results['rmse_kalman'] < results['rmse_zero']).sum()}/{n}")


if __name__ == "__main__":
    main()
