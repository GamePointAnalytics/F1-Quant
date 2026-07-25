"""
Walk-forward validation of each driver's OU/AR(1) mean-reversion coefficient
(phi): refit on an expanding training window, evaluate on the next block of
held-out races, repeat across several folds — instead of one fixed split —
so the reported RMSE comes with a sense of how much it varies fold to fold,
not just a single number from one lucky/unlucky split.

Four forecasts of the next lap's residual, scored by RMSE:
  ou_model    -> phi * x_t          (raw per-driver OLS fit)
  ou_shrunk   -> phi_shrunk * x_t   (phi pulled toward the training fold's
                                      population mean, weighted by how
                                      uncertain each driver's own phi is —
                                      analysis.lap_consistency.shrink_phi_across_drivers)
  persistence -> x_t                (assume no reversion — a random walk)
  zero        -> 0                  (assume the residual is already noise
                                      around zero with no useful structure)

Each fold does two passes over its training window: first fit every driver's
raw (phi, se_phi) independently, then fit ONE population prior across all of
them and shrink each driver's phi using it — shrinkage needs the whole
fold's drivers gathered before it can say anything about "the population."

Runs entirely against the season already cached locally — no new network calls.

Usage:
    python -m backtesting.lap_consistency_validation
    python -m backtesting.lap_consistency_validation --min-train-races 10 --test-size 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fastf1_loader import get_season_races, get_accurate_laps
from analysis.lap_consistency import (
    driver_residual_stints,
    fit_ou_params,
    build_lag_pairs,
    shrink_phi_across_drivers,
)
from backtesting.rolling_cv import expanding_folds

RESULTS_DIR = Path(__file__).parent / "results"


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def collect_driver_stints(races: list[pd.DataFrame], driver: str) -> list[pd.Series]:
    stints = []
    for laps in races:
        stints.extend(driver_residual_stints(laps, driver))
    return stints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--min-train-races", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=3)
    args = parser.parse_args()

    schedule = get_season_races(args.year)
    print(f"Season {args.year}: {len(schedule)} races\n")

    print("Loading cached race laps...")
    all_laps = []
    for race in schedule:
        try:
            all_laps.append(get_accurate_laps(args.year, race["event"]))
        except Exception as e:
            print(f"  [{race['event']}] SKIPPED — load failed: {e}")
    all_laps = [l for l in all_laps if l is not None]

    folds = expanding_folds(all_laps, min_train=args.min_train_races, test_size=args.test_size)
    print(f"{len(folds)} walk-forward folds "
          f"(min_train={args.min_train_races} races, test_size={args.test_size} races)\n")

    all_rows = []
    for fold_index, (train_laps, test_laps) in enumerate(folds):
        drivers = sorted(set().union(*[set(l["Driver"].unique()) for l in train_laps]))

        # Pass 1: each driver's raw OU fit on this fold's training window
        driver_params = {}
        param_rows = []
        for driver in drivers:
            train_stints = collect_driver_stints(train_laps, driver)
            params = fit_ou_params(train_stints)
            if params is None:
                continue
            driver_params[driver] = params
            param_rows.append({"driver": driver, "phi": params["phi"], "se_phi": params["se_phi"]})

        if not param_rows:
            continue

        # Pass 2: one population prior across all of this fold's drivers, then shrink each
        shrunk = shrink_phi_across_drivers(pd.DataFrame(param_rows)).set_index("driver")

        # Pass 3: evaluate both the raw and shrunk phi on this fold's held-out races
        for driver in shrunk.index:
            test_stints = collect_driver_stints(test_laps, driver)
            x_t, x_t1 = build_lag_pairs(test_stints)
            if len(x_t) < 5:
                continue

            params = driver_params[driver]
            phi_shrunk = float(shrunk.loc[driver, "phi_shrunk"])

            ou_pred = params["phi"] * x_t
            ou_shrunk_pred = phi_shrunk * x_t
            persistence_pred = x_t
            zero_pred = np.zeros_like(x_t)

            all_rows.append({
                "fold": fold_index,
                "train_races": len(train_laps),
                "driver": driver,
                "test_pairs": len(x_t),
                "phi": params["phi"],
                "phi_shrunk": round(phi_shrunk, 4),
                "theta": params["theta"],
                "sigma": params["sigma"],
                "rmse_ou_model": round(rmse(ou_pred, x_t1), 4),
                "rmse_ou_shrunk": round(rmse(ou_shrunk_pred, x_t1), 4),
                "rmse_persistence": round(rmse(persistence_pred, x_t1), 4),
                "rmse_zero": round(rmse(zero_pred, x_t1), 4),
            })

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("No results produced — not enough held-out data.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"lap_consistency_validation_{args.year}.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved per-fold, per-driver results to {out_path}\n")

    print("=== Per-driver, per-fold detail (phi/phi_shrunk/theta/sigma + RMSE) ===")
    print(results.to_string(index=False))

    cols = ["rmse_ou_model", "rmse_ou_shrunk", "rmse_persistence", "rmse_zero"]
    print("\n=== Per-fold field-wide mean RMSE ===")
    print(results.groupby("fold")[cols].mean().round(4).to_string())

    print("\n=== Across-fold summary: mean ± std of the per-fold means (lower is better) ===")
    for col in cols:
        per_fold_mean = results.groupby("fold")[col].mean()
        print(f"  {col}: {per_fold_mean.mean():.4f} ± {per_fold_mean.std():.4f}  "
              f"(per-fold: {per_fold_mean.round(4).tolist()})")

    n = len(results)
    print(f"\n(driver, fold) pairs where ou_shrunk beats ou_model:    "
          f"{(results['rmse_ou_shrunk'] < results['rmse_ou_model']).sum()}/{n}")
    print(f"(driver, fold) pairs where ou_shrunk beats persistence: "
          f"{(results['rmse_ou_shrunk'] < results['rmse_persistence']).sum()}/{n}")
    print(f"(driver, fold) pairs where ou_shrunk beats zero:        "
          f"{(results['rmse_ou_shrunk'] < results['rmse_zero']).sum()}/{n}")


if __name__ == "__main__":
    main()
