"""
Trains each driver's OU/AR(1) mean-reversion coefficient (phi) on the first
part of the 2023 season, then checks whether it actually predicts held-out
lap-time residuals better than two naive baselines:

  ou_model    -> phi * x_t          (this model's forecast)
  persistence -> x_t                (assume no reversion — a random walk)
  zero        -> 0                  (assume the residual is already noise
                                      around zero with no useful structure)

Runs entirely against the 2023 season already cached locally — no new
network calls.

Usage:
    python -m backtesting.lap_consistency_validation
    python -m backtesting.lap_consistency_validation --train-races 15
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
)

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
    parser.add_argument("--train-races", type=int, default=15)
    args = parser.parse_args()

    schedule = get_season_races(args.year)
    print(f"Season {args.year}: {len(schedule)} races, "
          f"train on first {args.train_races}, test on remaining {len(schedule) - args.train_races}\n")

    print("Loading cached race laps...")
    all_laps = []
    for race in schedule:
        try:
            all_laps.append(get_accurate_laps(args.year, race["event"]))
        except Exception as e:
            print(f"  [{race['event']}] SKIPPED — load failed: {e}")
            all_laps.append(None)

    train_laps = [l for l in all_laps[:args.train_races] if l is not None]
    test_laps = [l for l in all_laps[args.train_races:] if l is not None]

    drivers = sorted(set().union(*[set(l["Driver"].unique()) for l in train_laps]))

    rows = []
    for driver in drivers:
        train_stints = collect_driver_stints(train_laps, driver)
        test_stints = collect_driver_stints(test_laps, driver)

        params = fit_ou_params(train_stints)
        x_t, x_t1 = build_lag_pairs(test_stints)
        if params is None or len(x_t) < 5:
            continue

        ou_pred = params["phi"] * x_t
        persistence_pred = x_t
        zero_pred = np.zeros_like(x_t)

        rows.append({
            "driver": driver,
            "train_pairs": params["n_pairs"],
            "test_pairs": len(x_t),
            "phi": params["phi"],
            "theta": params["theta"],
            "sigma": params["sigma"],
            "rmse_ou_model": round(rmse(ou_pred, x_t1), 4),
            "rmse_persistence": round(rmse(persistence_pred, x_t1), 4),
            "rmse_zero": round(rmse(zero_pred, x_t1), 4),
        })

    results = pd.DataFrame(rows).sort_values("driver").reset_index(drop=True)
    if results.empty:
        print("No results produced — not enough held-out data.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"lap_consistency_validation_{args.year}.csv"
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")

    print("\n=== Field-wide mean RMSE (lower is better) ===")
    print(f"  ou_model:    {results['rmse_ou_model'].mean():.4f}")
    print(f"  persistence: {results['rmse_persistence'].mean():.4f}")
    print(f"  zero:        {results['rmse_zero'].mean():.4f}")

    beats_persistence = (results["rmse_ou_model"] < results["rmse_persistence"]).sum()
    beats_zero = (results["rmse_ou_model"] < results["rmse_zero"]).sum()
    print(f"\nDrivers where ou_model beats persistence: {beats_persistence}/{len(results)}")
    print(f"Drivers where ou_model beats zero:        {beats_zero}/{len(results)}")


if __name__ == "__main__":
    main()
