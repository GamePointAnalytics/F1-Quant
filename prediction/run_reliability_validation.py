"""
Trains team DNF rates on 2021-2022, then scores those rates against actual
2023 race outcomes (a season the model never saw) using Brier score (mean
squared error between predicted probability and the 0/1 actual outcome —
lower is better, 0 is a perfect prediction).

Runs the comparison under two different definitions of "DNF", because they
turn out to behave very differently against a 2023 holdout:

  mechanical   -> strict cause-attributed failures only (Engine, Gearbox, ...).
                  Ergast's cause labeling got much coarser starting in 2023
                  (nearly every 2023 DNF is just "Retired" with no cause), so
                  this target is almost unpopulated in the holdout — expect
                  the comparison here to be close to statistical noise.
  car_dnf      -> mechanical + unlabeled "Retired" folded together (excludes
                  Accident/Collision, which is a driver/racing-incident risk,
                  not a car reliability one). Well-populated in every season
                  regardless of how specific Ergast's labels are that year,
                  so this is the version actually worth trusting.

Compares three predictors per team-race for each target:
  team_model    -> that team's historical rate from 2021-2022
  naive_global  -> the field-wide 2021-2022 average applied to every team alike
  always_zero   -> "no one ever fails" (the Brier score of doing nothing)

Usage:
    python -m prediction.run_reliability_validation
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.results_loader import get_multi_season_results, get_season_results
from prediction.reliability import (
    categorize_status,
    team_reliability_rates,
    predict_dnf_probability,
)

TARGETS = {
    "mechanical": frozenset({"mechanical"}),
    "car_dnf": frozenset({"mechanical", "unknown_dnf"}),
}


def brier_score(predicted: pd.Series, actual: pd.Series) -> float:
    return float(((predicted - actual) ** 2).mean())


def evaluate_target(train: pd.DataFrame, test: pd.DataFrame, target_categories: frozenset, label: str):
    rates = team_reliability_rates(train, target_categories=target_categories)
    fallback_rate = rates["dnf_rate"].mul(rates["starts"]).sum() / rates["starts"].sum()

    # Only score rows where the outcome is unambiguous under this target:
    # drop non-races (DSQ/withdrew/DNS) always, and for the strict "mechanical"
    # target also drop unknown-cause DNFs since we don't know their true label.
    scoreable = test[test["category"] != "excluded"]
    if target_categories == frozenset({"mechanical"}):
        scoreable = scoreable[scoreable["category"] != "unknown_dnf"]

    actual = scoreable["category"].isin(target_categories).astype(float)
    team_pred = scoreable["constructorId"].apply(
        lambda team: predict_dnf_probability(rates, team, fallback_rate)
    )
    naive_pred = pd.Series(fallback_rate, index=scoreable.index)
    zero_pred = pd.Series(0.0, index=scoreable.index)

    print(f"\n=== Target: {label} ({len(scoreable)} scoreable rows, "
          f"{int(actual.sum())} positive) ===")
    print(f"  team_model:   {brier_score(team_pred, actual):.5f}")
    print(f"  naive_global: {brier_score(naive_pred, actual):.5f}")
    print(f"  always_zero:  {brier_score(zero_pred, actual):.5f}")
    return rates


def main():
    print("Fetching training data (2021-2022)...")
    train = get_multi_season_results([2021, 2022])
    print(f"  {len(train)} team-race entries")

    print("Fetching test data (2023)...")
    test = get_season_results(2023)
    test = test.copy()
    test["category"] = test["status"].apply(categorize_status)
    print(f"  {len(test)} team-race entries")
    print(f"  2023 status breakdown: {test['category'].value_counts().to_dict()}")

    for label, categories in TARGETS.items():
        rates = evaluate_target(train, test, categories, label)
        if label == "car_dnf":
            print("\n  (rates below are the car_dnf version, used for the simulator)")
            print(rates.to_string(index=False))


if __name__ == "__main__":
    main()
