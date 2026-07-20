"""
Walk-forward validation of team DNF rates across season-level folds (train on
an expanding set of past seasons, test on the next season the model never
saw) instead of one fixed 2021-2022-train/2023-test split — so the reported
Brier score comes with a sense of how much it varies year to year, and so a
season-specific data-quality quirk (like 2023's coarse retirement labels)
doesn't get mistaken for a permanent property of the model.

Extends the season range from [2021, 2022, 2023] to [2021..2025] specifically
so there's enough seasons for 3 folds instead of 1 — also lets the 2023
label-coarseness finding get checked against whether 2024/2025 look the same
or different, rather than assumed.

Runs the comparison under two different definitions of "DNF":
  mechanical   -> strict cause-attributed failures only (Engine, Gearbox, ...).
                  Only as reliable as that season's Ergast label granularity —
                  2023 was nearly unpopulated for this target; see the printed
                  per-season status breakdown for whether that's still true.
  car_dnf      -> mechanical + unlabeled "Retired" folded together (excludes
                  Accident/Collision, a driver/racing-incident risk, not a car
                  reliability one). Well-populated regardless of label
                  granularity, so this is the version actually worth trusting.

Compares three predictors per team-race for each target:
  team_model    -> that team's historical rate from the fold's training seasons
  naive_global  -> the fold's field-wide training-season average, applied to every team alike
  always_zero   -> "no one ever fails" (the Brier score of doing nothing)

Usage:
    python -m prediction.run_reliability_validation
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.results_loader import get_season_results
from prediction.reliability import (
    categorize_status,
    team_reliability_rates,
    predict_dnf_probability,
)
from backtesting.rolling_cv import expanding_folds

SEASONS = [2021, 2022, 2023, 2024, 2025]
TARGETS = {
    "mechanical": frozenset({"mechanical"}),
    "car_dnf": frozenset({"mechanical", "unknown_dnf"}),
}


def brier_score(predicted: pd.Series, actual: pd.Series) -> float:
    return float(((predicted - actual) ** 2).mean())


def evaluate_target(train: pd.DataFrame, test: pd.DataFrame, target_categories: frozenset):
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

    metrics = {
        "n_scoreable": len(scoreable),
        "n_positive": int(actual.sum()),
        "brier_team_model": brier_score(team_pred, actual),
        "brier_naive_global": brier_score(naive_pred, actual),
        "brier_always_zero": brier_score(zero_pred, actual),
    }
    return metrics, rates


def main():
    print(f"Fetching season results for {SEASONS}...")
    all_results = {year: get_season_results(year) for year in SEASONS}
    for year, df in all_results.items():
        print(f"  {year}: {len(df)} team-race entries")

    folds = expanding_folds(SEASONS, min_train=2, test_size=1)
    print(f"\n{len(folds)} walk-forward season folds\n")

    fold_rows = {label: [] for label in TARGETS}
    last_car_dnf_rates = None

    for fold_index, (train_years, test_years) in enumerate(folds):
        test_year = test_years[0]
        train = pd.concat([all_results[y] for y in train_years], ignore_index=True)
        test = all_results[test_year].copy()
        test["category"] = test["status"].apply(categorize_status)

        print(f"--- Fold {fold_index}: train {train_years} -> test {test_year} ---")
        print(f"  {test_year} status breakdown: {test['category'].value_counts().to_dict()}")

        for label, categories in TARGETS.items():
            metrics, rates = evaluate_target(train, test, categories)
            metrics["fold"] = fold_index
            metrics["test_year"] = test_year
            fold_rows[label].append(metrics)
            print(f"  [{label}] scoreable={metrics['n_scoreable']} positive={metrics['n_positive']} "
                  f"team_model={metrics['brier_team_model']:.5f} "
                  f"naive_global={metrics['brier_naive_global']:.5f} "
                  f"always_zero={metrics['brier_always_zero']:.5f}")
            if label == "car_dnf":
                last_car_dnf_rates = rates
        print()

    print("=== Across-fold summary: mean ± std of Brier score over folds (lower is better) ===")
    for label in TARGETS:
        fold_df = pd.DataFrame(fold_rows[label])
        print(f"\n[{label}]")
        for col in ["brier_team_model", "brier_naive_global", "brier_always_zero"]:
            print(f"  {col}: {fold_df[col].mean():.5f} ± {fold_df[col].std():.5f}  "
                  f"(per-fold: {fold_df[col].round(5).tolist()})")

    print("\n(Team reliability rates below are from the final fold's training window, car_dnf target)")
    print(last_car_dnf_rates.to_string(index=False))


if __name__ == "__main__":
    main()
