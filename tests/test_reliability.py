import pandas as pd

from prediction.reliability import (
    categorize_status,
    team_reliability_rates,
    predict_dnf_probability,
)


def test_categorize_status_buckets():
    assert categorize_status("Finished") == "finished"
    assert categorize_status("+1 Lap") == "finished"
    assert categorize_status("Engine") == "mechanical"
    assert categorize_status("Gearbox") == "mechanical"
    assert categorize_status("Accident") == "accident"
    assert categorize_status("Collision damage") == "accident"
    assert categorize_status("Retired") == "unknown_dnf"
    assert categorize_status("Disqualified") == "excluded"
    assert categorize_status("Withdrew") == "excluded"


def test_categorize_status_unseen_string_is_unknown_dnf_not_dropped():
    # A status Ergast has never used in our data yet shouldn't silently vanish
    assert categorize_status("Some Future Failure Mode") == "unknown_dnf"


def test_team_reliability_rates_excludes_dsq_and_withdrew_from_denominator():
    results = pd.DataFrame({
        "constructorId": ["red_bull"] * 5,
        "status": ["Finished", "Finished", "Engine", "Disqualified", "Withdrew"],
    })
    rates = team_reliability_rates(results)
    row = rates[rates["team"] == "red_bull"].iloc[0]
    # starts should be 3 (2 finished + 1 engine DNF), DSQ/withdrew excluded entirely
    assert row["starts"] == 3
    assert row["mechanical_dnf"] == 1
    assert abs(row["dnf_rate"] - round(1 / 3, 4)) < 1e-9


def test_team_reliability_rates_wilson_ci_widens_with_small_sample():
    small_sample = pd.DataFrame({
        "constructorId": ["team_a"] * 4,
        "status": ["Engine", "Finished", "Finished", "Finished"],
    })
    large_sample = pd.DataFrame({
        "constructorId": ["team_b"] * 40,
        "status": ["Engine"] * 10 + ["Finished"] * 30,
    })
    small = team_reliability_rates(small_sample).iloc[0]
    large = team_reliability_rates(large_sample).iloc[0]
    # both have a 25% point estimate, but the small-sample CI should be much wider
    small_width = small["wilson_ci_high"] - small["wilson_ci_low"]
    large_width = large["wilson_ci_high"] - large["wilson_ci_low"]
    assert small_width > large_width


def test_team_reliability_rates_broad_target_includes_unknown_dnf():
    results = pd.DataFrame({
        "constructorId": ["team_a"] * 4,
        "status": ["Engine", "Retired", "Accident", "Finished"],
    })
    narrow = team_reliability_rates(results, target_categories=frozenset({"mechanical"})).iloc[0]
    broad = team_reliability_rates(results, target_categories=frozenset({"mechanical", "unknown_dnf"})).iloc[0]
    # narrow: only the Engine DNF counts (1/4); broad: Engine + Retired both count (2/4)
    assert abs(narrow["dnf_rate"] - 0.25) < 1e-9
    assert abs(broad["dnf_rate"] - 0.5) < 1e-9


def test_predict_dnf_probability_falls_back_for_sparse_team():
    rates = pd.DataFrame([
        {"team": "well_sampled", "starts": 40, "dnf_rate": 0.10},
        {"team": "barely_sampled", "starts": 2, "dnf_rate": 0.50},
    ])
    assert predict_dnf_probability(rates, "well_sampled", fallback_rate=0.2) == 0.10
    # only 2 starts -> not enough signal to trust the team-specific rate over the field average
    assert predict_dnf_probability(rates, "barely_sampled", fallback_rate=0.2) == 0.2
    assert predict_dnf_probability(rates, "never_seen_team", fallback_rate=0.2) == 0.2
