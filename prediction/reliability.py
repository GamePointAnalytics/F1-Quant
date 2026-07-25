import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from analysis.shrinkage import fit_beta_binomial_prior, shrink_beta_binomial_rate

MIN_TEAMS_FOR_SHRINKAGE = 5

# Ergast's `status` field for a classified result. Categorized by cause so a
# team's mechanical DNF rate isn't diluted by crashes (a driver/racing-incident
# risk, not a car reliability signal) or by non-races (DSQ/withdrew/DNS, which
# say nothing about whether the car would have finished).
#
# 'Retired' on its own is Ergast's fallback when no specific cause was coded —
# it's a real DNF but of unknown cause, so it's tracked separately rather than
# silently folded into "mechanical" (which would inflate mechanical rates with
# DNFs that might actually have been driver error or accident damage).
FINISHED = {"Finished", "Lapped"}
ACCIDENT = {"Accident", "Collision", "Collision damage", "Spun off", "Damage"}
MECHANICAL = {
    "Brakes", "Cooling system", "Differential", "Driveshaft", "Electrical",
    "Engine", "Front wing", "Fuel leak", "Fuel pressure", "Fuel pump",
    "Gearbox", "Hydraulics", "Mechanical", "Oil leak", "Power Unit",
    "Power loss", "Puncture", "Rear wing", "Suspension", "Turbo",
    "Undertray", "Vibrations", "Water leak", "Water pressure", "Water pump",
    "Wheel nut",
}
UNKNOWN_DNF = {"Retired"}
EXCLUDED = {"Disqualified", "Withdrew", "Did not start", "Illness"}


def categorize_status(status: str) -> str:
    if status in FINISHED or status.startswith("+"):
        return "finished"
    if status in ACCIDENT:
        return "accident"
    if status in MECHANICAL:
        return "mechanical"
    if status in UNKNOWN_DNF:
        return "unknown_dnf"
    if status in EXCLUDED:
        return "excluded"
    return "unknown_dnf"  # unseen status string — treat as DNF of unknown cause, don't silently drop


def team_reliability_rates(
    results: pd.DataFrame,
    target_categories: frozenset = frozenset({"mechanical"}),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Per-team DNF rate from historical results, with a Wilson score confidence
    interval — team-race sample sizes are small enough (~40-70 starts over a
    few seasons) that a bare point estimate overstates precision.

    `target_categories` controls what counts as the DNF being modeled:
      {"mechanical"}                -> strict cause-attributed failures only.
        Ergast's cause labeling got much coarser starting in 2023 (most DNFs
        that season are just "Retired" with no cause), so this rate is only
        trustworthy when fit on seasons with rich cause labels (2021-2022) —
        and can't be meaningfully validated against a 2023 holdout, because
        2023 barely has any positively-labeled examples to check against.
      {"mechanical", "unknown_dnf"} -> broader "car didn't finish, for any
        non-accident reason" — the only version of this rate that a 2023
        holdout can actually evaluate, since it also counts unlabeled
        "Retired" results as a DNF instead of discarding them.
    """
    df = results.copy()
    df["category"] = df["status"].apply(categorize_status)
    df = df[df["category"] != "excluded"]

    rows = []
    for team, group in df.groupby("constructorId"):
        starts = len(group)
        mechanical = int((group["category"] == "mechanical").sum())
        accident = int((group["category"] == "accident").sum())
        unknown_dnf = int((group["category"] == "unknown_dnf").sum())
        finished = int((group["category"] == "finished").sum())
        target = int(group["category"].isin(target_categories).sum())

        rate = target / starts if starts > 0 else float("nan")
        ci_low, ci_high = proportion_confint(target, starts, alpha=alpha, method="wilson") if starts > 0 else (float("nan"), float("nan"))

        rows.append({
            "team": team,
            "starts": starts,
            "finished": finished,
            "mechanical_dnf": mechanical,
            "accident_dnf": accident,
            "unknown_dnf": unknown_dnf,
            "dnf_rate": round(rate, 4),
            "wilson_ci_low": round(ci_low, 4),
            "wilson_ci_high": round(ci_high, 4),
        })

    return pd.DataFrame(rows).sort_values("dnf_rate", ascending=False).reset_index(drop=True)


def predict_dnf_probability(rates: pd.DataFrame, team: str, fallback_rate: float) -> float:
    """Per-team rate if we've seen enough of that team's races, else the field-wide average.

    This is a discrete, step-function version of shrinkage — full trust in the
    team's own rate above the cutoff, full fallback below it. See
    team_reliability_rates_shrunk() for the continuous version, which pulls
    every team's rate toward the population by an amount proportional to its
    own sample size rather than switching at a hard threshold.
    """
    match = rates[rates["team"] == team]
    if match.empty or match.iloc[0]["starts"] < 5:
        return fallback_rate
    return match.iloc[0]["dnf_rate"]


def team_reliability_rates_shrunk(
    results: pd.DataFrame,
    target_categories: frozenset = frozenset({"mechanical"}),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Same per-team rates as team_reliability_rates(), plus a `dnf_rate_shrunk`
    column: each team's rate pulled toward the population mean by an amount
    proportional to how little data that team has (Beta-Binomial
    empirical-Bayes shrinkage, analysis.shrinkage.fit_beta_binomial_prior).
    Fits one population prior across all teams at once, then applies it
    per-team — a continuous alternative to predict_dnf_probability()'s
    hard `starts < 5` cutoff.
    """
    rates = team_reliability_rates(results, target_categories=target_categories, alpha=alpha)
    if len(rates) < MIN_TEAMS_FOR_SHRINKAGE:
        rates["dnf_rate_shrunk"] = rates["dnf_rate"]
        return rates

    target_counts = (rates["dnf_rate"] * rates["starts"]).round().astype(int)
    prior = fit_beta_binomial_prior(target_counts.values, rates["starts"].values)
    rates["dnf_rate_shrunk"] = [
        round(shrink_beta_binomial_rate(t, s, prior["alpha"], prior["beta"]), 4)
        for t, s in zip(target_counts, rates["starts"])
    ]
    return rates


def predict_dnf_probability_shrunk(rates_shrunk: pd.DataFrame, team: str, fallback_rate: float) -> float:
    """Like predict_dnf_probability(), but reading the continuously-shrunk rate."""
    match = rates_shrunk[rates_shrunk["team"] == team]
    if match.empty:
        return fallback_rate
    return match.iloc[0]["dnf_rate_shrunk"]
