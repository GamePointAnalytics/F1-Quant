import pandas as pd
from fastf1.ergast import Ergast

_ergast = Ergast()


def get_race_results(year: int, round_number: int) -> pd.DataFrame:
    """One race's classification: driver, team, grid, finishing position, status string."""
    response = _ergast.get_race_results(season=year, round=round_number)
    if not response.content:
        return pd.DataFrame()
    df = response.content[0]
    if df.empty:
        return df
    df = df[["driverId", "constructorId", "grid", "position", "positionText", "status"]].copy()
    df["year"] = year
    df["round"] = round_number
    return df


def get_season_results(year: int, max_round: int = 23) -> pd.DataFrame:
    """
    All races in a season, concatenated. Ergast doesn't expose a season-wide
    schedule call as cleanly as FastF1's own event schedule, so this just
    walks round numbers until one comes back empty (season over) or errors
    (round doesn't exist yet, e.g. mid-season).
    """
    rounds = []
    for round_number in range(1, max_round + 1):
        try:
            race = get_race_results(year, round_number)
        except Exception:
            break
        if race.empty:
            break
        rounds.append(race)
    if not rounds:
        return pd.DataFrame()
    return pd.concat(rounds, ignore_index=True)


def get_multi_season_results(years: list[int]) -> pd.DataFrame:
    return pd.concat([get_season_results(year) for year in years], ignore_index=True)
