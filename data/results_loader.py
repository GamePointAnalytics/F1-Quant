import time

import pandas as pd
from fastf1.ergast import Ergast
from requests.exceptions import RequestException

_ergast = Ergast()


def get_race_results(year: int, round_number: int, max_retries: int = 4, backoff_seconds: float = 2.0) -> pd.DataFrame:
    """
    One race's classification: driver, team, grid, finishing position, status string.

    Retries with backoff on transient request failures (rate limiting, network
    blips) — these are NOT the same signal as "this round doesn't exist" and
    must not be treated as end-of-season by the caller. Previously a single
    429 from the Ergast mirror during a multi-season fetch was silently
    read as "season over," truncating 2024/2025 to just round 1 (20 rows
    instead of a full ~440-row season) without any visible error.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = _ergast.get_race_results(season=year, round=round_number)
            break
        except RequestException as e:
            last_error = e
            time.sleep(backoff_seconds * (attempt + 1))
    else:
        raise RuntimeError(
            f"Repeated request failures fetching {year} round {round_number}"
        ) from last_error

    if not response.content:
        return pd.DataFrame()
    df = response.content[0]
    if df.empty:
        return df
    df = df[["driverId", "constructorId", "grid", "position", "positionText", "status"]].copy()
    df["year"] = year
    df["round"] = round_number
    return df


def get_season_results(year: int, max_round: int = 26) -> pd.DataFrame:
    """
    All races in a season, concatenated. Ergast doesn't expose a season-wide
    schedule call as cleanly as FastF1's own event schedule, so this just
    walks round numbers until one comes back empty (season over) — a real
    round-doesn't-exist signal, not the same thing as a request that merely
    failed. max_round is just a safety ceiling on the loop, not an assumption
    about calendar length — bumped from 23 to 26 since some seasons (2024+)
    already run 24 rounds.
    """
    rounds = []
    for round_number in range(1, max_round + 1):
        try:
            race = get_race_results(year, round_number)
        except RuntimeError as e:
            print(f"  WARNING: stopping {year} at round {round_number} after repeated failures: {e}")
            break
        if race.empty:
            break
        rounds.append(race)
    if not rounds:
        return pd.DataFrame()
    return pd.concat(rounds, ignore_index=True)


def get_multi_season_results(years: list[int]) -> pd.DataFrame:
    return pd.concat([get_season_results(year) for year in years], ignore_index=True)
