import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".fastf1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

LAP_COLUMNS = [
    "Driver", "Team", "LapNumber", "LapTime",
    "Stint", "Compound", "TyreLife",
    "Position", "IsAccurate",
]


def load_session(year: int, event: str, session_type: str = "R"):
    session = fastf1.get_session(year, event, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    return session


def get_race_laps(year: int, event: str) -> pd.DataFrame:
    session = load_session(year, event, "R")
    laps = session.laps[LAP_COLUMNS].copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["LapTime_s"])
    return laps


def get_accurate_laps(year: int, event: str) -> pd.DataFrame:
    laps = get_race_laps(year, event)
    return laps[laps["IsAccurate"]].reset_index(drop=True)


def get_driver_laps(year: int, event: str, driver: str) -> pd.DataFrame:
    laps = get_accurate_laps(year, event)
    return laps[laps["Driver"] == driver].reset_index(drop=True)


def get_teammate_laps(year: int, event: str, driver_a: str, driver_b: str) -> pd.DataFrame:
    laps = get_accurate_laps(year, event)
    mask = laps["Driver"].isin([driver_a, driver_b])
    return laps[mask].reset_index(drop=True)
