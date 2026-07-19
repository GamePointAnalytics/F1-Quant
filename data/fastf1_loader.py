import fastf1
import pandas as pd
from pathlib import Path

# Enable caching for faster data loading, less network requests, and offline access after the first load
# Do so to avoid hitting the API repeatedly and speed up iterations; cache is stored locally in a .fastf1_cache directory
CACHE_DIR = Path(__file__).parent.parent / ".fastf1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Columns we are interested in for lap data analysis from looking at FastF1
# similar to selecting only a few columns from a large DataFrame to reduce memory usage and focus on relevant data for our analysis
LAP_COLUMNS = [
    "Driver", "Team", "LapNumber", "LapTime",
    "Stint", "Compound", "TyreLife",
    "Position", "IsAccurate", "TrackStatus",
    "Sector1Time", "Sector2Time", "Sector3Time",
]

# FastF1 provides a convenient way to load session data, and we can specify what data to load for efficiency
# Load only the session metadata and lap data, skipping telemetry, weather, and messages
def load_session(year: int, event: str, session_type: str = "R"):
    session = fastf1.get_session(year, event, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    return session

# Extract only the relevant columns for our DataFrame
def get_race_laps(year: int, event: str) -> pd.DataFrame:
    session = load_session(year, event, "R")
    laps = session.laps[LAP_COLUMNS].copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds() # np.log(prices / prices.shift(1))
    laps["Sector1Time_s"] = laps["Sector1Time"].dt.total_seconds()
    laps["Sector2Time_s"] = laps["Sector2Time"].dt.total_seconds()
    laps["Sector3Time_s"] = laps["Sector3Time"].dt.total_seconds()
    laps = laps.dropna(subset=["LapTime_s"])
    return laps


def get_accurate_laps(year: int, event: str) -> pd.DataFrame:
    laps = get_race_laps(year, event)
    return laps[laps["IsAccurate"]].reset_index(drop=True)

# Similar to filtering a DataFrame for a specific driver, we can use boolean indexing to get laps for a particular driver or teammates
def get_driver_laps(year: int, event: str, driver: str) -> pd.DataFrame:
    laps = get_accurate_laps(year, event)
    return laps[laps["Driver"] == driver].reset_index(drop=True)

# very similar to pairs trading data, extracting two entities using isin().
def get_teammate_laps(year: int, event: str, driver_a: str, driver_b: str) -> pd.DataFrame:
    laps = get_accurate_laps(year, event)
    mask = laps["Driver"].isin([driver_a, driver_b])
    return laps[mask].reset_index(drop=True)


# Round numbers + names for every points-paying race in a season, skipping test events
# and sprint-only sessions — used to loop the analysis pipeline across a full season
# instead of a single hand-picked race.
def get_season_races(year: int) -> list[dict]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    races = schedule[schedule["EventFormat"] != "testing"]
    return [
        {"round": int(row["RoundNumber"]), "event": row["EventName"]}
        for _, row in races.iterrows()
    ]
