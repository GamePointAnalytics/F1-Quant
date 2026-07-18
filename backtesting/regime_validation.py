import pandas as pd

from analysis.regime_detection import detect_regimes

# FastF1 TrackStatus is a string of every status code seen during a lap, digits
# in no fixed order (e.g. "126" = AllClear + Yellow + VSCDeployed). Codes:
# 1 AllClear, 2 Yellow, 4 SafetyCar, 5 RedFlag, 6 VSCDeployed, 7 VSCEnding.
# Yellow (2) alone is often a local double-waved flag, not a field-wide caution,
# so it's excluded from "caution" — only conditions that slow the whole field count.
CAUTION_CODES = set("4567")


def is_caution_lap(track_status) -> bool:
    if not isinstance(track_status, str):
        return False
    return any(code in track_status for code in CAUTION_CODES)


def ground_truth(laps: pd.DataFrame) -> pd.Series:
    # Per-lap boolean: was this lap run under SC/VSC/red flag, per official FastF1 track status
    return laps["TrackStatus"].apply(is_caution_lap)


def score_predictions(predicted: pd.Series, actual: pd.Series) -> dict:
    # predicted/actual are aligned boolean Series over the same laps
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0
        else float("nan")
    )

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4) if precision == precision else precision,
        "recall": round(recall, 4) if recall == recall else recall,
        "f1": round(f1, 4) if f1 == f1 else f1,
        "actual_caution_laps": int(actual.sum()),
        "total_laps": len(actual),
    }


def validate_driver(source_laps: pd.DataFrame, driver: str) -> dict | None:
    """
    Run HMM regime detection for one driver on `source_laps` and score the
    predicted 'safety_car' state against real track-status ground truth.

    `source_laps` decides what the model is allowed to see. Pass laps still
    containing IsAccurate=False rows (i.e. data/fastf1_loader.get_race_laps,
    not get_accurate_laps) or the model has zero chance of detecting a caution
    period — every real SC/VSC lap gets filtered out before it reaches the HMM.
    """
    driver_laps = source_laps[source_laps["Driver"] == driver]
    if len(driver_laps) < 10:
        return None

    predicted_df = detect_regimes(source_laps, driver)
    if predicted_df.empty:
        return None

    merged = predicted_df.merge(
        driver_laps[["LapNumber", "TrackStatus"]], on="LapNumber", how="left"
    )
    actual = ground_truth(merged)
    predicted = merged["regime"] == "safety_car"

    result = score_predictions(predicted, actual)
    result["driver"] = driver
    return result


def validate_race(source_laps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for driver in source_laps["Driver"].unique():
        result = validate_driver(source_laps, driver)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def aggregate(results: pd.DataFrame) -> dict:
    totals = results[["tp", "fp", "fn", "tn"]].sum()
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else float("nan")
    )
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(totals["tn"]),
        "precision": round(precision, 4) if precision == precision else precision,
        "recall": round(recall, 4) if recall == recall else recall,
        "f1": round(f1, 4) if f1 == f1 else f1,
        "driver_races_scored": len(results),
    }
