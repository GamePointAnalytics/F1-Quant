import pandas as pd

from backtesting.regime_validation import (
    is_caution_lap,
    ground_truth,
    score_predictions,
    aggregate,
)


def test_is_caution_lap_detects_sc_vsc_red_flag():
    assert is_caution_lap("4")       # full safety car
    assert is_caution_lap("5")       # red flag
    assert is_caution_lap("126")     # AllClear + Yellow + VSCDeployed
    assert is_caution_lap("671")     # order isn't sorted, must not matter
    assert not is_caution_lap("1")   # clean green-flag lap
    assert not is_caution_lap("21")  # AllClear + Yellow only — no field-wide caution
    assert not is_caution_lap(None)
    assert not is_caution_lap(float("nan"))


def test_ground_truth_maps_series():
    laps = pd.DataFrame({"TrackStatus": ["1", "12", "126", "671", "21"]})
    result = ground_truth(laps)
    assert result.tolist() == [False, False, True, True, False]


def test_score_predictions_perfect_match():
    actual = pd.Series([True, True, False, False])
    predicted = pd.Series([True, True, False, False])
    result = score_predictions(predicted, actual)
    assert result["tp"] == 2 and result["fp"] == 0 and result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_score_predictions_zero_recall_when_model_never_fires():
    # This is the exact failure mode of the current pipeline: get_accurate_laps()
    # strips every real caution lap before the HMM sees it, so the model can never
    # predict True on a lap that's actually under caution.
    actual = pd.Series([True, True, False, False, False])
    predicted = pd.Series([False, False, False, False, False])
    result = score_predictions(predicted, actual)
    assert result["tp"] == 0
    assert result["fn"] == 2
    assert result["recall"] == 0.0
    # precision is undefined (no positive predictions at all) -> NaN, not 0 or 1
    assert result["precision"] != result["precision"]


def test_score_predictions_all_false_positives():
    actual = pd.Series([False, False, False])
    predicted = pd.Series([True, True, True])
    result = score_predictions(predicted, actual)
    assert result["precision"] == 0.0
    # recall undefined when there are no actual positives to find
    assert result["recall"] != result["recall"]


def test_aggregate_sums_across_driver_results():
    results = pd.DataFrame([
        {"tp": 2, "fp": 1, "fn": 0, "tn": 10},
        {"tp": 0, "fp": 3, "fn": 2, "tn": 8},
    ])
    agg = aggregate(results)
    assert agg["tp"] == 2 and agg["fp"] == 4 and agg["fn"] == 2 and agg["tn"] == 18
    assert agg["driver_races_scored"] == 2
    # precision = tp / (tp+fp) = 2/6, rounded to 4dp by aggregate()
    assert abs(agg["precision"] - round(2 / 6, 4)) < 1e-9
