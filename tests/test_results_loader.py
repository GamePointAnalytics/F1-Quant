from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests.exceptions import RequestException

from data import results_loader


def _fake_response(rows: int):
    if rows == 0:
        return MagicMock(content=[])
    df = pd.DataFrame({
        "driverId": [f"d{i}" for i in range(rows)],
        "constructorId": [f"c{i}" for i in range(rows)],
        "grid": range(rows),
        "position": range(rows),
        "positionText": range(rows),
        "status": ["Finished"] * rows,
    })
    return MagicMock(content=[df])


def test_get_race_results_retries_transient_failure_then_succeeds():
    mock_ergast = MagicMock()
    mock_ergast.get_race_results.side_effect = [RequestException("429"), _fake_response(20)]
    with patch.object(results_loader, "_ergast", mock_ergast), patch("time.sleep"):
        result = results_loader.get_race_results(2024, 1, max_retries=4, backoff_seconds=0)
    assert len(result) == 20
    assert mock_ergast.get_race_results.call_count == 2


def test_get_race_results_raises_after_exhausting_retries():
    mock_ergast = MagicMock()
    mock_ergast.get_race_results.side_effect = RequestException("429")
    with patch.object(results_loader, "_ergast", mock_ergast), patch("time.sleep"):
        with pytest.raises(RuntimeError):
            results_loader.get_race_results(2024, 1, max_retries=3, backoff_seconds=0)
    assert mock_ergast.get_race_results.call_count == 3


def test_get_season_results_stops_on_genuine_empty_round_not_error():
    # 2 real races then a clean empty response -> season is over, no error involved
    mock_ergast = MagicMock()
    mock_ergast.get_race_results.side_effect = [_fake_response(20), _fake_response(20), _fake_response(0)]
    with patch.object(results_loader, "_ergast", mock_ergast):
        season = results_loader.get_season_results(2024, max_round=5)
    assert len(season) == 40


def test_get_season_results_stops_gracefully_on_repeated_transient_failure_without_truncating_silently():
    # 1 real race, then every subsequent request fails transiently (simulates
    # the 429 rate-limit scenario that previously silently truncated a season
    # to 1 race with no visible error) -> should stop after exhausting
    # retries on round 2, keeping what it already collected, not raise.
    mock_ergast = MagicMock()
    mock_ergast.get_race_results.side_effect = [_fake_response(20)] + [RequestException("429")] * 10
    with patch.object(results_loader, "_ergast", mock_ergast), patch("time.sleep"):
        season = results_loader.get_season_results(2024, max_round=5)
    assert len(season) == 20  # only the one race that actually succeeded
