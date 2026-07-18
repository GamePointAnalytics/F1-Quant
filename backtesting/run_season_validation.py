"""
Validates the HMM regime detector's 'safety_car' state against FastF1's official
track-status ground truth, across every race in a season.

Runs two variants per driver per race:
  broken -> model fit on get_accurate_laps() output, exactly how the notebook
            and analysis/regime_detection.py currently do it. IsAccurate=False
            drops every real caution lap before the HMM ever sees one, so this
            variant is expected to score ~0 recall by construction.
  fixed  -> model fit on get_race_laps() output (no IsAccurate filtering), so
            caution laps are actually present for the HMM to learn from.

Usage:
    python -m backtesting.run_season_validation --year 2023
    python -m backtesting.run_season_validation --year 2023 --rounds 1,2,3
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fastf1_loader import get_season_races, get_race_laps, get_accurate_laps
from backtesting.regime_validation import validate_race, aggregate

RESULTS_DIR = Path(__file__).parent / "results"


def run_variant(source_laps: pd.DataFrame, event: str, variant: str) -> pd.DataFrame:
    results = validate_race(source_laps)
    if not results.empty:
        results["event"] = event
        results["variant"] = variant
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--rounds", type=str, default=None, help="comma-separated round numbers, default = full season")
    args = parser.parse_args()

    races = get_season_races(args.year)
    if args.rounds:
        wanted = {int(r) for r in args.rounds.split(",")}
        races = [r for r in races if r["round"] in wanted]

    print(f"Season {args.year}: {len(races)} races queued\n")

    all_results = []
    for race in races:
        event = race["event"]
        t0 = time.time()
        try:
            raw_laps = get_race_laps(args.year, event)
            accurate_laps = raw_laps[raw_laps["IsAccurate"]].reset_index(drop=True)
        except Exception as e:
            print(f"  [{event}] SKIPPED — load failed: {e}")
            continue

        n_caution_laps = (raw_laps["TrackStatus"].astype(str).str.contains("[4567]", regex=True)).sum()

        broken = run_variant(accurate_laps, event, "broken")
        fixed = run_variant(raw_laps, event, "fixed")
        all_results.append(broken)
        all_results.append(fixed)

        elapsed = time.time() - t0
        print(f"  [{event}] {n_caution_laps} raw caution-laps in season data | "
              f"broken recall={aggregate(broken)['recall'] if not broken.empty else 'n/a'} | "
              f"fixed recall={aggregate(fixed)['recall'] if not fixed.empty else 'n/a'} | "
              f"{elapsed:.1f}s")

    if not all_results:
        print("No results produced.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"regime_validation_{args.year}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved per-driver-race results to {out_path}")

    print("\n=== Season summary ===")
    for variant in ["broken", "fixed"]:
        subset = combined[combined["variant"] == variant]
        agg = aggregate(subset)
        print(f"{variant:>7}: precision={agg['precision']} recall={agg['recall']} "
              f"f1={agg['f1']} tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} tn={agg['tn']} "
              f"(scored {agg['driver_races_scored']} driver-races)")


if __name__ == "__main__":
    main()
