import numpy as np
import pandas as pd

from analysis.tyre_deg import stint_residuals

MIN_SECTOR_LAPS_PER_STINT = 5

SECTOR_COLUMNS = ("Sector1Time_s", "Sector2Time_s", "Sector3Time_s")
SECTOR_KEYS = ("s1", "s2", "s3")


def driver_sector_residual_stints(laps: pd.DataFrame, driver: str) -> list[dict]:
    """
    Per-stint sector-level residuals (same two-factor OLS as the whole-lap
    residual, applied separately to each sector's time). Because the design
    matrix (TyreLife, LapNumber) is identical across sectors, these three
    residual series sum exactly to the whole-lap residual — this is what lets
    estimate_measurement_noise() below decompose lap-level noise into a
    shared (real pace change) part and a sector-specific (measurement) part.
    """
    driver_laps = laps[laps["Driver"] == driver]
    stints = []
    for _, group in driver_laps.groupby(["Stint", "Compound"]):
        sector_series = {
            key: stint_residuals(group, time_column=col)
            for key, col in zip(SECTOR_KEYS, SECTOR_COLUMNS)
        }
        # Only keep laps where all three sectors survived the NaN guard in stint_residuals
        common_laps = set(sector_series["s1"].index) & set(sector_series["s2"].index) & set(sector_series["s3"].index)
        if len(common_laps) < MIN_SECTOR_LAPS_PER_STINT:
            continue
        aligned = {key: series.loc[sorted(common_laps)] for key, series in sector_series.items()}
        stints.append(aligned)
    return stints


def estimate_measurement_noise(sector_stints: list[dict]) -> dict | None:
    """
    Pools (s1, s2, s3) lap-triples across every stint, then decomposes each
    sector's variance into a shared component (estimated from the pairwise
    covariances — a genuine pace change should move all three sectors
    together) and a leftover idiosyncratic component (sector-specific noise:
    traffic in one sector, a small mistake in one corner). Summing the three
    idiosyncratic variances gives R_hat: an externally-derived measurement
    noise estimate that never touches the lap-to-lap AR(1) dynamics at all.

    Assumes each sector loads equally (1:1) onto the shared pace signal, and
    that sector-specific noise is independent across the three sectors within
    a lap — both are approximations, stated here rather than hidden.
    """
    s1 = np.concatenate([s["s1"].values for s in sector_stints]) if sector_stints else np.array([])
    s2 = np.concatenate([s["s2"].values for s in sector_stints]) if sector_stints else np.array([])
    s3 = np.concatenate([s["s3"].values for s in sector_stints]) if sector_stints else np.array([])

    n = len(s1)
    if n < 10:
        return None

    cov_12 = np.cov(s1, s2, ddof=1)[0, 1]
    cov_13 = np.cov(s1, s3, ddof=1)[0, 1]
    cov_23 = np.cov(s2, s3, ddof=1)[0, 1]
    shared_variance = (cov_12 + cov_13 + cov_23) / 3

    idio_var_s1 = max(np.var(s1, ddof=1) - shared_variance, 0.0)
    idio_var_s2 = max(np.var(s2, ddof=1) - shared_variance, 0.0)
    idio_var_s3 = max(np.var(s3, ddof=1) - shared_variance, 0.0)
    r_hat = idio_var_s1 + idio_var_s2 + idio_var_s3

    return {
        "shared_variance": round(float(shared_variance), 6),
        "idio_var_s1": round(float(idio_var_s1), 6),
        "idio_var_s2": round(float(idio_var_s2), 6),
        "idio_var_s3": round(float(idio_var_s3), 6),
        "R_hat": round(float(r_hat), 6),
        "n_laps": n,
    }


def driver_measurement_noise(laps: pd.DataFrame, driver: str) -> float | None:
    stints = driver_sector_residual_stints(laps, driver)
    result = estimate_measurement_noise(stints)
    return result["R_hat"] if result is not None else None
