import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def build_features(laps: pd.DataFrame, driver: str, window: int = 5) -> np.ndarray:
    # Extract one driver's laps and build the feature matrix the HMM will observe
    # Features: lap time + rolling std of lap time
    # finance equivalent: log returns + rolling vol fed into a regime-detection HMM
    d = laps[laps["Driver"] == driver].copy().sort_values("LapNumber")
    d["rolling_std"] = d["LapTime_s"].rolling(window, min_periods=2).std().fillna(0)
    features = d[["LapTime_s", "rolling_std"]].values
    return features, d["LapNumber"].values


def fit_hmm(features: np.ndarray, n_states: int = 3, random_state: int = 42) -> GaussianHMM:
    # Fit a Gaussian HMM to the feature matrix
    # n_states=3: green flag / degradation phase / safety car (or similar hidden structure)
    # The model learns mean and variance of each hidden state from the data — unsupervised
    # finance: same call used to detect Bull / Bear / Sideways regimes on return series
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",  # each state has its own variance per feature, no cross-feature correlation assumed
        n_iter=100,
        random_state=random_state,
    )
    # fit the HMM to the features — this is where the model learns the parameters of each hidden state (mean lap time, volatility) and the transition probabilities between states
    model.fit(features)
    return model


def label_states(model: GaussianHMM, n_states: int = 3) -> dict:
    # States come out of HMM as 0, 1, 2 — no inherent meaning
    # We label them by ranking their mean lap time: lowest mean = fastest = green flag pace
    means = model.means_[:, 0]  # first feature is LapTime_s
    ranking = np.argsort(means)  # ascending: index 0 = fastest state
    labels = ["green_flag", "degradation", "safety_car"] if n_states == 3 else [str(i) for i in range(n_states)]
    return {int(ranking[i]): labels[i] for i in range(n_states)}


def detect_regimes(laps: pd.DataFrame, driver: str, n_states: int = 3) -> pd.DataFrame:
    # Full pipeline: build features, fit HMM, predict state per lap, attach human-readable label
    features, lap_numbers = build_features(laps, driver)
    # Fitting or decoding can fail when the data is too short or a state is never observed
    # (e.g. degenerate transmat_ row when a driver has no laps in one of the n_states) —
    # wrap both steps and return an empty DataFrame rather than crashing the caller.
    try:
        model = fit_hmm(features, n_states=n_states)
        state_ids = model.predict(features)
    except Exception:
        return pd.DataFrame()
    state_labels = label_states(model, n_states)

    d = laps[laps["Driver"] == driver].copy().sort_values("LapNumber")
    d["state_id"] = state_ids
    d["regime"] = d["state_id"].map(state_labels)
    return d[["LapNumber", "LapTime_s", "Compound", "TyreLife", "state_id", "regime"]]
