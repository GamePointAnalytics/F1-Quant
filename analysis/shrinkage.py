import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

# Two generic hierarchical/empirical-Bayes shrinkage primitives, reused by
# both the team DNF-rate model (prediction/reliability.py) and the driver
# phi model (analysis/lap_consistency.py). Both are fit the same way the
# Kalman filter's (phi, Q, R) are fit in analysis/kalman_lap_state.py: build
# a marginal likelihood over candidate population-level hyperparameters and
# let scipy.optimize find the maximum, log-parameterizing any strictly
# positive parameter so the search never needs a hard boundary constraint.
#
# The point of both: an entity fit in isolation (a team with 40 starts, a
# driver with 300 lag-pairs) is estimated with real sampling noise. Instead
# of trusting that noisy estimate directly, or a discrete cutoff ("use the
# global rate below N starts" — the old approach), these let every entity's
# estimate be pulled toward the population mean by an amount proportional to
# how uncertain that entity's own estimate is.


# ---------------------------------------------------------------------------
# Beta-Binomial: for rates/proportions (team DNF rate). Each entity's true
# rate p_i ~ Beta(alpha, beta); observed successes_i ~ Binomial(trials_i, p_i).
# ---------------------------------------------------------------------------

def _beta_binomial_neg_log_likelihood(params, successes, trials):
    log_alpha, log_beta = params
    alpha, beta = np.exp(log_alpha), np.exp(log_beta)
    ll = (
        gammaln(trials + 1) - gammaln(successes + 1) - gammaln(trials - successes + 1)
        + betaln(successes + alpha, trials - successes + beta)
        - betaln(alpha, beta)
    )
    return -float(np.sum(ll))


def fit_beta_binomial_prior(successes: np.ndarray, trials: np.ndarray) -> dict:
    """MLE fit of a population-level Beta(alpha, beta) prior across many entities'
    (successes, trials) pairs — the compound Beta-Binomial likelihood."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    mean_rate = float(successes.sum() / trials.sum())
    mean_rate = min(max(mean_rate, 1e-4), 1 - 1e-4)
    x0 = [np.log(2.0), np.log(2.0 * (1 - mean_rate) / mean_rate)]

    result = minimize(
        _beta_binomial_neg_log_likelihood,
        x0=x0,
        args=(successes, trials),
        method="Nelder-Mead",
    )
    alpha, beta = np.exp(result.x)
    return {
        "alpha": round(float(alpha), 4),
        "beta": round(float(beta), 4),
        "log_likelihood": round(float(-result.fun), 4),
        "converged": bool(result.success),
    }


def shrink_beta_binomial_rate(successes: float, trials: float, alpha: float, beta: float) -> float:
    """Posterior mean of a Beta(alpha, beta) prior updated with one entity's
    own (successes, trials) — a continuous blend toward the population rate,
    weighted by how much data this entity itself has."""
    return (successes + alpha) / (trials + alpha + beta)


# ---------------------------------------------------------------------------
# Normal-Normal: for point estimates with a known standard error (driver phi).
# Each entity's true value x_i ~ Normal(mu, tau^2); observed estimate
# x_hat_i ~ Normal(x_i, se_i^2) — standard random-effects meta-analysis setup.
# ---------------------------------------------------------------------------

def _normal_hierarchical_neg_log_likelihood(params, estimates, standard_errors):
    mu, log_tau2 = params
    tau2 = np.exp(log_tau2)
    variance = tau2 + standard_errors ** 2
    ll = -0.5 * np.log(2 * np.pi * variance) - 0.5 * (estimates - mu) ** 2 / variance
    return -float(np.sum(ll))


def fit_normal_hierarchical_prior(estimates: np.ndarray, standard_errors: np.ndarray) -> dict:
    """MLE fit of a population-level Normal(mu, tau^2) prior across many entities'
    own point estimates and standard errors."""
    estimates = np.asarray(estimates, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)

    x0 = [float(np.mean(estimates)), np.log(max(float(np.var(estimates)), 1e-6))]
    result = minimize(
        _normal_hierarchical_neg_log_likelihood,
        x0=x0,
        args=(estimates, standard_errors),
        method="Nelder-Mead",
    )
    mu, log_tau2 = result.x
    return {
        "mu": round(float(mu), 4),
        "tau2": round(float(np.exp(log_tau2)), 6),
        "log_likelihood": round(float(-result.fun), 4),
        "converged": bool(result.success),
    }


def shrink_normal_estimate(estimate: float, se: float, mu: float, tau2: float) -> float:
    """Posterior mean blend of one entity's own noisy estimate and the
    population mean. weight -> 1 (trust the entity's own estimate) as its SE
    shrinks toward 0; weight -> 0 (trust the population mean) as its SE grows
    large relative to the population's own spread (tau2)."""
    denom = tau2 + se ** 2
    weight = tau2 / denom if denom > 0 else 0.0
    return weight * estimate + (1 - weight) * mu
