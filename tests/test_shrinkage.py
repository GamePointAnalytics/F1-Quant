import numpy as np

from analysis.shrinkage import (
    fit_beta_binomial_prior,
    shrink_beta_binomial_rate,
    fit_normal_hierarchical_prior,
    shrink_normal_estimate,
)


def simulate_beta_binomial(true_alpha, true_beta, trial_counts, seed=0):
    rng = np.random.default_rng(seed)
    rates = rng.beta(true_alpha, true_beta, size=len(trial_counts))
    successes = rng.binomial(trial_counts, rates)
    return np.array(successes, dtype=float), np.array(trial_counts, dtype=float)


def test_fit_beta_binomial_prior_recovers_population_mean_rate():
    true_alpha, true_beta = 3.0, 27.0  # true mean rate = 0.1
    trial_counts = [40, 60, 80, 100, 50, 70, 90, 45, 65, 85] * 3  # 30 entities
    successes, trials = simulate_beta_binomial(true_alpha, true_beta, trial_counts, seed=1)

    prior = fit_beta_binomial_prior(successes, trials)
    fitted_mean = prior["alpha"] / (prior["alpha"] + prior["beta"])
    true_mean = true_alpha / (true_alpha + true_beta)
    assert abs(fitted_mean - true_mean) < 0.05


def test_shrink_beta_binomial_rate_shrinks_low_trials_more_than_high_trials():
    alpha, beta = 3.0, 27.0  # population mean 0.1
    # both entities have the same raw rate (0.3), very different sample sizes
    low_trials_rate = shrink_beta_binomial_rate(successes=3, trials=10, alpha=alpha, beta=beta)
    high_trials_rate = shrink_beta_binomial_rate(successes=300, trials=1000, alpha=alpha, beta=beta)

    raw_rate = 0.3
    # both should be pulled down toward the population mean (0.1), but the
    # low-trials entity should be pulled much further
    assert low_trials_rate < high_trials_rate
    assert abs(low_trials_rate - 0.1) < abs(high_trials_rate - 0.1)
    # the high-trials entity should stay very close to its own raw rate
    assert abs(high_trials_rate - raw_rate) < 0.02


def test_shrink_beta_binomial_rate_limiting_behavior():
    alpha, beta = 3.0, 27.0
    # trials -> huge: shrinkage should vanish, rate should equal the raw rate
    huge_trials = shrink_beta_binomial_rate(successes=100_000, trials=1_000_000, alpha=alpha, beta=beta)
    assert abs(huge_trials - 0.1) < 0.001


def simulate_normal_hierarchical(true_mu, true_tau2, ses, seed=0):
    rng = np.random.default_rng(seed)
    true_values = rng.normal(true_mu, np.sqrt(true_tau2), size=len(ses))
    estimates = rng.normal(true_values, ses)
    return estimates, np.array(ses)


def test_fit_normal_hierarchical_prior_recovers_population_mean():
    true_mu, true_tau2 = 0.5, 0.04
    ses = [0.05, 0.08, 0.1, 0.06, 0.12] * 6  # 30 entities
    estimates, ses = simulate_normal_hierarchical(true_mu, true_tau2, ses, seed=2)

    prior = fit_normal_hierarchical_prior(estimates, ses)
    assert abs(prior["mu"] - true_mu) < 0.1
    assert prior["tau2"] >= 0


def test_shrink_normal_estimate_shrinks_noisy_estimate_more_than_precise_one():
    mu, tau2 = 0.5, 0.04
    # both entities estimate 0.8, but one is much noisier (larger SE)
    precise = shrink_normal_estimate(estimate=0.8, se=0.02, mu=mu, tau2=tau2)
    noisy = shrink_normal_estimate(estimate=0.8, se=1.0, mu=mu, tau2=tau2)

    assert abs(precise - 0.8) < abs(noisy - 0.8)
    assert abs(precise - 0.8) < 0.05  # precise estimate barely moves
    assert abs(noisy - mu) < 0.05     # noisy estimate is pulled almost fully to mu


def test_shrink_normal_estimate_limiting_behavior_se_near_zero():
    mu, tau2 = 0.5, 0.04
    result = shrink_normal_estimate(estimate=0.9, se=1e-6, mu=mu, tau2=tau2)
    assert abs(result - 0.9) < 1e-3


def test_shrink_normal_estimate_limiting_behavior_zero_population_spread():
    # tau2 -> 0: population has no real between-entity variation, so every
    # entity should be assigned the population mean regardless of its own SE
    result = shrink_normal_estimate(estimate=0.9, se=0.5, mu=0.5, tau2=0.0)
    assert abs(result - 0.5) < 1e-9
