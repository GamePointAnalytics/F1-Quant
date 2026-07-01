# QuantTradingSystem
Quantitative trading research framework — pairs trading, options pricing, regime detection. Built over summer 2026 as a learning project into quantitative finance

Status (in active development)

Planned scope:
- Pairs trading
- Black-Scholes/Greeks
- HMM regime detection
- to be continued...

References (papers and texts you'll work from):
- TBA

---

## Technical Deep Dives

### ADF Test — from first principles

Start from the AR(1) model: `y_t = ρy_{t-1} + ε_t`

If ρ = 1, shocks accumulate permanently — the series is a random walk and never reverts to any mean. If |ρ| < 1, shocks decay geometrically and the series is stationary.

Rearrange: `Δy_t = (ρ − 1)y_{t-1} + ε_t = δy_{t-1} + ε_t`

The null hypothesis is δ = 0 (ρ = 1, unit root). The alternative is δ < 0 (mean-reverting pull). The test statistic is δ̂ / SE(δ̂) — structurally a t-statistic, but it does not follow the standard t-distribution. It follows the Dickey-Fuller distribution, which has more negative critical values because under the null the OLS estimator is biased toward zero.

The "Augmented" part adds lagged differences to absorb serial correlation in the residuals: `Δy_t = δy_{t-1} + Σγᵢ Δy_{t-i} + ε_t`. Without these lags, residual autocorrelation inflates the test statistic and produces false rejections. `autolag="AIC"` in the code selects the lag count that minimises the Akaike Information Criterion.

In this codebase we apply it to the OLS residuals of `VER_laptime ~ β × PER_laptime`. A p-value below 0.05 means the spread is stationary — the pair is cointegrated — and mean reversion is a reasonable expectation. A high p-value means the spread can wander without bound and betting on reversion would be a losing strategy.

---

### Why Gaussian HMM over simpler regime models

The obvious alternatives fail for specific reasons.

A threshold model (flag lap as SC if laptime > X) requires an arbitrary circuit-specific threshold with no principled way to set it. It also has no memory — it treats each lap independently.

K-means fails because it ignores temporal structure. Regimes are persistent: a safety car typically runs for five or more consecutive laps. K-means assigns each observation to the nearest centroid with no constraint on consecutive assignments, so it can produce sequences like green / SC / green / SC on adjacent laps, which is physically impossible.

Gaussian HMM is appropriate for three reasons. First, the transition matrix explicitly models persistence — the probability of staying in regime k versus switching is learned from the data, not set by hand. Second, Baum-Welch (expectation-maximisation) optimises all parameters jointly — initial state probabilities, transition probabilities, and Gaussian emission parameters — maximising the likelihood of the observed lap time sequence. There are no arbitrary cutoffs. Third, Viterbi decoding finds the globally most probable state sequence, not just the most probable state per lap in isolation — this enforces temporal coherence.

Why not an LSTM or transformer: a race has roughly 57 laps per driver. Deep sequence models need orders of magnitude more data and would overfit immediately. Interpretability also matters here — we need to explain what the states mean and why the model assigned a specific lap to a specific regime.

---

### The confounding problem the two-factor tyre model addresses — and its honest limitation

The single-factor model `LapTime = α + β × TyreLife + ε` has an omitted variable problem. Fuel load is a confounder: as a stint progresses, TyreLife increases (tyres degrade, car gets slower) while simultaneously fuel burns off (car gets lighter, car gets faster). Both are correlated with LapNumber and both affect lap time in opposite directions. The single β absorbs both effects — fuel improvement partially cancels tyre degradation — so the estimated deg rate is biased downward, understating how fast the tyres are actually wearing.

The two-factor model `LapTime = α + β_tyre × TyreLife + β_fuel × LapNumber + ε` uses partial regression to isolate each effect. β_tyre is the effect of tyre age holding LapNumber fixed. β_fuel is the effect of fuel load holding TyreLife fixed. β_fuel should come out negative (car gets faster as fuel burns) and β_tyre positive (car gets slower as tyres wear).

The honest limitation: within a single stint, TyreLife and LapNumber are nearly perfectly collinear — TyreLife ≈ LapNumber − pit_lap. When two regressors are collinear, the design matrix is ill-conditioned and the individual coefficient estimates become unstable, even if their sum is well-identified. The model gives approximately correct results on average but may produce unreliable β estimates for short stints.

The more rigorous approach is a two-stage regression: estimate β_fuel once at the race level using cross-stint variation (where TyreLife resets at each pit stop but LapNumber does not, breaking the collinearity), subtract out the fuel effect, then fit per-stint deg curves on the residuals. The current implementation is a reasonable approximation and the right conceptual frame — this is the known limitation.

---

## Concept Summaries — Fluent Explanations

**ADF test — why it tests for stationarity and why stationarity matters for pairs trading.**
The core idea is that if the spread between two drivers isn't stationary it has no natural mean to revert to, making the pairs strategy meaningless. The ADF tests whether a unit root exists — a unit root means shocks are permanent rather than mean reverting.

**OLS hedge ratio — why you regress one driver's lap times on the other's to get the hedge ratio rather than just taking the raw spread.**
The raw spread assumes a 1:1 relationship between the two drivers. OLS finds the actual linear relationship that minimizes residual variance, giving you a spread that's more likely to be stationary.

**Two-factor tyre model — why separating TyreLife and LapNumber matters.**
Both variables increase monotonically through a stint. Without separating them you can't distinguish how much of the lap time increase is rubber degradation versus fuel burn off. Including both as separate regressors lets the OLS isolate each effect independently.
