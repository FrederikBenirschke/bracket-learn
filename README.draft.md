# bracketlearn

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)
![Version](https://img.shields.io/badge/version-0.8.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Linter: Ruff](https://img.shields.io/badge/linter-ruff-D7FF64.svg)
![Type checked: mypy](https://img.shields.io/badge/types-mypy-2A6DB2.svg)
![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)

**A scikit-learn-style toolkit for estimating the predictive distribution of a
scalar outcome and pricing the prediction-market contracts written on it.**

## Problem statement

Let `Y` be a continuous outcome (tomorrow's high temperature, a game's final
margin, the next GDP print) with features `X`. A prediction-market contract
pays a known function `g(Y) ∈ {0, 1}` of that outcome, so its risk-neutral fair
price is the conditional expectation `E[g(Y) | X]`, a functional of the
conditional predictive distribution `F(y | X) = P(Y ≤ y | X)`. Every contract a
venue lists reduces to one such functional:

| Contract | Payoff `g(Y)` | Fair price as a functional of `F` |
|---|---|---|
| Bracket `[a, b)` | `1[a ≤ Y < b]` | `F(b) − F(a)` |
| Threshold above `k` | `1[Y > k]` | `1 − F(k)` |
| Threshold below `k` | `1[Y ≤ k]` | `F(k)` |
| Twin (paired) at `k` | `(1[Y ≤ k], 1[Y > k])` | `(F(k), 1 − F(k))` |

Pricing a venue therefore decomposes into two estimands: the predictive
distribution `F(· | X)`, and the functionals of `F` that the listed contracts
select. bracketlearn estimates the first and evaluates the second, then scores
both against realized outcomes with proper scoring rules.

## Method

The library implements three stages behind one scikit-learn-style API.

1. **Estimate `F(· | X)`.** Fit a probabilistic regressor and obtain a typed
   `DistributionForecast`, an explicit representation of the predictive density
   (Normal, Student-t, mixture, quantile, or bracket). Cross-validation,
   calibration, and conformal adjustment compose as pipeline stages.
2. **Evaluate the contract functionals.** A `ContractAdapter` maps `F` to the
   fair prices of the listed contracts in closed form (the CDF differences and
   tail probabilities above). Bracket grids may rotate per observation, as on
   the daily-relisted Kalshi temperature ladders.
3. **Score both estimands.** Assess `F` with strictly proper scores (CRPS, log
   score) and its calibration with the probability integral transform; assess
   the contract prices with the Brier and log-loss scores on the realized
   indicators.

Most probabilistic-forecasting libraries terminate at stage 1. bracketlearn
carries the same typed forecast through pricing and scoring, so the
contract-pricing functionals and the calibration diagnostics are part of the
library rather than per-project glue. The outcome is treated as an abstract
scalar; temperature appears in the examples only as a concrete continuous
quantity, and any outcome with bracket, threshold, or spread contracts uses the
same interface.

## Install

bracketlearn has not reached PyPI. Install from source:

```bash
git clone https://github.com/FrederikBenirschke/bracket-learn
pip install -e ./bracket-learn

# With the optional trainer backends (LightGBM, NGBoost, torch, ...):
pip install -e "./bracket-learn[demo]"
```

After publication: `pip install bracket-learn` or `pip install "bracket-learn[demo]"`.

## Quickstart

**What is a "contract"?** A prediction-market contract is a yes/no bet on an
outcome that pays $1 if it happens and $0 if it does not. "Will tomorrow's NYC
high be above 75°F?" is one contract; "will it land in the 70–80°F bracket?" is
another. Because the payoff is just $0 or $1, the *fair price* of the contract
equals the **probability** of the yes-event — so pricing a contract is the same
problem as forecasting that probability. bracketlearn does this in three stages:
estimate the full distribution of the outcome once, read each contract's
probability off that distribution, then score those probabilities against what
actually happened.

```python
import numpy as np
from bracketlearn import EMOS, BracketLadder, BinaryAbove
from bracketlearn.score import brier_bracket, log_loss_bracket

# Synthetic NYC max-temperature data: 200 days, predict the daily high (y)
# from yesterday's high and the seasonal average (the two columns of X).
rng = np.random.default_rng(0)
N = 200
day = np.arange(N)
season = 70 + 15 * np.sin(2 * np.pi * day / 365.0)
prior_high = season + rng.normal(0, 4, N)
X = np.column_stack([prior_high, season])
y = season + 0.6 * (prior_high - season) + rng.normal(0, 5, N)
X_tr, X_te, y_tr, y_te = X[:150], X[150:], y[:150], y[150:]   # train on 150, test on 50

# Stage 1: estimate the outcome distribution F(.|X). For each of the 50 test
# days EMOS predicts a full bell curve (a mean mu and spread sigma), not just a
# point. ids/timestamps are row labels that ride along with each forecast so
# prices can later be matched to real venue quotes; they do NOT affect the
# prediction. Pass one per test row (50 here).
emos = EMOS().fit(X_tr, y_tr)
dist = emos.predict_dist(X_te, ids=np.arange(50), timestamps=np.arange(50, dtype=float))

# Stage 2: read each contract's probability off that distribution. Every
# .price(dist) returns a ContractForecast; its .fair_price holds the prices
# (= probabilities), one number per contract per day.
above = BinaryAbove(strike=75.0).price(dist)   # P(high > 75) = 1 - F(75)
edges = np.array([0.0, 60.0, 70.0, 80.0, 90.0, 100.0])
ladder = BracketLadder(edges_per_row=[edges] * 50).price(dist)  # P(high in each bracket)

print("P(high > 75), first 3 days:", above.fair_price[:3])

# Stage 3: grade the bracket prices against the realized highs y_te. Lower is
# better for both scores (they reward putting probability on the bracket that
# actually occurred).
print(f"Brier:    {brier_bracket(ladder, edges, y_te):.4f}")
print(f"log-loss: {log_loss_bracket(ladder, edges, y_te):.4f}")
```

Every adapter's `.price(dist)` returns the same type, a `ContractForecast`: a
`fair_price` vector plus `entity_ids`, `contract_ids`, and `group_id` columns
that let you align the prices against venue quotes. The example fits a single
EMOS on a holdout split; in practice the estimator is wrapped in a `Pipeline`
and evaluated under `WalkForward` for cross-validated, calibrated forecasts. See
the [quickstart](docs/guides/quickstart.md) and
[concepts](docs/guides/concepts.md) guides.

## Estimators and composition

```python
from bracketlearn import Pipeline, WalkForward
from bracketlearn.lift import GlobalResidual, Isotonic
from bracketlearn.trainers import EMOS, QuantileReg, SklearnPoint
from sklearn.linear_model import RidgeCV

ridge = Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()], name="ridge")
emos = Pipeline([EMOS(), Isotonic()], name="emos")
qreg = Pipeline([QuantileReg(n_estimators=100)], name="qreg")

wf = WalkForward(cv="expanding-window", n_folds=5, refit_on_full=True)
result = wf.fit_predict([ridge, emos, qreg], X, y, ids=ids, timestamps=ts)
print(result.to_table(y, metrics=["crps", "log_score", "brier_bracket"], edges=edges))
```

A `Pipeline` composes stages by protocol type; `WalkForward` provides the
cross-validation and out-of-fold predictions. The five protocols form the type
algebra of the estimation stage:

| Protocol          | Signature                                     | Examples                                                       |
|-------------------|-----------------------------------------------|----------------------------------------------------------------|
| `PointForecaster` | `X → PointForecast` (a conditional mean)      | `SklearnPoint(Ridge())`, `OnlineAggregator`, `RNNHourly`       |
| `DistForecaster`  | `X → DistributionForecast`                    | `EMOS`, `NGBoostNormal`, `QuantileReg`, `CumulativeBinary`     |
| `Lifter`          | `PointForecast → DistributionForecast`        | `GlobalResidual`, `StudentTResidual`, `GARCHResidual`          |
| `Calibrator`      | `DistributionForecast → DistributionForecast` | `Isotonic`, `ConformalCalibrate`                               |
| `ContractAdapter` | `DistributionForecast → ContractForecast`     | `BinaryAbove`, `BinaryBelow`, `Twin`, `ThresholdLadder`, `BracketLadder` |

The estimator families (parametric, quantile, bracket-native, stacking,
value-tilted), the distribution representations, and the `integrate()`
discretization functional are documented in
[concepts](docs/guides/concepts.md) and the [catalog](docs/guides/catalog.md).

## Pricing adapters

A `ContractAdapter` evaluates the fair-price functional of any
`DistributionForecast` for the contracts a venue lists:

| Adapter                        | Functional                       | Venue examples                                              |
|--------------------------------|----------------------------------|-------------------------------------------------------------|
| `BinaryAbove(k)`               | `1 − F(k)`                       | Kalshi "high above 80°F", "S&P > 5000 by Friday"           |
| `BinaryBelow(k)`               | `F(k)`                           | Kalshi "GDP ≤ 2.5%", "low below freezing"                  |
| `Twin(k)`                      | `(F(k), 1 − F(k))`               | Polymarket spread (`Eagles -3.5`), total (`Over 47.5`)     |
| `ThresholdLadder(ks)`          | `[1 − F(k_i)]` (survival)        | Kalshi multi-threshold temperature ladders                 |
| `BracketLadder(edges_per_row)` | `[F(b) − F(a)]` per bin          | Kalshi daily-rotating brackets; Polymarket weather brackets|

The [adapters guide](docs/guides/adapters.md) derives the Kalshi temperature and
spread/total mappings in full.

## Accuracy and value

Proper scores measure the distance from the forecast to the outcome (accuracy).
A second criterion measures whether the forecast price improves on a reference
price `m` already quoted by the market (value). Under proportional betting the
expected payoff of price `q` against reference `m` and outcome `r` is
`(q − m)(r − m)`, whose population maximizer is *not* the most accurate
forecast: value accrues where the forecast's error is orthogonal to the
reference's mispricing. `score.edge_alignment(q, m, r)` estimates this quantity
and `score.value_report` gives its additive decomposition; the
`bracketlearn.value` trainers (`BlendedBracketGBM`, `BlendedBracketNet`)
optimize a fee-aware blend of calibration and value directly. The connections
to the Kelly criterion, Grinold's Fundamental Law, and Grossman–Stiglitz are
set out in [value vs accuracy](docs/guides/value_vs_accuracy.md),
[value with fees](docs/guides/value_with_fees.md), and
[value trainers](docs/guides/value_trainers.md).

## Documentation

The [guides](docs/guides/) cover each component:

- Foundations: [quickstart](docs/guides/quickstart.md),
  [concepts](docs/guides/concepts.md), [catalog](docs/guides/catalog.md),
  [package map](docs/guides/package_map.md)
- Evaluation protocol: [cv](docs/guides/cv.md), [weights](docs/guides/weights.md),
  [multitarget](docs/guides/multitarget.md), [search](docs/guides/search.md),
  [persistence](docs/guides/persistence.md)
- Pricing and scoring: [adapters](docs/guides/adapters.md),
  [scoring](docs/guides/scoring.md), [tail policies](docs/guides/tail_policies.md),
  [bracket expander](docs/guides/bracket_expander.md)
- Value: [value vs accuracy](docs/guides/value_vs_accuracy.md),
  [value with fees](docs/guides/value_with_fees.md),
  [value trainers](docs/guides/value_trainers.md)
- [baselines](docs/guides/baselines.md), [examples](docs/guides/examples.md)

## Scope

bracketlearn estimates fair prices. It does not size positions. Mapping a fair
price to a trade (edge gates, correlated-ladder selection, group Kelly, fee
schedules, queue models) depends on private signal and venue microstructure and
is left to the caller.

## Status

Version 0.8.0, pre-PyPI. Every component above is implemented and covered by the
test suite. [CHANGELOG.md](CHANGELOG.md) records the version history and
migration notes.

## License

MIT.
