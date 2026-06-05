# bracketlearn

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![Version](https://img.shields.io/badge/version-0.6.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Linter: Ruff](https://img.shields.io/badge/linter-ruff-D7FF64.svg)
![Type checked: mypy](https://img.shields.io/badge/types-mypy-2A6DB2.svg)
![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)

**A scikit-learn-style toolkit for forecasting a continuous number, then
pricing the prediction-market contracts that pay out on it.**

## Contents

- [Prediction markets and the pricing problem](#prediction-markets-and-the-pricing-problem)
- [The three steps](#the-three-steps)
- [Install](#install)
- [Quickstart: the three steps end to end](#quickstart-the-three-steps-end-to-end)
- [Step 1: forecast a distribution](#step-1-forecast-a-distribution) (Pipeline, protocols, distribution backings, estimator families)
- [Step 2: price the contracts](#step-2-price-the-contracts) (the adapter catalogue and venue mappings)
- [Step 3: score the prices](#step-3-score-the-prices)
- [Operating the pipeline](#operating-the-pipeline) (CV, sample weights, pooling, multi-target, search)
- [Out of scope: trade decisions](#out-of-scope-trade-decisions)
- [Status](#status)
- [License](#license)

## Prediction markets and the pricing problem

A prediction market sells contracts that pay **$1 when an event happens** and
**$0 when it doesn't**. Browse [Kalshi](https://kalshi.com) or
[Polymarket](https://polymarket.com) and you find markets like this:

> **"Will today's high temperature in New York land between 70°F and 72°F?"**
> YES trades at 31¢.

A YES contract pays $1 when the event happens, so its price reads as a
probability: 31¢ means traders put the chance near 31%. You meet the same
underlying quantity sold three ways. Brackets split it into mutually-exclusive
buckets (`68–70°F`, `70–72°F`, `72–74°F`). Thresholds ask one cutoff ("high
above 75°F"). Spreads and totals settle a game ("Eagles −3.5", "over 47.5
points").

Trading these contracts takes two things the venue won't give you. First, your
own probability distribution over the underlying number: tomorrow's high, the
final margin, the next GDP print. Calibrate it so the events you call
30%-likely arrive about 30% of the time. Second, a way to read a fair price for
every contract shape off that one distribution. A bracket needs a bucket
probability, a threshold needs a tail probability, a ladder needs a survival
value. bracketlearn turns your features into those fair prices.

## The three steps

You forecast, you price, you score, all through a scikit-learn-style API. The
rest of this README follows these three steps in order.

1. **Forecast a distribution.** Fit a probabilistic model on your features:
   `EMOS`, `NGBoostNormal`, `QuantileReg`, `MixtureNormals`, `CumulativeBinary`,
   and more. Cross-validation, calibration, and conformal correction come built
   in. You get back a typed `DistributionForecast` that carries the full
   predictive density.
2. **Price the contracts.** Convert that distribution into fair prices for each
   venue shape: single-threshold binaries, paired YES/NO twins, threshold
   ladders, and bracket ladders whose edges rotate per row. Kalshi reshuffles
   its temperature, GDP, and Fed-decision ladders daily; pass repeated edges
   when every row shares one grid.
3. **Score the prices.** Check the fair prices against realized outcomes with
   proper scoring rules: CRPS, log-score, and PIT on the distribution, Brier
   and log-loss on the contracts. The numbers tell you whether your prices were
   calibrated.

Most probabilistic-forecasting libraries stop at step 1. bracketlearn carries
the same typed forecast through pricing and scoring, so the contract math and
the calibration check live in the library instead of scattered glue in your
notebook.

> **Beyond weather.** This README runs on temperature because it makes the
> cleanest continuous underlying. Nothing in the library knows about weather.
> Any continuous quantity with bracket, threshold, or spread contracts uses the
> same API: sports margins, index levels, economic releases.

## Install

bracketlearn has not reached PyPI yet. Install from source:

```bash
git clone https://github.com/FrederikBenirschke/bracketlearn
pip install -e ./bracketlearn

# With the full set of optional trainers (LightGBM, NGBoost, torch, ...):
pip install -e "./bracketlearn[demo]"
```

After PyPI publication the install becomes `pip install bracketlearn` or
`pip install "bracketlearn[demo]"`.

## Quickstart: the three steps end to end

This one script runs all three steps: generate synthetic weather features, fit
EMOS (step 1), price the four contract shapes a venue lists (step 2), then score
the fair prices against what happened (step 3). Each building block gets its own
section after this.

```python
import numpy as np
from bracketlearn import EMOS, BracketLadder, BinaryAbove, Twin
from bracketlearn.score import brier_bracket, log_loss_bracket

# --- synthetic NYC max-temperature data ---
rng = np.random.default_rng(0)
N = 200
day = np.arange(N)
season = 70 + 15 * np.sin(2 * np.pi * day / 365.0)
prior_high = season + rng.normal(0, 4, N)
X = np.column_stack([prior_high, season])
y = season + 0.6 * (prior_high - season) + rng.normal(0, 5, N)
X_tr, X_te, y_tr, y_te = X[:150], X[150:], y[:150], y[150:]

# --- step 1: fit EMOS (ensemble-mean + spread regression) ---
emos = EMOS().fit(X_tr, y_tr)
dist = emos.predict_dist(
    X_te,
    ids=np.arange(50),
    timestamps=np.arange(50, dtype=float),
)
# dist.params["mu"], dist.params["sigma"] now hold (50,) forecasts.

# --- step 2: price the contracts you'd see on a prediction market ---

# (1) Single threshold: "high above 75°F today"
fair_above_75 = BinaryAbove(strike=75.0).price(dist).fair_price
# fair_above_75[i] = model P(high_i > 75); feed this into your own
# trading layer alongside whatever quotes you scraped from the venue.

# (2) Paired YES/NO at 70°F (spread / total style)
twin = Twin(strike=70.0).price(dist)
yes = twin.fair_price[twin.contract_ids == 0][0]
no  = twin.fair_price[twin.contract_ids == 1][0]
print(f"Twin(70)  yes={yes:.3f}  no={no:.3f}  (sum=1.000)")

# (3) Bracket ladder, shared edges across all rows (Polymarket weekly style):
edges = np.array([0.0, 60.0, 70.0, 80.0, 90.0, 100.0])
ladder = BracketLadder(edges_per_row=[edges] * 50).price(dist)
# 5 contracts per entity: P([0,60)), P([60,70)), P([70,80)), ...

# (4) Bracket ladder, edges varying per row (Kalshi daily-rotating style):
edges_per_day = [
    np.array([mu - 10, mu - 3, mu, mu + 3, mu + 10])
    for mu in dist.params["mu"]
]
per_row = BracketLadder(
    edges_per_row=edges_per_day,
    include_tail_buckets=True,      # add "below" and "above" rows
).price(dist)
# Per-entity rows sum to exactly 1.0.

# --- step 3: score the fair prices against the realized outcomes ---
# Brier / log-loss on the bracket: are the fair prices calibrated?
print(f"BracketLadder Brier:    {brier_bracket(ladder, edges, y_te):.4f}")
print(f"BracketLadder log-loss: {log_loss_bracket(ladder, edges, y_te):.4f}")
```

Output for the first entity:

```
Twin(70)  yes=0.912  no=0.088  (sum=1.000)

BracketLadder fair prices for entity 0 (shared edges):
    [0, 60)   = 0.000
    [60, 70)  = 0.088
    [70, 80)  = 0.635
    [80, 90)  = 0.271
    [90, 100) = 0.006

BracketLadder (per-row edges centered on each day's forecast):
    < 67.0          = 0.026
    [67.0, 74.0)    = 0.254
    [74.0, 77.0)    = 0.220
    [77.0, 80.0)    = 0.220
    [80.0, 87.0)    = 0.254
    > 87.0          = 0.026
                sum = 1.000

BracketLadder Brier:    0.4684
BracketLadder log-loss: 0.7950
```

`.price(dist)` returns a `ContractForecast`: a `fair_price` array plus the typed
`entity_ids`, `contract_ids`, and `group_id` indexing you need to line it up
against venue quotes, with provenance attached. Gating on edge, sizing by Kelly,
hedging across a ladder: that trading layer is yours to write.

The example fits a bare EMOS on a train/test split. In practice you wrap the
forecaster in a `Pipeline` and run it under `WalkForward` for cross-validated,
calibrated forecasts. [Step 1](#step-1-forecast-a-distribution) shows how.

## Step 1: forecast a distribution

Everything starts with a `DistributionForecast`, a typed predictive density over
the underlying number. You build one by chaining stages into a `Pipeline` and
running it under `WalkForward`. This section covers how models compose, the
distribution types they emit, and the trainer families you pick from.

### Compose models with Pipeline and WalkForward

```python
import numpy as np
from sklearn.linear_model import RidgeCV

from bracketlearn import Pipeline, WalkForward
from bracketlearn.lift import GlobalResidual, Isotonic
from bracketlearn.trainers import EMOS, QuantileReg, SklearnPoint

edges = np.linspace(0, 100, 11)   # 10 brackets

# Each model is a Pipeline (a sequential chain of stages); names are labels.
ridge = Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()], name="ridge")
emos = Pipeline([EMOS(), Isotonic(pre_integrate_edges=edges)], name="emos")
qreg = Pipeline([QuantileReg(n_estimators=100)], name="qreg")

# WalkForward is the CV/OOF driver. Pass one model or a list of them.
wf = WalkForward(cv="expanding-window", n_folds=5, refit_on_full=True)
result = wf.fit_predict([ridge, emos, qreg], X, y, ids=ids, timestamps=ts)

# Distribution-level metrics on OOF predictions.
print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

# Bracket-contract metrics: pass the shared edge vector; the result builds
# the per-row bracket ladder internally per stage.
print(result.to_table(y, metrics=["log_loss_bracket", "brier_bracket"],
                      edges=edges))

# Predict on unseen data using each model's full-train refit.
new_dists = wf.predict(X_new, ids=new_ids, timestamps=new_ts)
```

### The sklearn contract

Every forecaster, lifter, and calibrator inherits from `BaseEstimator` and
supports `get_params`, `set_params`, and `clone()`. `WalkForward` clones each
model before every fold's fit, so your instances stay unmutated and you reuse
them across runs.

### The five protocols

| Protocol          | Input → Output                                | Examples                                                       |
|-------------------|-----------------------------------------------|----------------------------------------------------------------|
| `PointForecaster` | `X → PointForecast` (μ̂)                     | `SklearnPoint(Ridge())`, `OnlineAggregator`, `RNNHourly`       |
| `DistForecaster`  | `X → DistributionForecast`                    | `EMOS`, `NGBoostNormal`, `QuantileReg`, `CumulativeBinary`     |
| `Lifter`          | `PointForecast → DistributionForecast`        | `GlobalResidual`, `StudentTResidual`, `GARCHResidual`          |
| `Calibrator`      | `DistributionForecast → DistributionForecast` | `Isotonic`, `ConformalCalibrate`                               |
| `ContractAdapter` | `DistributionForecast → ContractForecast`     | `BinaryAbove`, `BinaryBelow`, `Twin`, `ThresholdLadder`, `BracketLadder` |

List stages in a `Pipeline` and it wires them left-to-right by protocol type. A
`PointForecaster` followed by a `Lifter` becomes a `DistForecaster`; add a
`Calibrator` and it stays one. For parallel ensembling, wrap upstream `Pipeline`
objects in a `Stacker`. `WalkForward` drives the CV and OOF. Names label the
leaderboard; they never wire anything.

### Distribution backings

`DistributionForecast` is an `abc.ABC` base with five concrete subclasses. Each
subclass owns typed storage and its own math; metrics and adapters dispatch
through `isinstance` (or the compat `dist.backing` property).

| Subclass                  | Storage                                | Math notes                                            |
|---------------------------|----------------------------------------|-------------------------------------------------------|
| `NormalForecast`          | `mu, sigma` per row                    | Closed-form scipy.stats.norm                          |
| `StudentTForecast`        | `mu, sigma, df` per row                | Closed-form scipy.stats.t; requires df > 2            |
| `MixtureNormalForecast`   | `weights, mus, sigmas` per row (N, K)  | CDF = Σ w_k Φ((x−μ_k)/σ_k); PPF via bisection        |
| `QuantileForecast`        | shared `taus` + per-row `qvals` (N, Q) | Pinball-trapezoidal CRPS; `TailPolicy` required      |
| `BracketForecast`         | per-row `edges` (N, B+1) + `probs`     | Uniform-within-bin; NaN-padded ragged rows supported |

Construct a subclass directly:

```python
from bracketlearn import NormalForecast
d = NormalForecast.from_arrays(
    mu=mu, sigma=sigma,
    ids=ids, timestamps=ts, provenance=prov,
)
```

The `DistributionForecast.from_*` classmethods route to the subclasses
(`from_normal` calls `NormalForecast.from_arrays`, and so on).

#### Per-row brackets

`BracketForecast.edges` has shape `(N, B+1)`. Each row carries its own bracket
grid. The Kalshi temperature contract listed on May 26 shares no edges with the
May 27 listing, and bracketlearn stores them apart. Ragged rows ride on NaN
padding: row i's valid prefix runs to the first `B_i + 1` non-NaN edges and the
first `B_i` non-NaN probs.

`BracketForecast.from_arrays` also takes a 1-D shared edge vector and broadcasts
it to every row, so callers on a genuine shared ladder pay no ergonomic cost.
Per-row `self.edges` (2-D, NaN-padded for ragged rows) stays the canonical
access path.

#### The `integrate()` bridge

Every `DistributionForecast` subclass implements
`integrate(edges_per_row) → BracketForecast`. One method turns a continuous
distribution into a discrete one on a specific grid:

```python
# EMOS emits a NormalForecast; price it on per-row Kalshi ladders.
normal_dist = emos.predict_dist(X, ids=ids, timestamps=ts)
bracket_dist = normal_dist.integrate(edges_per_row)
# bracket_dist.probs has shape (N, B_max) with each row's prob mass on
# its own grid (NaN-padded if rows differ in length).
```

`edges_per_row` takes three shapes: 1-D shared `(B+1,)`, 2-D dense `(N, B+1)`,
or a length-N sequence of 1-D arrays (NaN-padded for you). Each row renormalises
to sum to 1. A row that lands entirely outside the distribution's support raises
rather than fabricate a silent uniform.

### Estimator families

Six families group the trainers by **what they model**. Pick the family from the
shape of your signal; inside a family the members trade off linearity, priors,
and compute.

| Family | Estimators | What it models |
|---|---|---|
| **Point** | `SklearnPoint`, `OnlineAggregator`, `RNNHourly` | a single μ̂ per row; lift to a distribution with a residual σ (or a calibration stage) |
| **Parametric distribution** | `EMOS`, `HeteroscedasticNormal`, `NGBoostNormal`, `MixtureNormals`, `BayesianRidge`, `HierarchicalNormal` | a closed-form density (Normal / mixture) whose moments are functions of the features |
| **Quantile / non-parametric** | `QuantileReg`, `QuantileForest` | a quantile function / empirical CDF, no distributional shape assumed |
| **Bracket-native** | `CumulativeBinary` (+ the `BracketExpander` entry point) | bracket / cutpoint indicators directly on each row's own grid |
| **Stacking / combiners** | `StackedParametric`, `BMAStacking`, `BracketStacking`, `LinearPoolDist`, `TailSpecialist`, `CDFBoostBracket`, `DistAsFeatures` | a combination of upstream forecasts (parametric meta-learner, Bayesian average, opinion pool, tail specialist) |
| **Baselines** | `Persistence`, `PersistenceDist`, `EmpiricalDistribution` | reference forecasts to beat; plus convenience factories `ridge`, `emos_calibrated` |

Within the parametric family, the mean/variance flexibility ladder is the part
to learn:

- `EMOS` puts an affine mean on `ens_mean` and a fixed-function scale on
  `ens_std`. Two hard-wired inputs.
- `HeteroscedasticNormal` generalises it to the features: `μ = Xμ·βμ`,
  `log σ = Xσ·βσ`. Any columns (cloud, wind, dewpoint, spread) drive both the
  location and the width, with readable linear coefficients. `EMOS` is the
  special case `Xμ=[ens_mean]`, `Xσ=[ens_std]`.
- `NGBoostNormal` targets the same `(μ̂, σ̂)`-from-features but gradient-boosts
  it: non-linear, noisier at low N, and you lose interpretability.
- `MixtureNormals` handles bi- and multi-modal outcomes.
- `BayesianRidge` and `HierarchicalNormal` bring conjugate priors and cross-site
  partial pooling for small samples.

#### Distribution-first vs bracket-aware trainers

A second axis cuts across the families: what a trainer sees at fit time.

- **Distribution-first** (`EMOS`, `NGBoostNormal`, `MixtureNormals`,
  `QuantileReg`, `QuantileForest`, `StackedParametric`, `BMAStacking`,
  `BayesianRidge`, `HierarchicalNormal`, `OnlineAggregator`, `RNNHourly`,
  `ridge`, `emos_calibrated`) never touch brackets at fit time. They fit on
  `(X, y)`, emit a continuous-ish distribution, and you call
  `.integrate(edges_per_row)` to price on a grid.
- **Bracket-aware** (`CumulativeBinary`, `TailSpecialist`, `CDFBoostBracket`)
  train on bracket-derived indicators. Each takes a `cutpoints_by_id` or
  `brackets_by_id` dict (id to 1-D edge array) at construction, so per-row grids
  flow through fit and predict. Their `fit()` signatures require an explicit
  `ids=` kwarg; a `Pipeline` forwards it for you.

  For the "fit any sklearn classifier or regressor on brackets" path, reach for
  `BracketExpander` (in `bracketlearn.transformers`). It owns the per-row to
  per-(row, bracket) reshape and leaves model choice and target construction to
  you. `fit_transform(X, y, ids=...)` returns `(X_expanded, y_expanded)`:
  `X_expanded` is `(M, F+2)` with `[..., lo, hi]` appended, and `y_expanded` is
  the default bracket-hit indicator `1[y ∈ [lo, hi))`. Fit any sklearn estimator
  on those arrays, then pack the predictions back into a row-renormalised
  `BracketForecast` with `assemble_dist`.

  ```python
  from bracketlearn import BracketExpander
  from lightgbm import LGBMClassifier

  exp = BracketExpander(brackets_by_id=bbi)
  X_exp, y_exp = exp.fit_transform(X, y, ids=ids)
  clf = LGBMClassifier(...).fit(X_exp, y_exp)
  X_pred_exp, _ = exp.transform(X_pred, ids=pred_ids)
  scores = clf.predict_proba(X_pred_exp)[:, 1]
  d = exp.assemble_dist(scores, ids=pred_ids, timestamps=ts)
  ```

  For a custom per-(row, bracket) target (a mispricing residual, an
  importance-weighted hit), build it on top of `fit_transform` output. The
  expander holds no opinion about the loss.

## Step 2: price the contracts

You have a `DistributionForecast`. A `ContractAdapter` reads fair prices off it
for the exact contracts a venue lists. The Quickstart used `BinaryAbove`,
`Twin`, and `BracketLadder`; here is the full set, with each adapter mapped to
the venue shape it prices.

### Adapter catalogue

| Adapter                | Pricing                            | Maps to (examples)                                          |
|------------------------|------------------------------------|-------------------------------------------------------------|
| `BinaryAbove(k)`       | `P(X > k)`                         | Kalshi "high above 80°F", "S&P > 5000 by Friday"            |
| `BinaryBelow(k)`       | `P(X ≤ k)`                         | Kalshi "GDP ≤ 2.5%", "low below freezing"                   |
| `Twin(k)`              | paired `P(X > k)` / `P(X ≤ k)`     | Polymarket spread (`Eagles -3.5`), total (`Over 47.5`)      |
| `ThresholdLadder(ks)`  | `[P(X > k_i)]` per strike          | Kalshi multi-threshold temperature ladders                  |
| `BracketLadder(edges_per_row)` | `[P(lo ≤ X < hi)]` per-row edges | Kalshi daily-rotating brackets; Polymarket weather brackets (pass `[edges]*N`) |

All five adapters take any `DistributionForecast` (normal, student-t,
mixture-normal, quantile, or bracket backing) and return a long-form
`ContractForecast` carrying `fair_price`, `entity_ids`, `group_id`,
`contract_spec`, and provenance.

### Worked mapping: Kalshi NYC temperature

Kalshi runs a daily-rotating bracket ladder on NYC max temperature. The
brackets shift every day: Monday lists `{<60, 60–65, 65–70, …}`, Tuesday
`{<58, 58–62, 62–66, …}`. The mapping:

| Venue                                       | Library                                                                  |
|---------------------------------------------|--------------------------------------------------------------------------|
| Underlying = today's NYC max temp (°F)      | `y` is a length-N vector of realized temps                               |
| One ladder per day, edges differ            | `edges_per_row[i]` = day `i`'s edges                                     |
| 5–7 mutually-exclusive YES contracts        | `BracketLadder(edges_per_row=..., include_tail_buckets=True)`            |
| Outermost `< X` and `> Y` "tail" contracts  | `include_tail_buckets=True` adds them; per-entity rows then sum to 1.0   |
| YES pays $1 if temp falls in bracket        | `fair_price` is `P(lo ≤ temp < hi)` for that row                         |
| Calibration check after settlement          | `score.brier_bracket(contracts, edges, y)` on the realized temps         |

When every day shares one edge set (Polymarket weekly weather contracts), pass
`edges_per_row=[edges] * N`. The inner list holds N references to the same
array, so it costs no extra memory.

### Worked mapping: spread / total markets

An NFL spread of "Eagles −3.5" pays YES when `(Eagles − opp) > 3.5`. A total of
"Over 47.5" pays YES when `(Eagles + opp) > 47.5`. Both are single-strike
binaries with paired YES/NO sides:

| Venue                                        | Library                                            |
|----------------------------------------------|----------------------------------------------------|
| Underlying = signed margin (spread)          | `y` is the realized margin per game                |
| Underlying = total points (total)            | `y` is the realized total per game                 |
| Strike = the spread / total number           | `Twin(strike=3.5)` / `Twin(strike=47.5)`           |
| YES and NO sides quoted separately on venue  | Two rows per game, shared `group_id`               |
| YES + NO sum to 1 by construction            | `Twin` rows always sum to 1 within a game          |
| Calibration check after settlement           | `score.log_loss_bracket(...)` on the YES/NO ladder |

For multi-strike lines ("Eagles −3, −3.5, −4"), price the same `dist` through
several `Twin` instances at different strikes. For a one-sided Kalshi
temperature ladder ("above 70", "above 75", "above 80"), use
`ThresholdLadder(strikes=[70, 75, 80])`. It returns survival probabilities at
rising strikes: monotone, and they don't sum to 1.

## Step 3: score the prices

With fair prices in hand, check them against what settled. bracketlearn scores
on two levels, both through `result.to_table(y, metrics=[...])` on a
`WalkForward` run:

- **Distribution metrics** read the predictive density directly: `crps`,
  `log_score`, and `pit`.
- **Contract metrics** read the priced ladder: `brier_bracket` and
  `log_loss_bracket`, each taking the shared `edges=` vector. They answer the
  practical question, were the bracket prices calibrated.

The standalone `score.brier_bracket` and `score.log_loss_bracket` helpers, used
in the [Quickstart](#quickstart-the-three-steps-end-to-end), score a single
`ContractForecast` directly. The docs cover the per-backing scoring math.

## Operating the pipeline

The sections above cover a single fit. These control how `WalkForward` runs
across folds and how the pipeline scales to more data, more sites, and more
targets.

### CV variants

`cv=` takes three modes:

- `"expanding-window"` (default) grows the train window by one chunk per fold.
  Use it for sequential and time-series data.
- `"rolling-window"` slides a fixed-width train window forward and needs
  `rolling_window=<int>`. It forgets old rows, which helps through regime change.
- `"kfold"` runs i.i.d. k-fold. Pass `shuffle=True, random_state=...` to permute
  rows. Use it only when rows are exchangeable.

### Sample weights

`WalkForward(...).fit_predict(model, X, y, ids=..., timestamps=...,
sample_weight=w)` threads `w` through every stage. Trainers whose `fit`
signature accepts `sample_weight=` receive it: EMOS, StackedParametric, NGBoost,
the LightGBM-based QuantileReg / QuantileForest / CumulativeBinary /
TailSpecialist, MixtureNormals, and SklearnPoint when its inner estimator
supports it. `WalkForward` detects the online and sequence trainers without
weight support (OnlineAggregator, RNNHourly) by signature and passes them
through unweighted, so nothing crashes.

### Cross-site partial pooling

Multi-city and multi-entity workloads fit here: Kalshi weather across NYC, CHI,
and LAX; NHL spreads across teams; fixture pricing across players. Pass a per-row
site label through `groups=` and use `HierarchicalNormal`:

```python
from bracketlearn import Pipeline, WalkForward
from bracketlearn.trainers import HierarchicalNormal

hn = Pipeline([HierarchicalNormal()], name="hn")
wf = WalkForward(cv="kfold", n_folds=5, refit_on_full=True)
res = wf.fit_predict(hn, X, y, ids=ids, timestamps=ts, groups=city_id)
hn_pred = wf.predict(X_new, ids=..., timestamps=..., groups=city_id_new)["hn"]
```

Each city earns its own coefficient vector β_s, all shrunk toward a common β₀
with the shrinkage strength learned from data (empirical-Bayes on τ²). A city
with little history borrows strength from the rest; a city with deep history
stays close to its own data. For a city unseen at fit, predictive σ inflates
(and raises by default; set `allow_unseen_sites=True` to opt in).

`groups=` routes through `WalkForward` by signature introspection. A trainer
without a `groups` kwarg ignores it, so you mix `HierarchicalNormal` with
site-blind stages like EMOS or ridge and it runs.

### Multi-target

For `y` of shape `(N, M)`, wrap a single-target model and its `WalkForward`
driver in `MultiOutput`:

```python
from bracketlearn import MultiOutput, Pipeline, WalkForward
from bracketlearn.trainers import EMOS

mt = MultiOutput(
    Pipeline([EMOS()], name="emos"),
    WalkForward(n_folds=5),
    target_names=["high", "low"],
)
result = mt.fit_predict(X, Y, ids=ids, timestamps=ts)
print(result.score(Y, metrics=["crps"]))   # per-target × per-stage
```

Each target trains its own cloned model, with no cross-target sharing.

### Hyperparameter search

`GridSearch` enumerates a param grid, cloning the model and its `WalkForward`
driver at each grid point. (It skips `sklearn.GridSearchCV` because that KFold
would shred time ordering.) Use `node__field` syntax for nested params;
`WalkForward` params like `n_folds` and `cv` appear unprefixed:

```python
from bracketlearn import Pipeline, WalkForward
from bracketlearn.search import GridSearch
from bracketlearn.trainers import EMOS

gs = GridSearch(Pipeline([EMOS()], name="emos"),
                WalkForward(cv="expanding-window", n_folds=5),
                param_grid={"emos__sigma_floor": [0.3, 0.5, 1.0],
                            "n_folds": [3, 5]},
                scoring="crps", refit_node="emos")
gs.fit(X, y, ids=ids, timestamps=ts)
print(gs.best_params_, gs.best_score_)
```

## Out of scope: trade decisions

bracketlearn stops at the fair price. Turning `fair_price` into a position size
stays out, by design. That step holds your private signal: side selection on
correlated ladders, edge gates tuned to liquidity, group Kelly across a bracket,
fee schedules, queue assumptions. Ship a default and it lands wrong for the next
user or leaks the edge of the one who had it. You get the calibrated fair price.
You write the trading layer.

## Status

Version 0.6.0, pre-PyPI. The pieces this README documents are built and covered
by the test suite: the composition API (`Pipeline`, `Stacker`, `WalkForward`),
the trainer families, the five contract adapters, and the distribution- and
contract-level scoring. [CHANGELOG.md](CHANGELOG.md) records the version history
and the migration recipes for past API changes.

## License

MIT.
