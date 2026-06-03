# bracketlearn

**Train a calibrated probabilistic forecast for a continuous quantity,
then price the prediction-market contracts (Kalshi / Polymarket binaries,
brackets, spreads, totals) that pay off on it.**

`bracketlearn` is an sklearn-style framework that does three things:

1. **Forecast a distribution** — `EMOS`, `NGBoostNormal`, `QuantileReg`,
   `MixtureNormals`, `CumulativeBinary`, `TailSpecialist` and friends,
   with sklearn-compatible CV, calibration, and conformal correction.
2. **Convert that distribution into fair prices** for the venue's listed
   contracts — single-threshold binaries, paired YES/NO twins, threshold
   ladders, and bracket ladders with per-row varying edges (the
   daily-rotating ladders Kalshi runs on temperature / GDP / Fed-decision
   contracts; pass repeated edges if every row shares the same grid).
3. **Score those fair prices** against realized outcomes with proper
   scoring rules (CRPS, log-score, PIT) on the distribution side and
   Brier / log-loss on the contract side.

Most probabilistic-forecasting libraries stop at "predict a distribution."
Here every forecast is a typed `DistributionForecast` that knows how to
price the resulting contracts and how to be scored on them.

### Out of scope: trade decisions

The conversion from `fair_price` to a position size is **not** in this
library and won't be. That conversion is where private signal lives —
side selection on correlated ladders, liquidity-aware edge gates, group
Kelly across a bracket, fee schedules, queue assumptions — and shipping
a default would be either presumptuous (wrong for the next user) or
alpha-leaking (right for one user who didn't want it public). `bracketlearn`
gives you the calibrated fair price; the trading layer is yours.

## Install

bracketlearn is not yet on PyPI. Install from source:

```bash
git clone https://github.com/FrederikBenirschke/bracketlearn
pip install -e ./bracketlearn

# With the full set of optional trainers (LightGBM, NGBoost, torch, ...):
pip install -e "./bracketlearn[demo]"
```

PyPI publication is planned; once live the install becomes
`pip install bracketlearn` / `pip install "bracketlearn[demo]"`.

## Adapter catalogue — venue → math

| Adapter                | Pricing                            | Maps to (examples)                                          |
|------------------------|------------------------------------|-------------------------------------------------------------|
| `BinaryAbove(k)`       | `P(X > k)`                         | Kalshi "high above 80°F", "S&P > 5000 by Friday"            |
| `BinaryBelow(k)`       | `P(X ≤ k)`                         | Kalshi "GDP ≤ 2.5%", "low below freezing"                   |
| `Twin(k)`              | paired `P(X > k)` / `P(X ≤ k)`     | Polymarket spread (`Eagles -3.5`), total (`Over 47.5`)      |
| `ThresholdLadder(ks)`  | `[P(X > k_i)]` per strike          | Kalshi multi-threshold temperature ladders                  |
| `BracketLadder(edges_per_row)` | `[P(lo ≤ X < hi)]` per-row edges | Kalshi daily-rotating brackets; Polymarket weather brackets (pass `[edges]*N`) |

All five adapters take any `DistributionForecast` (normal / student-t /
mixture-normal / quantile / bracket backings) and emit a long-form
`ContractForecast` with `fair_price`, `entity_ids`, `group_id`,
`contract_spec`, and provenance.

## Worked mapping — weather markets (Kalshi NYC temperature)

Kalshi lists a daily-rotating bracket ladder on NYC max temperature. The
brackets shift each day — Monday's might be `{<60, 60–65, 65–70, …}`,
Tuesday's `{<58, 58–62, 62–66, …}`. Library mapping:

| Venue                                       | Library                                                                  |
|---------------------------------------------|--------------------------------------------------------------------------|
| Underlying = today's NYC max temp (°F)      | `y` is a length-N vector of realized temps                               |
| One ladder per day, edges differ            | `edges_per_row[i]` = day `i`'s edges                                     |
| 5–7 mutually-exclusive YES contracts        | `BracketLadder(edges_per_row=..., include_tail_buckets=True)`            |
| Outermost `< X` and `> Y` "tail" contracts  | `include_tail_buckets=True` adds them; per-entity rows then sum to 1.0   |
| YES pays $1 if temp falls in bracket        | `fair_price` is `P(lo ≤ temp < hi)` for that row                         |
| Calibration check after settlement          | `score.brier_bracket(contracts, edges, y)` on the realized temps         |

If every day shares the same edges (Polymarket-style weekly weather
contracts), pass `edges_per_row=[edges] * N` — the inner list holds N
references to the same array, so there's no memory cost.

## Worked mapping — spread / total markets

An NFL spread of "Eagles −3.5" pays YES if `(Eagles − opp) > 3.5`. A total
of "Over 47.5" pays YES if `(Eagles + opp) > 47.5`. Both are single-strike
binaries with paired YES/NO sides:

| Venue                                        | Library                                            |
|----------------------------------------------|----------------------------------------------------|
| Underlying = signed margin (spread)          | `y` is the realized margin per game                |
| Underlying = total points (total)            | `y` is the realized total per game                 |
| Strike = the spread / total number           | `Twin(strike=3.5)` / `Twin(strike=47.5)`           |
| YES and NO sides quoted separately on venue  | Two rows per game, shared `group_id`               |
| YES + NO sum to 1 by construction            | `Twin` rows always sum to 1 within a game          |
| Calibration check after settlement           | `score.log_loss_bracket(...)` on the YES/NO ladder |

For multi-strike lines ("Eagles −3, −3.5, −4"), price the same `dist`
through several `Twin` instances at different strikes. For a one-sided
multi-strike Kalshi temperature ladder ("above 70", "above 75", "above
80"), use `ThresholdLadder(strikes=[70, 75, 80])` — survival probabilities
at increasing strikes, monotone but not summing to 1.

## Worked example — synthetic NYC max-temperature contracts

A short demo: synthetic weather features → fit EMOS → price the four
prediction-market shapes you'd actually see on the venue → score the
fair prices against realized outcomes.

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

# --- fit EMOS (ensemble-mean + spread regression) ---
emos = EMOS().fit(X_tr, y_tr)
dist = emos.predict_dist(
    X_te,
    ids=np.arange(50),
    timestamps=np.arange(50, dtype=float),
)
# dist.params["mu"], dist.params["sigma"] now hold (50,) forecasts.

# --- price the contracts you'd see on a prediction market ---

# (1) Single threshold: "high above 75°F today"
fair_above_75 = BinaryAbove(strike=75.0).price(dist).fair_price
# fair_above_75[i] = model P(high_i > 75) — feed this into your own
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

# --- score the fair prices against the realized outcomes ---
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

The output of `.price(dist)` is a `ContractForecast` with a `fair_price`
array, the typed `entity_ids` / `contract_ids` / `group_id` indexing you
need to align it with venue quotes, and provenance metadata. What you
do with those fair prices — gate on edge, size by Kelly, hedge across a
ladder — is the trading layer you write on top.

Wrap this whole flow in `ForecastPipeline` to get CV, calibration, and
conformal correction on the distribution before pricing — see the longer
example below.

## Pipeline quick start

```python
import numpy as np
from sklearn.linear_model import RidgeCV

from bracketlearn import (
    BracketLadder, CalibratedForecaster, EMOS, ForecastPipeline,
    GlobalResidual, Isotonic, LiftedForecaster, QuantileReg, SklearnPoint,
)

edges = np.linspace(0, 100, 11)   # 10 brackets

pipeline = ForecastPipeline(
    steps=[
        ("ridge", LiftedForecaster(SklearnPoint(RidgeCV()), GlobalResidual())),
        ("emos",  CalibratedForecaster(EMOS(), Isotonic(edges=edges))),
        ("qreg",  QuantileReg(n_estimators=100)),
    ],
    cv="expanding-window", n_folds=5,
)

result = pipeline.fit_predict(X, y, ids=ids, timestamps=ts)

# Distribution-level metrics on OOF predictions.
print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

# Bracket-contract metrics.
ladder = BracketLadder(edges=edges)
print(result.to_table(y, metrics=["log_loss_bracket", "brier_bracket"],
                      ladder=ladder))

# Predict on truly unseen data using each stage's full-train refit.
new_dists = pipeline.predict(X_new, ids=new_ids, timestamps=new_ts)
```

## sklearn contract

Every forecaster, lifter, and calibrator inherits from `BaseEstimator` and
supports `get_params` / `set_params` / `clone()`. The pipeline clones each
stage's forecaster before every fold's fit, so the user-supplied
instances are never mutated and can be safely reused across pipelines.

## Concepts

Five protocols, no inheritance maze:

| Protocol          | Input → Output                                | Examples                                                       |
|-------------------|-----------------------------------------------|----------------------------------------------------------------|
| `PointForecaster` | `X → PointForecast` (μ̂)                     | `SklearnPoint(Ridge())`, `OnlineAggregator`, `RNNHourly`       |
| `DistForecaster`  | `X → DistributionForecast`                    | `EMOS`, `NGBoostNormal`, `QuantileReg`, `CumulativeBinary`     |
| `Lifter`          | `PointForecast → DistributionForecast`        | `GlobalResidual`, `StudentTResidual`, `GARCHResidual`          |
| `Calibrator`      | `DistributionForecast → DistributionForecast` | `Isotonic`, `ConformalCalibrate`                               |
| `ContractAdapter` | `DistributionForecast → ContractForecast`     | `BinaryAbove`, `BinaryBelow`, `Twin`, `ThresholdLadder`, `BracketLadder` |

Compose `PointForecaster + Lifter` with `LiftedForecaster`, and
`DistForecaster + Calibrator` with `CalibratedForecaster`. Pipeline stays a
flat `[(name, forecaster)]` list — sklearn-style.

## Distribution backings

`DistributionForecast` is an `abc.ABC` base with five concrete
subclasses. Each subclass owns typed storage and its own math; metrics
and adapters dispatch via `isinstance` (or the compat `dist.backing`
property).

| Subclass                  | Storage                                | Math notes                                            |
|---------------------------|----------------------------------------|-------------------------------------------------------|
| `NormalForecast`          | `mu, sigma` per row                    | Closed-form scipy.stats.norm                          |
| `StudentTForecast`        | `mu, sigma, df` per row                | Closed-form scipy.stats.t; requires df > 2            |
| `MixtureNormalForecast`   | `weights, mus, sigmas` per row (N, K)  | CDF = Σ w_k Φ((x−μ_k)/σ_k); PPF via bisection        |
| `QuantileForecast`        | shared `taus` + per-row `qvals` (N, Q) | Pinball-trapezoidal CRPS; `TailPolicy` required      |
| `BracketForecast`         | per-row `edges` (N, B+1) + `probs`     | Uniform-within-bin; NaN-padded ragged rows supported |

Construct via the subclass directly:

```python
from bracketlearn import NormalForecast
d = NormalForecast.from_arrays(
    mu=mu, sigma=sigma,
    ids=ids, timestamps=ts, provenance=prov,
)
```

The `DistributionForecast.from_*` classmethods are kept as routing
shims (`from_normal` → `NormalForecast.from_arrays`, etc.).

### Per-row brackets

`BracketForecast.edges` is `(N, B+1)`. Each row has its own bracket
grid — the Kalshi temperature contract listed on May 26 doesn't share
edges with the one listed on May 27, and bracketlearn doesn't pretend
it does. Ragged-row support is via NaN padding: row i's valid prefix
is the first `B_i + 1` non-NaN edges and the first `B_i` non-NaN
probs.

`BracketForecast.from_arrays` also accepts a 1-D shared edge vector,
broadcasting it to all rows — so callers that genuinely use a shared
ladder pay no ergonomic cost. `BracketForecast.shared_edges()` returns
the 1-D vector iff every row's edges are identical and not NaN-padded;
raises otherwise. Use it from legacy code that still assumes a shared
ladder.

### The `integrate()` bridge

Every `DistributionForecast` subclass implements
`integrate(edges_per_row) → BracketForecast`. This is the single place
where "continuous distribution" becomes "discrete distribution on a
specific grid":

```python
# EMOS emits a NormalForecast; price it on per-row Kalshi ladders.
normal_dist = emos.predict_dist(X, ids=ids, timestamps=ts)
bracket_dist = normal_dist.integrate(edges_per_row)
# bracket_dist.probs has shape (N, B_max) with each row's prob mass on
# its own grid (NaN-padded if rows differ in length).
```

`edges_per_row` accepts: 1-D shared `(B+1,)`, 2-D dense `(N, B+1)`, or
a length-N sequence of 1-D arrays (NaN-padded internally). Each row
is renormalised to sum to 1; rows that land entirely outside the
distribution's support raise (no silent uniform fabrication).

### Estimator families

The trainers group into six families by **what they model**. Pick the
family by the shape of the signal you have; within a family the members
trade off linearity, priors, and compute.

| Family | Estimators | What it models |
|---|---|---|
| **Point** | `SklearnPoint`, `OnlineAggregator`, `RNNHourly` | a single μ̂ per row; lift to a distribution with a residual σ (or a calibration stage) |
| **Parametric distribution** | `EMOS`, `HeteroscedasticNormal`, `NGBoostNormal`, `MixtureNormals`, `BayesianRidge`, `HierarchicalNormal` | a closed-form density (Normal / mixture) whose moments are functions of the features |
| **Quantile / non-parametric** | `QuantileReg`, `QuantileForest` | a quantile function / empirical CDF — no distributional shape assumed |
| **Bracket-native** | `CumulativeBinary`, `TailSpecialist`, `CDFBoostBracket` (+ the `BracketExpander` entry point) | bracket / cutpoint indicators directly on each row's own grid |
| **Stacking / combiners** | `StackedParametric` (alias `Stacking`), `BMAStacking`, `BracketStacking`, `LinearPoolDist`, `DistAsFeatures` | a combination of upstream forecasts (parametric meta-learner, Bayesian average, opinion pool) |
| **Baselines** | `Persistence`, `PersistenceDist`, `EmpiricalDistribution` | reference forecasts to beat; plus convenience factories `ridge`, `emos_calibrated` |

Within the **parametric** family the mean/variance flexibility ladder is
the thing to know:

- `EMOS` — affine mean in `ens_mean`, scale a fixed function of `ens_std`.
  Two hard-wired inputs.
- `HeteroscedasticNormal` — the feature-driven generalisation: `μ = Xμ·βμ`,
  `log σ = Xσ·βσ`, so *any* columns (cloud, wind, dewpoint, spread, …) can
  drive **both** the location and the width, with readable linear
  coefficients. `EMOS` is the special case `Xμ=[ens_mean]`,
  `Xσ=[ens_std]`.
- `NGBoostNormal` — same `(μ̂, σ̂)`-from-features target as
  `HeteroscedasticNormal` but gradient-boosted (non-linear, higher
  variance at low N, not interpretable).
- `MixtureNormals` — multimodal, for bi-/multi-modal outcomes.
- `BayesianRidge` / `HierarchicalNormal` — conjugate priors / cross-site
  partial pooling for small samples.

### Distribution-first vs bracket-aware trainers

Orthogonal to the families above, trainers split by **fit interface** into
two modes:

- **Distribution-first** (`EMOS`, `NGBoostNormal`, `MixtureNormals`,
  `QuantileReg`, `QuantileForest`, `StackedParametric` (alias
  `Stacking`), `BMAStacking`, `BayesianRidge`, `HierarchicalNormal`,
  `OnlineAggregator`, `RNNHourly`, `ridge`, `emos_calibrated`):
  never see brackets at fit time. Fit on `(X, y)`, emit a
  continuous-ish distribution. Call `.integrate(edges_per_row)` to
  price on a specific grid.
- **Bracket-aware** (`CumulativeBinary`, `TailSpecialist`,
  `CDFBoostBracket`): train on bracket-derived indicators. Each takes
  a `cutpoints_by_id` or `brackets_by_id` dict (id → 1-D edge array)
  at construction so per-row grids flow through fit and predict.
  Their `fit()` signatures require an explicit `ids=` kwarg; inside
  `ForecastPipeline` this is forwarded automatically.

  For the "use any sklearn classifier or regressor" entry point, use
  `BracketExpander` (in `bracketlearn.transformers`): it owns the
  per-row → per-(row, bracket) reshape, leaving model choice and
  target construction to the caller. `fit_transform(X, y, ids=...)`
  returns `(X_expanded, y_expanded)` where `X_expanded` is
  `(M, F+2)` with `[..., lo, hi]` appended and `y_expanded` is the
  default bracket-hit indicator `1[y ∈ [lo, hi))`. Fit any sklearn
  estimator on those arrays; pack predictions back into a
  row-renormalised `BracketForecast` via `assemble_dist`.

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

  For a custom per-(row, bracket) target (mispricing residual,
  importance-weighted hit, etc.), build it on top of `fit_transform`
  output — the expander has no opinion about the loss.

## CV variants

`cv=` accepts three modes:

- `"expanding-window"` (default) — train window grows by one chunk per fold;
  use for sequential / time-series data.
- `"rolling-window"` — fixed-width train window slides forward; requires
  `rolling_window=<int>`. Forgets old rows; useful for regime change.
- `"kfold"` — i.i.d. k-fold; pass `shuffle=True, random_state=...` to
  permute rows. Use only when rows are exchangeable.

## Sample weights

`fit_predict(X, y, ids=..., timestamps=..., sample_weight=w)` threads `w`
through every stage. Trainers whose `fit` signature accepts
`sample_weight=` get it (EMOS, Stacking, NGBoost, LightGBM-based
QuantileReg/QuantileForest/CumulativeBinary/TailSpecialist, MixtureNormals,
SklearnPoint when the inner estimator supports it). Online/sequence
trainers without weight support (OnlineAggregator, RNNHourly) are detected
by signature and pass through unweighted — no silent crash.

## Cross-site partial pooling

For multi-city / multi-entity workloads — Kalshi weather contracts
across NYC / CHI / LAX, NHL spreads across teams, fixture pricing
across players — pass a per-row site label via `groups=` and use
`HierarchicalNormal`:

```python
from bracketlearn import ForecastPipeline, HierarchicalNormal

p = ForecastPipeline(steps=[("hn", HierarchicalNormal())], cv="kfold", n_folds=5)
res = p.fit_predict(X, y, ids=ids, timestamps=ts, groups=city_id)
hn_pred = p.predict(X_new, ids=..., timestamps=..., groups=city_id_new)["hn"]
```

Each city gets its own coefficient vector β_s, all shrunk toward a
common β₀ with shrinkage strength learned from data (empirical-Bayes
on τ²). Cities with little history borrow strength from the others;
cities with lots of history stay close to their own data. Predictive
σ inflates automatically for cities not seen at fit (raises by
default — set `allow_unseen_sites=True` to opt in).

`groups=` routes through the pipeline by signature introspection:
trainers without a `groups` kwarg silently ignore it, so mixing
`HierarchicalNormal` with site-blind stages (EMOS, ridge, …) just
works.

## Multi-target

For `y` of shape `(N, M)`, wrap a single-target pipeline:

```python
from bracketlearn.multitarget import MultiOutputForecastPipeline

mt = MultiOutputForecastPipeline(pipeline, target_names=["high", "low"])
result = mt.fit_predict(X, Y, ids=ids, timestamps=ts)
print(result.score(Y, metrics=["crps"]))   # per-target × per-stage
```

Each target gets its own cloned pipeline — no cross-target sharing.

## Hyperparameter search

`GridSearch` enumerates a param grid against the pipeline's own CV (we
do not reuse `sklearn.GridSearchCV` because its KFold would destroy time
ordering). Use `stage__field` syntax for nested params:

```python
from bracketlearn.search import GridSearch

gs = GridSearch(pipeline,
                param_grid={"emos__sigma_floor": [0.3, 0.5, 1.0],
                            "n_folds": [3, 5]},
                scoring="crps", refit_stage="emos")
gs.fit(X, y, ids=ids, timestamps=ts)
print(gs.best_params_, gs.best_score_)
```

## Status

Unreleased — `HeteroscedasticNormal` added to the parametric family:
distributional linear regression with a feature-driven mean **and**
feature-driven (log) scale (`μ = Xμ·βμ`, `log σ = Xσ·βσ`), fit by MLE.
The interpretable generalisation of `EMOS` and linear counterpart to
`NGBoostNormal`. See the Estimator-families table above.

v0.6.0 — `Backing` / `ParametricFamily` enums removed along with the
`DistributionForecast.backing` / `.family` properties. The enums were
compat shims carried over from v0.3.0 when the class became an
`abc.ABC` base; `isinstance(dist, NormalForecast)` etc. is the
supported dispatch. See [CHANGELOG.md](CHANGELOG.md) for the
migration recipe.

v0.5.0 — `BracketClassifier` / `BracketRegressor` removed; their two
conflated concerns (per-row → per-(row, bracket) reshape, plus model
fit on the augmented design) split into the new
`bracketlearn.BracketExpander` transformer + plain sklearn `.fit` on
the caller's chosen estimator. Custom per-(row, bracket) targets
(mispricing residuals, importance-weighted hits) now compose by
construction instead of requiring a fork. See
[CHANGELOG.md](CHANGELOG.md) for the migration recipe.

v0.4.0 — three Bayesian trainers added (`BayesianRidge`,
`BMAStacking`, `HierarchicalNormal`); pipeline gains a `groups=` kwarg
that routes site labels to trainers whose `fit` accepts them and is
silently ignored by site-blind stages. Monolithic `forecast.py` and
`trainers.py` split into typed subpackages.

v0.3.0 — `DistributionForecast` becomes an `abc.ABC` base with five
concrete subclasses; `BracketForecast` stores per-row edges natively;
bracket-aware trainers (`CumulativeBinary`, `TailSpecialist`,
`CDFBoostBracket`) and the `Isotonic` calibrator switch to id-keyed
dict APIs so each row carries its own bracket grid. New
`DistributionForecast.integrate(edges_per_row)` lifts any subclass to
a per-row `BracketForecast`.

v0.2 baseline carries forward: protocols, three CV modes
(expanding-window / rolling-window / kfold), sample-weight threading,
multi-target wrapper, grid-search wrapper, 19 trainers, 6
prediction-market adapters, full distribution and contract-level
scoring. See `bracketlearn/examples/` for runnable demos.

Not yet:
- Vanilla options / option-spread adapters (intentionally out of scope —
  prediction-market binaries only)
- Quantile-backed `DistributionForecast` requires a `TailPolicy` for
  `cdf` / `ppf` / `pdf` / `mean` / `variance` / `sample`; calling those
  without one raises `NotImplementedError`. Constructor demands the
  policy explicitly, so the failure is at construction time, not silent.
  Only `TailRule.clip()` is implemented; if your use case needs
  smoother tail extrapolation (`gpd`, slope-matched Gaussian, ...)
  open an issue with the contract shape that requires it.

## License

MIT.
