# Catalog — what's available, and when to reach for it

Everything bracketlearn ships, organised by the **five protocol slots** a
stage can fill (see [Concepts](concepts.md) for the protocol table). A
`Pipeline` is a left→right chain of these; a `Stacker` combines upstream
pipelines; `WalkForward` cross-validates. Names are leaderboard labels,
never wiring.

| Slot | Turns | Section |
|---|---|---|
| **Transformer** | features/target → normalised, dist → un-normalised | [Transformers](#transformers) |
| **Forecaster** (point / dist) | `X` → `PointForecast` or `DistributionForecast` | [Forecasters](#forecasters) |
| **Lifter** | `PointForecast` → `DistributionForecast` | [Lifters](#lifters) |
| **Calibrator** | `DistributionForecast` → `DistributionForecast` | [Calibrators](#calibrators) |
| **ContractAdapter** | `DistributionForecast` → priced `ContractForecast` | [Contract adapters](#contract-adapters) |

A full single-model chain reads:

```python
Pipeline([
    GroupByZScore(level_cols=()),   # Transformer  — normalise target
    EMOS(),                         # Forecaster   — closed-form Normal
    Isotonic(),                     # Calibrator   — fix bracket-prob miscalibration
])
```

---

## Forecasters

The trainers group into six families by **what they model**. Pick the
family from the shape of the signal you have; within a family the members
trade off linearity, priors, and compute. All are re-exported from
`bracketlearn.trainers` (and the common ones from the top-level package).

| Family | Estimators | What it models |
|---|---|---|
| **Point** | `SklearnPoint`, `OnlineAggregator`, `RNNHourly` | a single μ̂ per row; lift to a distribution with a [Lifter](#lifters) |
| **Parametric distribution** | `EMOS`, `HeteroscedasticNormal`, `NGBoostNormal`, `MixtureNormals`, `BayesianRidge`, `HierarchicalNormal` | a closed-form density (Normal / mixture) whose moments are functions of the features |
| **Quantile / non-parametric** | `QuantileReg`, `QuantileForest` | a quantile function / empirical CDF — no distributional shape assumed |
| **Bracket-native** | `CumulativeBinary` (+ the `BracketExpander` entry point) | bracket / cutpoint indicators directly on each row's own grid |
| **Combiners** | `StackedParametric`, `BMAStacking`, `BracketStacking`, `LinearPoolDist`, `TailSpecialist`, `CDFBoostBracket`, `DistAsFeatures` | a combination of **upstream** forecasts (parametric meta-learner, Bayesian average, opinion pool, …) |
| **Baselines** | `Persistence`, `PersistenceDist`, `EmpiricalDistribution` | reference forecasts to beat; plus convenience factories `ridge`, `emos_calibrated` |

### The parametric flexibility ladder

Within the parametric family the mean/variance flexibility is the thing to
know — they form a ladder from two hard-wired inputs to fully feature-driven:

- **`EMOS`** — affine mean in `ens_mean`, scale a fixed function of
  `ens_std`. Two hard-wired inputs. The classic ensemble post-processor.
- **`HeteroscedasticNormal`** — the feature-driven generalisation:
  `μ = Xμ·βμ`, `log σ = Xσ·βσ`, so *any* columns (cloud, wind, dewpoint,
  spread, …) can drive **both** the location and the width, with readable
  linear coefficients. `EMOS` is the special case `Xμ=[ens_mean]`,
  `Xσ=[ens_std]`.
- **`NGBoostNormal`** — same `(μ̂, σ̂)`-from-features target as
  `HeteroscedasticNormal` but gradient-boosted: non-linear, higher variance
  at low N, not interpretable. Reach for it when the relationship is
  clearly non-linear and you have the rows.
- **`MixtureNormals`** — multimodal, for bi-/multi-modal outcomes (e.g. a
  bimodal score margin).
- **`BayesianRidge` / `HierarchicalNormal`** — conjugate priors /
  cross-site partial pooling for small samples. ⚠️ `BayesianRidge`'s prior
  shrinks weights toward zero on *unstandardised* features — pair it with a
  `GroupByZScore` transformer or set `standardize=True`.

### Quantile vs parametric

`QuantileReg` / `QuantileForest` assume **no** distributional shape — they
emit `qvals` at fixed `taus` and interpolate. Use them when the residual
distribution is skewed or otherwise non-Gaussian and you don't want to
commit to a parametric family. Tail behaviour beyond the outermost quantile
is governed by the [tail policy](tail_policies.md).

### Distribution-first vs bracket-aware (fit interface)

Orthogonal to the families, trainers split by **what they see at fit time**:

- **Distribution-first** (`EMOS`, `NGBoostNormal`, `MixtureNormals`,
  `QuantileReg`, `QuantileForest`, the parametric/combiner trainers,
  `OnlineAggregator`, `RNNHourly`, `ridge`, `emos_calibrated`): never see
  brackets at fit time. Fit on `(X, y)`, emit a continuous-ish distribution,
  then `.integrate(edges_per_row)` to price on a specific grid.
- **Bracket-aware** (`CumulativeBinary`, `TailSpecialist`,
  `CDFBoostBracket`): train on bracket-derived indicators and take a
  `cutpoints_by_id` / `brackets_by_id` dict (id → 1-D edge array) at
  construction, so per-row grids flow through fit and predict. Their `fit()`
  requires an explicit `ids=` kwarg (a `Pipeline` forwards it automatically).

For the **"use any sklearn classifier/regressor on brackets"** entry point,
use `BracketExpander` (see [bracket_expander](bracket_expander.md)) — it owns
the per-row → per-(row, bracket) reshape and leaves model + target to you.

### Combiners — needing upstream forecasts

The combiner family is special: each takes the **OOF predictions of other
pipelines** as input, so it only makes sense inside a `Stacker`, not a bare
`Pipeline`. See [CV & stacking](cv.md).

- **`StackedParametric`** — a parametric meta-learner over upstream means/sigmas.
- **`BMAStacking`** — Bayesian model averaging (likelihood-weighted mixture).
- **`LinearPoolDist`** — a linear opinion pool (weighted average of CDFs).
- **`BracketStacking`** — stacks in bracket-probability space.
- **`CDFBoostBracket`** — boosts a base CDF on bracket residuals.
- **`TailSpecialist`** — a second model dedicated to the outer brackets,
  blended with the body model.
- **`DistAsFeatures`** — flattens upstream dists into features for an
  arbitrary downstream estimator.

---

## Transformers

A `Transformer` normalises features/target before fitting and **inverts the
predicted distribution back to the original units** — the piece that makes
normalisation native to the pipeline rather than glue code.

| Transformer | What it does | When to use |
|---|---|---|
| `GroupByZScore` | per-group standardized anomaly: learns scale = `std(y − center)` per group (e.g. per station), maps `(v − center)/scale`, and inverse-maps the predicted dist back to real units | the proven weather win — strips a per-group location/scale confound. Use `level_cols=()` for **target-only** z-scoring (right when X holds mixed/heterogeneous columns and the model is scale-invariant, e.g. trees) |
| `IdentityTransformer` | no-op pass-through | the shim shape for plugging a plain sklearn X-only transformer into a pipeline (override `transform`; target + inverse stay identity), or as an explicit "normalise nothing" marker |
| `BracketExpander` | per-row → per-(row, bracket) reshape, appending `[lo, hi]` to each expanded row | the "fit any sklearn estimator on brackets" entry point; see [bracket_expander](bracket_expander.md) |

`GroupByZScore` also takes `spread_cols` (divide-only, for an ensemble std)
and `passthrough_cols` (leave untouched, e.g. binary missing-flags).

---

## Lifters

A `Lifter` turns a bare point forecast (μ̂) into a distribution by supplying
the spread. All three fit on **OOF residuals** (`y − μ̂`), so they need a
point forecaster upstream in the pipeline.

| Lifter | Residual model | When to use |
|---|---|---|
| `GlobalResidual` | one constant σ from OOF residual std (Normal output) | the default — homoscedastic Gaussian residuals; simplest, most robust at low N |
| `StudentTResidual` | MLE `(σ, ν)`, Student-t output, ν clipped to a finite-variance range | residuals fatter-tailed than Gaussian (sports margins, short-horizon returns); ν near the floor flags heavy tails |
| `GARCHResidual` | GARCH(1,1) volatility recursion → per-row σ (one-step); optional Student-t innovations | residual **volatility clusters** over time (returns, vol regimes); gives each row its own σ from the fitted history |

---

## Calibrators

A `Calibrator` maps a distribution to a better-calibrated distribution on a
held-out set. They differ by **which representation they correct** — pick the
one matching your forecaster's backing.

| Calibrator | Operates on | What it fixes | When to use |
|---|---|---|---|
| `Isotonic` | bracket probabilities | a single monotone curve on (predicted prob → realized hit), per cell, then row-renormalised | systematic over/under-statement of bracket probabilities; grid-agnostic (pass `pre_integrate_edges=` to auto-integrate a non-bracket dist first) |
| `PITCalibrate` | the predictive CDF (any backing) | isotonic recalibration of the PIT `u = F̂(y)` toward Uniform(0,1) | whole-distribution miscalibration — U-shaped PIT (over-confident) or humped (under-confident) |
| `ConformalCalibrate` | quantile backings only | per-τ offsets giving finite-sample `(1−τ)` coverage under exchangeability (CQR, Romano 2019) | quantile forecasters where you want a coverage *guarantee*; rejects non-quantile dists loudly |

---

## Contract adapters

A `ContractAdapter` prices a distribution onto the binary contracts a venue
actually lists. See [adapters](adapters.md) for the full venue→math mapping.

| Adapter | Prices | Venue shape |
|---|---|---|
| `BinaryAbove` | `P(X > k) = 1 − cdf(k)` | single "above" threshold ("high above 80°F") |
| `BinaryBelow` | `P(X ≤ k) = cdf(k)` | single "below" threshold ("low below 32°F") |
| `Twin` | paired YES/NO at one strike, summing to 1 | spread / total ("Eagles −3.5", "Over 47.5") |
| `BracketLadder` | full per-row ladder, each row's probs sum to 1 | bracketed markets with **per-row** grids (Kalshi daily-rotating temp brackets) |
| `ThresholdLadder` | one row per `P(X > kᵢ)`, monotone-decreasing, **not** summing to 1 | single-side strike ladders ("high above 70/75/80°F") |

---

## Higher-level helpers

Not stages — they wrap a model or a fitted result:

| Helper | Module | Role |
|---|---|---|
| `WalkForward` | `bracketlearn` | the CV / OOF driver (`fit_predict` / `predict`); see [CV](cv.md) |
| `MultiOutput` | `bracketlearn.multitarget` | wrap a single-target model for `(N, M)` targets; see [multitarget](multitarget.md) |
| `GridSearch` | `bracketlearn.search` | time-aware hyperparameter search; see [search](search.md) |
| `save` / `load` | `bracketlearn.persistence` | versioned pickle envelope; see [persistence](persistence.md) |

For *where each symbol lives in the source tree*, see the
[package map](package_map.md). For *how the pieces compose*, see
[Concepts](concepts.md).
