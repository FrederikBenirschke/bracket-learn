# Package map

Where each thing lives, and why. The package has four layers: **compose**
(how models are built and cross-validated), **data types** (what a forecast
*is*), **trainers** (what produces a forecast), and **pricing / scoring**
(what you do with one).

## Composition & orchestration

| Module | Owns |
|---|---|
| `bracketlearn.pipeline` | `Pipeline` (sequential chain of stages → one `DistForecaster`), `PipelineResult` (leaderboard of OOF dists, `result[name]`). |
| `bracketlearn.compose` | `Stacker` (parallel combiner over upstream model *objects*), `WalkForward` (the CV / OOF driver: `fit_predict` / `predict`). |
| `bracketlearn.protocols` | The five stage protocols: `PointForecaster`, `DistForecaster`, `Lifter`, `Calibrator`, `Transformer`. |
| `bracketlearn.base` | `BaseEstimator`, `clone` — the sklearn-style get/set-params contract every stage inherits. |

## Data types — what a forecast is

| Module | Owns |
|---|---|
| `bracketlearn.forecast` | `DistributionForecast` (ABC) + the five backings — `NormalForecast`, `StudentTForecast`, `MixtureNormalForecast`, `QuantileForecast`, `BracketForecast` — plus `PointForecast`, `ContractForecast`, `ProvenanceMeta`, `TailPolicy`. Each backing owns its own CRPS / CDF / `integrate` math. |

## Trainers — what produces a forecast

All re-exported from `bracketlearn.trainers`; grouped by **what the trainer
models**:

| Module | Trainers |
|---|---|
| `trainers.point` | `SklearnPoint`, `OnlineAggregator`, `RNNHourly` — emit a μ̂ (lift to a dist with a `Lifter`). |
| `trainers.parametric` | `EMOS`, `HeteroscedasticNormal`, `NGBoostNormal`, `MixtureNormals`, `BayesianRidge`, `HierarchicalNormal` — closed-form densities. |
| `trainers.quantile` | `QuantileReg`, `QuantileForest` — quantile functions / empirical CDFs. |
| `trainers.bracket` | `CumulativeBinary` — bracket-native cutpoint classifier on each row's own grid. |
| `trainers.combiners` | Everything that combines **upstream** forecasts: `StackedParametric`, `BMAStacking`, `DistAsFeatures`, `BracketStacking`, `LinearPoolDist`, `TailSpecialist`, `CDFBoostBracket`. |
| `bracketlearn.baselines` | `EmpiricalDistribution`, `Persistence`, `PersistenceDist` — the floors to beat. |

The split is by *what is modelled*, not output backing — that's why the
bracket-emitting combiners (`TailSpecialist`, `CDFBoostBracket`) live with the
other combiners rather than in `trainers.bracket`: they consume upstream
forecasts, which is the defining trait of a combiner.

## Transforms, lifting, calibration

| Module | Owns |
|---|---|
| `bracketlearn.transform` | `GroupByZScore`, `IdentityTransformer` — `Transformer` stages (normalise features / target, inverse-map the predicted dist). |
| `bracketlearn.lift` | Lifters (`GlobalResidual`, `StudentTResidual`, `GARCHResidual`) and calibrators (`Isotonic`, `ConformalCalibrate`, `PITCalibrate`). |
| `bracketlearn.transformers` | `BracketExpander` — per-row → per-(row, bracket) reshape for "use any sklearn estimator" workflows. |
| `bracketlearn.restrict` | `BracketMask` — restrict a bracket dist to a sub-grid. |

## Pricing & scoring

| Module | Owns |
|---|---|
| `bracketlearn.adapters` | `BracketLadder`, `BinaryAbove`, `BinaryBelow`, `Twin`, `ThresholdLadder` — turn a dist into priced `ContractForecast`s. |
| `bracketlearn.score` | CRPS / log-score / PIT / `log_loss_bracket` / `brier_bracket` and the `to_point` helper. |

## Higher-level helpers

| Module | Owns |
|---|---|
| `bracketlearn.search` | `GridSearch` — time-aware hyperparameter search (clones model + `WalkForward` per grid point). |
| `bracketlearn.multitarget` | `MultiOutput` — wrap a single-target model for `(N, M)` targets. |
| `bracketlearn.persistence` | `save` / `load` / `envelope_info` — versioned pickle envelope. |
