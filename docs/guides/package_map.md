# Package map

Each module's home in the tree, with the reason it sits there. The package
splits into four layers. **Compose** builds and cross-validates models, **data
types** define what a forecast *is*, **trainers** produce a forecast, and
**pricing / scoring** covers what is done with one.

## Composition & orchestration

| Module | Owns |
|---|---|
| `bracketlearn.pipeline` | `Pipeline`, a sequential chain of stages forming one `DistForecaster`, and `PipelineResult`, a leaderboard of OOF dists indexed as `result[name]`. |
| `bracketlearn.compose` | `Stacker`, a parallel combiner over upstream model *objects*, and `WalkForward`, the CV / OOF driver with `fit_predict` and `predict`. |
| `bracketlearn.protocols` | The five stage protocols, `PointForecaster`, `DistForecaster`, `Lifter`, `Calibrator`, and `Transformer`. |
| `bracketlearn.base` | `BaseEstimator` and `clone`, the sklearn-style get/set-params contract every stage inherits. |

## Data types: what a forecast is

| Module | Owns |
|---|---|
| `bracketlearn.forecast` | `DistributionForecast` (ABC) plus the five backings (`NormalForecast`, `StudentTForecast`, `MixtureNormalForecast`, `QuantileForecast`, `BracketForecast`), plus `PointForecast`, `ContractForecast`, `ProvenanceMeta`, and `TailPolicy`. Each backing owns its own CRPS / CDF / `integrate` math. |

## Trainers: what produces a forecast

The package re-exports all trainers from `bracketlearn.trainers`, grouped by
what the trainer models.

| Module | Trainers |
|---|---|
| `trainers.point` | `SklearnPoint`, `OnlineAggregator`, `RNNHourly`, which emit a μ̂ that a `Lifter` raises to a dist. |
| `trainers.parametric` | `EMOS`, `HeteroscedasticNormal`, `NGBoostNormal`, `MixtureNormals`, `BayesianRidge`, `HierarchicalNormal`, all closed-form densities. |
| `trainers.quantile` | `QuantileReg`, `QuantileForest`, giving quantile functions and empirical CDFs. |
| `trainers.bracket` | `CumulativeBinary`, a bracket-native cutpoint classifier on each row's own grid. |
| `trainers.combiners` | Everything that combines **upstream** forecasts, namely `StackedParametric`, `BMAStacking`, `DistAsFeatures`, `BracketStacking`, `LinearPoolDist`, `TailSpecialist`, `CDFBoostBracket`. |
| `bracketlearn.baselines` | `EmpiricalDistribution`, `Persistence`, `PersistenceDist`, the floors to beat. |

The split follows *what is modelled*. The bracket-emitting combiners
(`TailSpecialist`, `CDFBoostBracket`) consume upstream forecasts, the defining
trait of a combiner, so they sit with the other combiners rather than in
`trainers.bracket` despite emitting brackets.

## Transforms, lifting, calibration

| Module | Owns |
|---|---|
| `bracketlearn.transform` | `GroupByZScore` and `IdentityTransformer`, the `Transformer` stages that normalise features and target and inverse-map the predicted dist. |
| `bracketlearn.lift` | Lifters (`GlobalResidual`, `StudentTResidual`, `GARCHResidual`) and calibrators (`Isotonic`, `ConformalCalibrate`, `PITCalibrate`). |
| `bracketlearn.transformers` | `BracketExpander`, the per-row to per-(row, bracket) reshape for "use any sklearn estimator" workflows. |
| `bracketlearn.restrict` | `BracketMask`, which restricts a bracket dist to a sub-grid. |

## Pricing & scoring

| Module | Owns |
|---|---|
| `bracketlearn.adapters` | `BracketLadder`, `BinaryAbove`, `BinaryBelow`, `Twin`, `ThresholdLadder`, which turn a dist into priced `ContractForecast`s. |
| `bracketlearn.score` | CRPS / log-score / PIT / `log_loss_bracket` / `brier_bracket`, the reference-relative value metrics (`edge_alignment`, `edge_alignment_costed`, `value_report` and their `_bracket` forms), `bootstrap_ci` for clustered confidence intervals, and the `to_point` helper. Imported from `bracketlearn.score`, not re-exported at the top level. |

## Higher-level helpers

| Module | Owns |
|---|---|
| `bracketlearn.search` | `GridSearch`, a time-aware hyperparameter search that clones model and `WalkForward` per grid point. |
| `bracketlearn.multitarget` | `MultiOutput`, which wraps a single-target model for `(N, M)` targets. |
| `bracketlearn.persistence` | `save` / `load` / `envelope_info`, a versioned pickle envelope. |
