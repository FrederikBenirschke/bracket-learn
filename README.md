# bracketlearn

Sklearn-style framework for **probabilistic forecasting** + **bracket-contract pricing**.

Built for the case where you predict a continuous quantity (temperature, score
margin, asset return) and need to price a ladder of binary contracts
("HIGH > 75°F?", "score in [10, 20)?") against the forecast distribution.

## Why

Most probabilistic forecasting libraries stop at "predict a distribution."
bracketlearn keeps going: every forecast has a typed `DistributionForecast`
that knows how to convert itself onto a bracket ladder and price the
resulting contracts. Calibration, conformal correction, and tail
specialisation are first-class transformer stages — not glue code in your
notebook.

## Install

```bash
pip install bracketlearn

# With the full set of optional trainers:
pip install "bracketlearn[demo]"
```

## Quick start

```python
import numpy as np
from sklearn.linear_model import RidgeCV

from bracketlearn.adapters import BracketLadder
from bracketlearn.composite import CalibratedForecaster, LiftedForecaster
from bracketlearn.lift import GlobalResidual, Isotonic
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.trainers import EMOS, QuantileReg, SklearnPoint

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
print(new_dists["qreg"].params)
```

## sklearn contract

Every forecaster, lifter, and calibrator inherits from `BaseEstimator` and
supports `get_params` / `set_params` / `clone()`. The pipeline clones each
stage's forecaster before every fold's fit, so the user-supplied
instances are never mutated and can be safely reused across pipelines.

## Concepts

Five protocols, no inheritance maze:

| Protocol         | Input → Output                         | Examples |
| ---------------- | -------------------------------------- | -------- |
| `PointForecaster`| `X → PointForecast` (μ̂)              | `SklearnPoint(Ridge())`, `OnlineAggregator`, `RNNHourly` |
| `DistForecaster` | `X → DistributionForecast`             | `EMOS`, `NGBoostNormal`, `QuantileReg`, `CumulativeBinary` |
| `Lifter`         | `PointForecast → DistributionForecast` | `GlobalResidual` |
| `Calibrator`     | `DistributionForecast → DistributionForecast` | `Isotonic`, `ConformalCalibrate` |
| `ContractAdapter`| `DistributionForecast → ContractForecast` | `BracketLadder` |

Compose `PointForecaster + Lifter` with `LiftedForecaster`, and
`DistForecaster + Calibrator` with `CalibratedForecaster`. Pipeline stays a
flat `[(name, forecaster)]` list — sklearn-style.

## Distribution backings

A `DistributionForecast` can carry any of four backings; metrics dispatch
on the type:

- **parametric** (`normal`, `mixture_normal`) — closed-form CRPS / log-score / CDF.
- **quantile** — array of `qvals` at fixed `taus`; CRPS via pinball trapezoidal
  integral; tail policy controls extrapolation beyond outermost quantile.
- **bracket** — array of `probs` on `edges`; uniform-within-bin density.
- **empirical** (planned) — array of `members`.

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

v0.2 — protocols, three CV modes (expanding-window / rolling-window / kfold),
sample-weight threading, multi-target wrapper, grid-search wrapper, 14 trainers,
4 backings, full distribution and bracket-level scoring. See
`bracketlearn/examples/` for runnable demos.

Not yet:
- Persistence / pickling
- Empirical + student_t backings
- `gpd` / `gaussian_match` tail rules (only `clip`)

## License

MIT.
