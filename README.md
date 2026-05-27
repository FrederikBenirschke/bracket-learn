# bracketlearn

**Train a probabilistic forecast for a continuous quantity, then price the
prediction-market contracts (Kalshi / Polymarket binaries, brackets,
spreads, totals) that pay off on it.**

`bracketlearn` is an sklearn-style framework that does the two things you
need when trading prediction markets on a continuous underlying:

1. **Forecast a distribution** — `EMOS`, `NGBoostNormal`, `QuantileReg`,
   `MixtureNormals`, `CumulativeBinary`, `TailSpecialist` and friends,
   with sklearn-compatible CV, calibration, and conformal correction.
2. **Convert that distribution into fair prices** for the venue's listed
   contracts — single-threshold binaries, paired YES/NO twins, fixed
   bracket ladders, and per-row varying brackets (the daily-rotating
   ladders Kalshi runs on temperature / GDP / Fed-decision contracts).

Most probabilistic-forecasting libraries stop at "predict a distribution."
Here every forecast is a typed `DistributionForecast` that knows how to
price the resulting contracts.

## Install

bracketlearn is not yet on PyPI. Install from source:

```bash
git clone https://github.com/FrederikBenirschke/bracketlearn
pip install -e ./bracketlearn

# With the full set of optional trainers (LightGBM, NGBoost, torch, ...):
pip install -e "./bracketlearn[demo]"
```

PyPI publication is planned for the `v0.2.0` tag; once live the install
becomes `pip install bracketlearn` / `pip install "bracketlearn[demo]"`.

## Adapter catalogue — venue → math

| Adapter                | Pricing                            | Maps to (examples)                                          |
|------------------------|------------------------------------|-------------------------------------------------------------|
| `BinaryAbove(k)`       | `P(X > k)`                         | Kalshi "high above 80°F", "S&P > 5000 by Friday"            |
| `BinaryBelow(k)`       | `P(X ≤ k)`                         | Kalshi "GDP ≤ 2.5%", "low below freezing"                   |
| `Twin(k)`              | paired `P(X > k)` / `P(X ≤ k)`     | Polymarket spread (`Eagles -3.5`), total (`Over 47.5`)      |
| `ThresholdLadder(ks)`  | `[P(X > k_i)]` per strike          | Kalshi multi-threshold temperature ladders                  |
| `BracketLadder(edges)` | `[P(lo ≤ X < hi)]` shared edges    | Polymarket weather brackets, fixed weekly contracts         |
| `PerRowBracketLadder`  | per-row edges (each row its own)   | Kalshi daily-rotating brackets (edges shift day-by-day)     |

All six adapters take any `DistributionForecast` (normal / student-t /
mixture-normal / quantile / bracket backings) and emit a long-form
`ContractForecast` with `fair_price`, `entity_ids`, `group_id`,
`contract_spec`, and provenance.

## Worked example — synthetic NYC max-temperature contracts

A 10-line demo: synthetic weather features → fit EMOS → price the four
prediction-market shapes you'd actually see on the venue → flag +EV.

```python
import numpy as np
from bracketlearn import (
    EMOS, BracketLadder, PerRowBracketLadder, BinaryAbove, Twin,
)

# --- synthetic NYC max-temperature data ---
rng = np.random.default_rng(0)
N = 200
day = np.arange(N)
season = 70 + 15 * np.sin(2 * np.pi * day / 365.0)
prior_high = season + rng.normal(0, 4, N)
X = np.column_stack([prior_high, season])
y = season + 0.6 * (prior_high - season) + rng.normal(0, 5, N)
X_tr, X_te, y_tr = X[:150], X[150:], y[:150]

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
fair = BinaryAbove(strike=75.0).price(dist).fair_price[0]
market = 0.55                       # what the venue is quoting
print(f"BinaryAbove(75)   fair={fair:.3f}  market={market:.3f}  edge={fair-market:+.3f}")
# → fair=0.648  market=0.550  edge=+0.098  ← BUY YES

# (2) Paired YES/NO at 70°F (spread / total style)
twin = Twin(strike=70.0).price(dist)
yes = twin.fair_price[twin.contract_ids == 0][0]
no  = twin.fair_price[twin.contract_ids == 1][0]
print(f"Twin(70)          yes={yes:.3f}  no={no:.3f}  (sum=1.000)")

# (3) Fixed bracket ladder (Polymarket-style weekly contracts)
edges = np.array([0.0, 60.0, 70.0, 80.0, 90.0, 100.0])
ladder = BracketLadder(edges=edges).price(dist)
# 5 contracts per entity: P([0,60)), P([60,70)), P([70,80)), ...

# (4) Kalshi-style daily-rotating brackets (edges differ each day)
edges_per_day = [
    np.array([mu - 10, mu - 3, mu, mu + 3, mu + 10])
    for mu in dist.params["mu"]
]
per_row = PerRowBracketLadder(
    edges_per_row=edges_per_day,
    include_tail_buckets=True,      # add "below" and "above" rows
).price(dist)
# Per-entity rows sum to exactly 1.0.
```

Output for the first entity:

```
BinaryAbove(75)   fair=0.648  market=0.550  edge=+0.098  ← BUY YES
Twin(70)          yes=0.912  no=0.088  (sum=1.000)

BracketLadder fair prices for entity 0:
    [0, 60)   = 0.000
    [60, 70)  = 0.088
    [70, 80)  = 0.635
    [80, 90)  = 0.271
    [90, 100) = 0.006

PerRowBracketLadder (brackets centered on each day's forecast):
    < 67.0          = 0.026
    [67.0, 74.0)    = 0.254
    [74.0, 77.0)    = 0.220
    [77.0, 80.0)    = 0.220
    [80.0, 87.0)    = 0.254
    > 87.0          = 0.026
                sum = 1.000
```

Wrap this in `ForecastPipeline` to get CV, calibration, and conformal
correction — see the longer example below.

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
| `ContractAdapter` | `DistributionForecast → ContractForecast`     | `BinaryAbove`, `Twin`, `BracketLadder`, `PerRowBracketLadder`  |

Compose `PointForecaster + Lifter` with `LiftedForecaster`, and
`DistForecaster + Calibrator` with `CalibratedForecaster`. Pipeline stays a
flat `[(name, forecaster)]` list — sklearn-style.

## Distribution backings

A `DistributionForecast` can carry any of four backings; metrics and
adapters dispatch on the type:

- **parametric** (`normal`, `student_t`, `mixture_normal`) — closed-form
  CRPS / log-score / CDF.
- **quantile** — array of `qvals` at fixed `taus`; CRPS via pinball
  trapezoidal integral; tail policy controls extrapolation beyond
  outermost quantile.
- **bracket** — array of `probs` on `edges`; uniform-within-bin density.

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
3 backings, 6 prediction-market adapters, full distribution and
contract-level scoring. See `bracketlearn/examples/` for runnable demos.

Not yet:
- Empirical backing
- `gpd` / `gaussian_match` tail rules (only `clip`)
- Vanilla options / option-spread adapters (intentionally out of scope —
  prediction-market binaries only)
- Quantile-backed `DistributionForecast` requires a `TailPolicy` for
  `cdf` / `ppf` / `pdf` / `mean` / `variance` / `sample`; calling those
  without one raises `NotImplementedError`. Constructor demands the
  policy explicitly, so the failure is at construction time, not silent.

## License

MIT.
