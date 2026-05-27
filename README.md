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

PyPI publication is planned for the `v0.2.0` tag; once live the install
becomes `pip install bracketlearn` / `pip install "bracketlearn[demo]"`.

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
