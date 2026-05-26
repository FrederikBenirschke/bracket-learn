# Baselines

Every probabilistic-forecasting paper compares against trivial baselines.
If your fancy quantile-regression-stacked-ensemble ties one of these,
the features aren't predictive or the validation is leaking. bracketlearn
ships two — both `BaseEstimator` subclasses that slot into
`ForecastPipeline` unchanged.

## `EmpiricalDistribution` — climatology floor

Ignores `X` entirely; emits the empirical CDF of training `y` as a fixed
quantile-backed forecast.

```python
from bracketlearn import EmpiricalDistribution

emp = EmpiricalDistribution()
emp.fit(X_train, y_train)
dist = emp.predict_dist(X_test)
# dist.qvals is the same vector repeated N times — no X-conditioning.
```

The default τ-grid is `(0.05, 0.10, ..., 0.95)`. Pass `taus=(...)` to
override.

### Why this is the floor

`EmpiricalDistribution` honors marginal calibration by construction —
its CDF is the empirical CDF, so PIT values on i.i.d. holdout data are
exactly uniform. A model that doesn't beat its CRPS isn't learning any
conditional structure from `X`.

### sample_weight

Supported: weighted quantiles via cumulative-weight interpolation on
sorted-y. Use it to upweight recent observations (rolling climatology)
or rare regimes.

```python
sw = np.where(y_train > extreme_threshold, 10.0, 1.0)
emp.fit(X_train, y_train, sample_weight=sw)
```

## `Persistence` — autoregressive lag-k baseline

`mu_t = y_{t - lag}`. Records the last `lag` training y values at fit
time; at predict time tiles them cyclically across the inference horizon.

```python
from bracketlearn import Persistence

p = Persistence(lag=1)
p.fit(X_train, y_train)
pred = p.predict(X_test)   # constant: last training y everywhere
```

### Lag semantics

`Persistence` is a `PointForecaster` — no σ. Pair with `GlobalResidual`
(fit on OOF residuals) or another `Lifter` if you want distributional
output.

| `lag` | behaviour | natural use |
|-------|-----------|-------------|
| 1 | predict y_{T-1} for every inference row | random-walk / "no signal" floor |
| 24 | replay the last 24 hours: row i predicts y_{T-24+(i mod 24)} | hourly diurnal-cycle baseline (bike-share, electricity load) |
| 168 | replay the last week | weekly seasonal baseline |

### CV constraints

`Persistence` only makes sense under time-aware CV. Combine with
`cv="expanding-window"` or `cv="rolling-window"` — `cv="kfold"` on
shuffled rows makes "last y" meaningless and the metric becomes a
random number.

```python
from bracketlearn import ForecastPipeline, LiftedForecaster
from bracketlearn.lift import GlobalResidual

pipeline = ForecastPipeline(
    steps=[
        ("persist24", LiftedForecaster(
            Persistence(lag=24), GlobalResidual(), name="persist24",
        )),
    ],
    cv="expanding-window", n_folds=5,
)
```

## When to use which

- **Tabular data with no time axis**: `EmpiricalDistribution` is the
  floor.
- **Hourly / daily time series with a strong seasonal cycle**:
  `Persistence(lag=24)` or `Persistence(lag=168)` is the floor — beating
  it means you learned more than the cycle.
- **Both** are cheap to fit and add to a `ForecastPipeline` as named
  steps; that way your `result.score(y)` table prints baselines and
  models side by side and the reader can read off skill scores.
