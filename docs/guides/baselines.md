# Baselines

Every probabilistic-forecasting paper compares against trivial baselines. When
a quantile-regression-stacked ensemble only ties one of these, either the
features carry no signal or the validation leaks. bracketlearn ships two, both
`BaseEstimator` subclasses that drop unchanged into a `Pipeline` run under
`WalkForward`.

## `EmpiricalDistribution`: the climatology floor

This baseline ignores `X` entirely and emits the empirical CDF of training `y`
as a fixed quantile-backed forecast.

```python
from bracketlearn import EmpiricalDistribution

emp = EmpiricalDistribution()
emp.fit(X_train, y_train)
dist = emp.predict_dist(X_test)
# dist.qvals is the same vector repeated N times, with no X-conditioning.
```

The default τ-grid is `(0.05, 0.10, ..., 0.95)`. Pass `taus=(...)` to
override.

### Why this is the floor

`EmpiricalDistribution` honors marginal calibration by construction. Its CDF is
the empirical CDF, so PIT values on i.i.d. holdout data come out uniform. A
model that fails to beat its CRPS has learned no conditional structure from `X`.

### sample_weight

Weighted quantiles are supported, via cumulative-weight interpolation on
sorted-y. Use it to upweight recent observations, giving a rolling climatology,
or to upweight rare regimes.

```python
sw = np.where(y_train > extreme_threshold, 10.0, 1.0)
emp.fit(X_train, y_train, sample_weight=sw)
```

## `Persistence`: the autoregressive lag-k baseline

`mu_t = y_{t - lag}`. It records the last `lag` training y values at fit
time and tiles them cyclically across the inference horizon at predict time.

```python
from bracketlearn import Persistence

p = Persistence(lag=1)
p.fit(X_train, y_train)
pred = p.predict(X_test)   # constant: last training y everywhere
```

### Lag semantics

`Persistence` is a `PointForecaster`, so it carries no σ. Pair it with
`GlobalResidual`, fit on OOF residuals, or another `Lifter` for distributional
output.

| `lag` | behaviour | natural use |
|-------|-----------|-------------|
| 1 | predict y_{T-1} for every inference row | random-walk / "no signal" floor |
| 24 | replay the last 24 hours: row i predicts y_{T-24+(i mod 24)} | hourly diurnal-cycle baseline (bike-share, electricity load) |
| 168 | replay the last week | weekly seasonal baseline |

### CV constraints

`Persistence` only makes sense under time-aware CV. Combine it with
`cv="expanding-window"` or `cv="rolling-window"`. On `cv="kfold"` with shuffled
rows, "last y" loses its meaning and the metric turns into a random number.

```python
from bracketlearn import Pipeline, WalkForward
from bracketlearn.lift import GlobalResidual

persist24 = Pipeline([Persistence(lag=24), GlobalResidual()], name="persist24")
result = WalkForward(cv="expanding-window", n_folds=5).fit_predict(
    persist24, X, y, ids=ids, timestamps=ts,
)
```

## When to use which

- **Tabular data with no time axis.** `EmpiricalDistribution` is the
  floor.
- **Hourly / daily time series with a strong seasonal cycle.**
  `Persistence(lag=24)` or `Persistence(lag=168)` is the floor, and beating it
  means more than the cycle has been learned.
- **Both** are cheap to fit and can be added to a `WalkForward` run as named
  models. The `result.score(y)` table then prints baselines and models side by
  side, and skill scores can be read off directly.
