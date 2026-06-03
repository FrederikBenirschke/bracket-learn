# Sample weights

`WalkForward(...).fit_predict(model, X, y, ids=..., timestamps=...,
sample_weight=w)` threads `w` through every stage. Common uses:

- **Time-decay weighting**: more recent rows count more.
- **Importance reweighting**: market regimes that matter most get boosted.
- **Cost-sensitive training**: rows where a wrong forecast is expensive
  get higher weight.

```python
import numpy as np

# Exponential time-decay: half-life of 60 rows.
w = np.exp(-np.arange(n)[::-1] / 60.0)

result = wf.fit_predict(model, X, y, ids=ids, timestamps=ts, sample_weight=w)
```

## Which trainers honor weights

Native weighted fits:

- `EMOS` — weighted OLS for both (a, b) and (c, d).
- `StackedParametric` — weighted OLS over upstream μ.
- `MixtureNormals` — weighted per-vendor RMSE.
- `SklearnPoint(estimator)` — forwarded to estimator's `fit(sample_weight=...)`
  if the estimator supports it.

Wrapped through the underlying gradient-boosting / forest library:

- `NGBoostNormal`, `QuantileReg`, `QuantileForest`, `CumulativeBinary`,
  `TailSpecialist` — `sample_weight=` forwarded to LightGBM / NGBoost /
  quantile-forest.

Pass-through (no native weight support):

- `OnlineAggregator` — online-learning loss accumulators don't have a
  natural weight slot; the pipeline detects this via signature inspection
  and skips the kwarg.
- `RNNHourly` — same; sequence-batched SGD doesn't accept per-row weights
  in the current implementation.

The pass-through detection is signature-based, not TypeError-based, so a
missing kwarg won't mask an unrelated bug in the trainer.
