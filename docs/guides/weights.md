# Sample weights

`WalkForward(...).fit_predict(model, X, y, ids=..., timestamps=...,
sample_weight=w)` threads `w` through every stage. Common uses:

- **Time-decay weighting**: more recent rows count more.
- **Importance reweighting**: the market regimes that matter most get boosted.
- **Cost-sensitive training**: rows where a wrong forecast costs you get higher
  weight.

```python
import numpy as np

# Exponential time-decay: half-life of 60 rows.
w = np.exp(-np.arange(n)[::-1] / 60.0)

result = wf.fit_predict(model, X, y, ids=ids, timestamps=ts, sample_weight=w)
```

## Which trainers honor weights

Native weighted fits:

- `EMOS`: weighted OLS for both (a, b) and (c, d).
- `StackedParametric`: weighted OLS over upstream μ.
- `MixtureNormals`: weighted per-vendor RMSE.
- `SklearnPoint(estimator)`: forwards to the estimator's `fit(sample_weight=...)`
  when the estimator supports it.

Wrapped through the underlying gradient-boosting or forest library:

- `NGBoostNormal`, `QuantileReg`, `QuantileForest`, `CumulativeBinary`, and
  `TailSpecialist` forward `sample_weight=` to LightGBM, NGBoost, or
  quantile-forest.

Pass-through (no native weight support):

- `OnlineAggregator`: online-learning loss accumulators hold no natural weight
  slot, so the pipeline detects this by signature inspection and skips the
  kwarg.
- `RNNHourly`: sequence-batched SGD takes no per-row weights in the current
  implementation, and the pipeline skips the kwarg the same way.

The pass-through detection reads the signature instead of catching a
`TypeError`, so a missing kwarg never masks an unrelated bug in the trainer.
