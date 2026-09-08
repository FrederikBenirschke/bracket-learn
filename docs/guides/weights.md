# Sample weights

`WalkForward(...).fit_predict(model, X, y, ids=..., timestamps=...,
sample_weight=w)` threads `w` through every stage. Common uses follow.

- **Time-decay weighting** makes more recent rows count more.
- **Importance reweighting** boosts the market regimes that matter most.
- **Cost-sensitive training** gives higher weight to rows where a wrong
  forecast is expensive.

```python
import numpy as np

# Exponential time-decay: half-life of 60 rows.
w = np.exp(-np.arange(n)[::-1] / 60.0)

result = wf.fit_predict(model, X, y, ids=ids, timestamps=ts, sample_weight=w)
```

## Which trainers honor weights

Native weighted fits are these.

- `EMOS` uses weighted OLS for both (a, b) and (c, d).
- `StackedParametric` uses weighted OLS over upstream μ.
- `MixtureNormals` uses a weighted per-vendor RMSE.
- `SklearnPoint(estimator)` forwards to the estimator's `fit(sample_weight=...)`
  when the estimator supports it.

The following are wrapped through the underlying gradient-boosting or forest
library.

- `NGBoostNormal`, `QuantileReg`, `QuantileForest`, `CumulativeBinary`, and
  `TailSpecialist` forward `sample_weight=` to LightGBM, NGBoost, or
  quantile-forest.

These are pass-through, with no native weight support.

- `OnlineAggregator` has online-learning loss accumulators with no natural
  weight slot, so the pipeline detects this by signature inspection and skips
  the kwarg.
- `RNNHourly` uses sequence-batched SGD, which takes no per-row weights in the
  current implementation, and the pipeline skips the kwarg the same way.

The pass-through detection reads the signature instead of catching a
`TypeError`, so a missing kwarg never masks an unrelated bug in the trainer.
