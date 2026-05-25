# Multi-target

For `Y` of shape `(N, M)`, wrap a single-target pipeline in
`MultiOutputForecastPipeline`:

```python
from bracketlearn.multitarget import MultiOutputForecastPipeline
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.trainers import EMOS

proto = ForecastPipeline(steps=[("emos", EMOS())], n_folds=5)
mt = MultiOutputForecastPipeline(proto, target_names=["high", "low"])
result = mt.fit_predict(X, Y, ids=ids, timestamps=ts)

# Per-target × per-stage metrics.
scores = result.score(Y, metrics=["crps"])
print(scores["high"]["emos"]["crps"])
print(scores["low"]["emos"]["crps"])
```

## Design choice: wrap, don't thread

Each target gets its own cloned pipeline. There is no cross-target sharing.

Why not natively make every `DistributionForecast` carry an `(N, M)`
shape? It would multiply every backing's storage, break every scoring
rule, and turn a niche feature into pervasive complexity. Users who
genuinely want joint modelling can write a single trainer that consumes
`(N, M)` y and slot it into an ordinary `ForecastPipeline`.

`predict()` on the multi-target wrapper returns
`{target_name: {stage_name: DistributionForecast}}`:

```python
preds = mt.predict(X_new, ids=new_ids, timestamps=new_ts)
preds["high"]["emos"].params["mu"]   # shape (n_new,)
```
