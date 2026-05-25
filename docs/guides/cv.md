# Cross-validation

`ForecastPipeline(cv=...)` accepts three modes:

## `cv="expanding-window"` (default)

Train window grows by one chunk per fold; test fold sits immediately after.
Use for sequential / time-series data. `embargo=k` skips `k` rows between
train and test to handle look-ahead leakage when rows are autocorrelated.

```
fold 0:  [train:0..40]                      [test:40..80]
fold 1:  [train:0..80]                      [test:80..120]
fold 2:  [train:0..120]                     [test:120..160]
```

## `cv="rolling-window"`

Fixed-width train window slides forward. Requires `rolling_window=<int>`.
Older rows fall out — use when regime drift makes old data harmful.

```python
ForecastPipeline(steps=[...], cv="rolling-window",
                 rolling_window=120, n_folds=4)
```

```
fold 0:  [train:0..120]                     [test:120..145]
fold 1:  [train:25..145]                    [test:145..170]
fold 2:  [train:50..170]                    [test:170..195]
```

## `cv="kfold"`

Plain k-fold. Splits rows into `n_folds` disjoint test sets. Pass
`shuffle=True, random_state=...` for a permuted split.

**Use only when rows are exchangeable** — never for time-series data,
where it would silently train on future rows and inflate OOF metrics.

```python
ForecastPipeline(steps=[...], cv="kfold", n_folds=5,
                 shuffle=True, random_state=0)
```

## Disabling refit-on-full

By default, `fit_predict` ends with a full-data refit per stage so
`pipeline.predict(X_new)` works on unseen rows. If you only want OOF
metrics and want to skip the refit, pass `refit_on_full=False`. Calling
`predict()` afterward then raises (loud failure rather than silently
producing OOF-style predictions).
