# Cross-validation

`WalkForward(cv=...)` accepts three modes:

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
Older rows fall out, which helps when regime drift makes old data harmful.

```python
WalkForward(cv="rolling-window", rolling_window=120, n_folds=4)
```

```
fold 0:  [train:0..120]                     [test:120..145]
fold 1:  [train:25..145]                    [test:145..170]
fold 2:  [train:50..170]                    [test:170..195]
```

## `cv="kfold"`

Plain k-fold. Splits rows into `n_folds` disjoint test sets. Pass
`shuffle=True, random_state=...` for a permuted split.

**Use only when rows are exchangeable.** On time-series data it trains on
future rows and inflates OOF metrics, so keep it away from sequential data.

```python
WalkForward(cv="kfold", n_folds=5, shuffle=True, random_state=0)
```

## Enabling refit-on-full

By default (`refit_on_full=False`) `fit_predict` produces OOF predictions
only. To call `wf.predict(X_new)` on unseen rows, pass `refit_on_full=True`.
`fit_predict` then ends with a full-data refit per model and stores it.
Calling `predict()` without it raises, a loud failure that beats handing back
OOF-style predictions in disguise.

```python
wf = WalkForward(cv="expanding-window", n_folds=5, refit_on_full=True)
wf.fit_predict(model, X, y, ids=ids, timestamps=ts)
wf.predict(X_new, ids=new_ids, timestamps=new_ts)
```

## Embargo

`embargo` defaults to 0. On an autocorrelated series the row immediately after
the train boundary carries information about the last training row, so a
nonzero embargo drops a gap between the two. Set it to the horizon over which
the target is autocorrelated, in rows: for a daily series with a week of
persistence, `embargo=7`. It applies to the time-series splitters only;
`kfold` ignores it and warns if you set it.

## Holdouts nested inside a fold

Two stages hold out data of their own, inside whatever slice the fold hands
them. Both are automatic; neither needs configuring beyond its fraction.

- A `Pipeline` ending in a **Calibrator** reserves the last
  `calibration_fraction` of the fold's train rows. The transformers and the
  core forecaster are fit on the remainder, the calibrator is fit on the
  core's out-of-sample predictions for the reserved tail, and both are then
  refit on the whole train slice for prediction. Fitting the core on
  everything first would show the calibrator in-sample predictions, which are
  better than the ones it will be applied to, so the correction it learned
  would be systematically too small.

- A **`Stacker`** leaf that feeds a meta is fit twice on the train slice, on
  each half in turn, so the predictions the meta receives as `upstream=` are
  out-of-sample. Without this the meta sees each upstream at its in-sample
  best and learns to weight whichever one overfits hardest; measured on a
  linear DGP with a deep tree beside a ridge, the stack scored CRPS 0.809
  against the ridge's 0.555, worse than either input. Leaves not consumed by a
  meta skip the extra fit.
