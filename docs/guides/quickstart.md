# Quickstart

```python
import numpy as np
from sklearn.linear_model import RidgeCV

from bracketlearn import Pipeline, WalkForward
from bracketlearn.lift import GlobalResidual, Isotonic
from bracketlearn.trainers import EMOS, QuantileReg, SklearnPoint

# --- synthetic data: 500 hourly observations of a noisy linear signal --------
rng = np.random.default_rng(0)
N = 500
X = rng.normal(size=(N, 3))
y = 50.0 + X @ np.array([3.0, -1.5, 2.0]) + rng.normal(scale=4.0, size=N)
ids = np.arange(N)
ts = np.datetime64("2024-01-01T00", "h") + np.arange(N)   # hourly

# Hold the last 50 rows out for the .predict() demo at the end.
X, X_new = X[:-50], X[-50:]
y = y[:-50]
ids, new_ids = ids[:-50], ids[-50:]
ts, new_ts = ts[:-50], ts[-50:]
# -----------------------------------------------------------------------------

edges = np.linspace(0, 100, 11)   # 10 brackets

# Each model is a Pipeline (a sequential chain of stages). Names are
# leaderboard labels, not wiring.
ridge = Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()], name="ridge")
emos = Pipeline([EMOS(), Isotonic(pre_integrate_edges=edges)], name="emos")
qreg = Pipeline([QuantileReg(n_estimators=100)], name="qreg")

# WalkForward is the CV/OOF driver. Pass one model or a list of them.
wf = WalkForward(cv="expanding-window", n_folds=5, refit_on_full=True)
result = wf.fit_predict([ridge, emos, qreg], X, y, ids=ids, timestamps=ts)

# Distribution-level metrics on OOF predictions.
print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

# Bracket-contract metrics: pass the shared edge vector; the result builds
# the per-row bracket ladder internally per stage.
print(result.to_table(y, metrics=["log_loss_bracket", "brier_bracket"],
                       edges=edges))

# Predict on unseen data using each stage's full-train refit.
new_dists = wf.predict(X_new, ids=new_ids, timestamps=new_ts)
print(new_dists["qreg"].params)
```

## What just happened

1. `Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()])` chains a point
   regressor and a lifter into one parametric-normal forecaster: ridge
   predicts μ̂, the global-residual lifter estimates one σ from OOF residuals.
2. `Pipeline([EMOS(), Isotonic(pre_integrate_edges=edges)])` fits EMOS on the
   ensemble columns, then per-fold runs isotonic calibration on the bracket
   probabilities (`pre_integrate_edges` tells `Isotonic` to project the
   Normal onto the ladder before calibrating).
3. `QuantileReg` fits one LightGBM per τ; a single-stage `Pipeline` stores the
   result as a quantile-backed distribution.
4. `WalkForward` runs **expanding-window CV** under the hood: each model is
   cloned per fold, fit on the train slice, predicted on the test slice, and
   OOF predictions are stitched into one `DistributionForecast` per model.
5. `result.score()` and `result.to_table()` align y to each model's OOF
   coverage via `dist.ids`, so you never touch row indices by hand.
6. `wf.predict(X_new)` uses canonical full-train refits stored at the end of
   `fit_predict` (enabled by `refit_on_full=True`).
