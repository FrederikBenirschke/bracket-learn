# Quickstart

```python
import numpy as np
from sklearn.linear_model import RidgeCV

from bracketlearn.adapters import BracketLadder
from bracketlearn.composite import CalibratedForecaster, LiftedForecaster
from bracketlearn.lift import GlobalResidual, Isotonic
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.trainers import EMOS, QuantileReg, SklearnPoint

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
print(new_dists["qreg"].params)
```

## What just happened

1. `LiftedForecaster(SklearnPoint(RidgeCV()), GlobalResidual())` wraps a
   point regressor as a parametric-normal forecast: ridge predicts μ̂, the
   global-residual lifter estimates one σ from OOF residuals.
2. `CalibratedForecaster(EMOS(), Isotonic(edges=edges))` fits EMOS on the
   ensemble columns, then per-fold runs isotonic calibration on a held-out
   tail to correct bracket-probability bias.
3. `QuantileReg` fits one LightGBM per τ; the framework stores the result
   as a quantile-backed distribution.
4. The pipeline runs **expanding-window CV** under the hood: each forecaster
   is cloned per fold, fit on the train slice, predicted on the test slice,
   and OOF predictions are stitched into one `DistributionForecast` per stage.
5. `result.score()` aligns y to each stage's OOF coverage via `dist.ids` —
   you never touch row indices manually.
6. `pipeline.predict(X_new)` uses canonical full-train refits stored at the
   end of `fit_predict`.
