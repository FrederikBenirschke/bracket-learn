# Hyperparameter search

`GridSearch` enumerates a parameter grid, cloning the model **and** the
`WalkForward` driver per grid point so every combo is scored under the same
time-aware CV.

```python
from bracketlearn import Pipeline, WalkForward
from bracketlearn.search import GridSearch
from bracketlearn.trainers import EMOS

gs = GridSearch(
    Pipeline([EMOS()], name="emos"),
    WalkForward(cv="expanding-window", n_folds=5, refit_on_full=True),
    param_grid={
        "emos__sigma_floor": [0.3, 0.5, 1.0],
        "n_folds": [3, 5],
    },
    scoring="crps", refit_node="emos",
)
gs.fit(X, y, ids=ids, timestamps=ts)

print(gs.best_params_)        # e.g. {"emos__sigma_floor": 0.5, "n_folds": 5}
print(gs.best_score_)         # mean CRPS at that combo
gs.best_wf_.predict(X_new, ids=new_ids, timestamps=new_ts)  # refit driver
```

## Why not sklearn.GridSearchCV?

`sklearn.model_selection.GridSearchCV` re-splits the data with its own
`KFold`. That destroys time ordering and silently inflates OOF metrics
on sequential data. `GridSearch` runs `WalkForward`'s own
`expanding-window` / `rolling-window` CV inside each grid point so OOF
estimates remain honest.

## Param-grid syntax

- `WalkForward`-level params (`cv`, `n_folds`, `embargo`, `rolling_window`,
  `shuffle`, `random_state`, `refit_on_full`) appear without a prefix —
  they are applied to the cloned driver.
- Model stage params use sklearn-style `node__field`:
  `"emos__sigma_floor"`, `"ridge__base__estimator"`, etc.

## Scoring

Built-in metrics (all losses — lower is better):

- `"crps"` — continuous ranked probability score.
- `"log_score"` — predictive negative log-likelihood.
- `"log_loss_bracket"` — bracket-contract log loss (requires `edges=`).
- `"brier_bracket"` — bracket-contract Brier (requires `edges=`).

The objective is `result.score(...)[refit_node][scoring]`. Pass
`refit_node=None` to average across all nodes — useful if you have a
single combined node and want grid points scored on that average.
