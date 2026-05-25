# Hyperparameter search

`GridSearch` enumerates a parameter grid against the pipeline's own
time-aware CV.

```python
from bracketlearn.search import GridSearch

gs = GridSearch(
    pipeline,
    param_grid={
        "emos__sigma_floor": [0.3, 0.5, 1.0],
        "n_folds": [3, 5],
    },
    scoring="crps", refit_stage="emos",
)
gs.fit(X, y, ids=ids, timestamps=ts)

print(gs.best_params_)        # e.g. {"emos__sigma_floor": 0.5, "n_folds": 5}
print(gs.best_score_)         # mean CRPS at that combo
print(gs.best_pipeline_)      # fitted pipeline ready for .predict()
```

## Why not sklearn.GridSearchCV?

`sklearn.model_selection.GridSearchCV` re-splits the data with its own
`KFold`. That destroys time ordering and silently inflates OOF metrics
on sequential data. `GridSearch` runs the pipeline's own
`expanding-window` / `rolling-window` CV inside each grid point so OOF
estimates remain honest.

## Param-grid syntax

- Pipeline-level params (`cv`, `n_folds`, `embargo`, `rolling_window`,
  `shuffle`, `random_state`, `refit_on_full`, `calibration_fraction`)
  appear without a prefix.
- Stage-level params use sklearn-style `stage_name__field`:
  `"emos__sigma_floor"`, `"ridge__base__estimator"`, etc.

## Scoring

Built-in metrics (all losses — lower is better):

- `"crps"` — continuous ranked probability score.
- `"log_score"` — predictive negative log-likelihood.
- `"log_loss_bracket"` — bracket-contract log loss (requires `ladder=`).
- `"brier_bracket"` — bracket-contract Brier (requires `ladder=`).

The objective is `result.score(...)[refit_stage][scoring]`. Pass
`refit_stage=None` to average across all stages — useful if you have a
single combined stage and want grid points scored on that average.
