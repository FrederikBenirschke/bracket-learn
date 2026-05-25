"""GridSearch over ForecastPipeline hyperparameters.

We do *not* reuse ``sklearn.model_selection.GridSearchCV`` because our
pipeline owns its own time-aware CV (``expanding-window`` / ``rolling-window``
/ ``kfold``). Sklearn's GridSearchCV would re-split the data with its own
KFold, which destroys time ordering and silently inflates OOF metrics on
sequential data.

Instead we expose a small loop that:

1. Iterates over every combination from ``param_grid``.
2. For each combination, clones the prototype pipeline, applies the params
   via ``set_params`` (sklearn ``__``-nested syntax — e.g.
   ``emos__sigma_floor=0.5``), runs ``fit_predict``, and scores the chosen
   stage with the chosen metric.
3. Returns the best params and a full results table.

Usage::

    grid = {
        "emos__sigma_floor": [0.3, 0.5, 1.0],
        "n_folds": [3, 5],
    }
    search = GridSearch(prototype, param_grid=grid, scoring="crps",
                       refit_stage="emos")
    search.fit(X, y, ids=ids, timestamps=ts)
    print(search.best_params_)        # {"emos__sigma_floor": 0.5, "n_folds": 5}
    print(search.best_score_)         # mean CRPS at that combo
    print(search.results_)            # list of (params, score) rows
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np

from bracketlearn.base import clone
from bracketlearn.pipeline import ForecastPipeline


class GridSearch:
    """Brute-force grid search over a ``ForecastPipeline``'s params.

    Args:
        pipeline: prototype pipeline. Cloned per grid point; never mutated.
        param_grid: dict mapping param name (or ``stage__param`` nested name)
            to a list of candidate values.
        scoring: metric name passed to ``PipelineResult.score``. Must be one
            of: ``crps``, ``log_score``, ``log_loss_bracket``, ``brier_bracket``.
            Lower is better for all four — this is a *loss*, not a score.
        refit_stage: name of the stage whose OOF metric is the objective.
            If ``None``, the *mean* across all stages is used (rarely useful;
            usually a single 'final' stage is the comparison target).
        ladder: required if ``scoring`` is a bracket metric.
        greater_is_better: defaults to ``False`` since all built-in metrics
            are losses. Flip if you wire in a custom higher-is-better metric.
    """

    def __init__(
        self,
        pipeline: ForecastPipeline,
        *,
        param_grid: dict[str, Sequence[Any]],
        scoring: str = "crps",
        refit_stage: str | None = None,
        ladder: Any = None,
        greater_is_better: bool = False,
    ):
        if not param_grid:
            raise ValueError("param_grid is empty")
        _ALLOWED = {"crps", "log_score", "log_loss_bracket", "brier_bracket"}
        if scoring not in _ALLOWED:
            raise ValueError(f"scoring={scoring!r} not in {_ALLOWED}")
        if scoring in ("log_loss_bracket", "brier_bracket") and ladder is None:
            raise ValueError(f"scoring={scoring!r} requires ladder=...")
        self.pipeline = pipeline
        self.param_grid = {k: list(v) for k, v in param_grid.items()}
        self.scoring = scoring
        self.refit_stage = refit_stage
        self.ladder = ladder
        self.greater_is_better = greater_is_better
        self.results_: list[dict[str, Any]] = []
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_pipeline_: ForecastPipeline | None = None

    def _iter_grid(self) -> list[dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> GridSearch:
        self.results_ = []
        best_score = -np.inf if self.greater_is_better else np.inf
        best_params: dict[str, Any] | None = None
        best_pipeline: ForecastPipeline | None = None

        for params in self._iter_grid():
            p = _clone_pipeline_with_params(self.pipeline, params)
            result = p.fit_predict(
                X, y, ids=ids, timestamps=timestamps, sample_weight=sample_weight,
            )
            scores = result.score(
                y, metrics=[self.scoring], ladder=self.ladder,
            )
            if self.refit_stage is not None:
                if self.refit_stage not in scores:
                    raise ValueError(
                        f"refit_stage={self.refit_stage!r} not in pipeline stages "
                        f"{list(scores)}"
                    )
                metric_val = float(scores[self.refit_stage][self.scoring])
            else:
                vals = [
                    s[self.scoring] for s in scores.values()
                    if self.scoring in s and not np.isnan(s[self.scoring])
                ]
                if not vals:
                    raise RuntimeError(
                        f"no stage produced a finite {self.scoring} score; "
                        "set refit_stage=... explicitly"
                    )
                metric_val = float(np.mean(vals))
            self.results_.append({
                "params": dict(params),
                self.scoring: metric_val,
            })
            improved = (
                metric_val > best_score if self.greater_is_better
                else metric_val < best_score
            )
            if improved:
                best_score = metric_val
                best_params = dict(params)
                best_pipeline = p

        self.best_params_ = best_params
        self.best_score_ = float(best_score) if best_params is not None else None
        self.best_pipeline_ = best_pipeline
        return self


def _clone_pipeline_with_params(
    prototype: ForecastPipeline, params: dict[str, Any],
) -> ForecastPipeline:
    """Clone the prototype, then route each param to the right level:
    keys without ``__`` (or starting with a recognised pipeline ctor arg)
    set pipeline-level params; keys like ``stage__field`` route into the
    stage's forecaster.

    Pipeline-level params recognised: ``cv``, ``n_folds``, ``embargo``,
    ``calibration_fraction``, ``refit_on_full``, ``shuffle``,
    ``random_state``, ``rolling_window``.
    """
    pipeline_keys = {
        "cv", "n_folds", "embargo", "calibration_fraction", "refit_on_full",
        "shuffle", "random_state", "rolling_window",
    }
    pipeline_params: dict[str, Any] = {}
    stage_params: dict[str, dict[str, Any]] = {}
    for k, v in params.items():
        if "__" in k:
            head, _, tail = k.partition("__")
            stage_params.setdefault(head, {})[tail] = v
        elif k in pipeline_keys:
            pipeline_params[k] = v
        else:
            raise ValueError(
                f"param {k!r} is neither a pipeline ctor arg nor a "
                f"stage-nested key (use 'stage_name__field' form)"
            )

    base_kwargs = dict(
        cv=prototype.cv, n_folds=prototype.n_folds, embargo=prototype.embargo,
        calibration_fraction=prototype.calibration_fraction,
        refit_on_full=prototype.refit_on_full,
        shuffle=prototype.shuffle, random_state=prototype.random_state,
        rolling_window=prototype.rolling_window,
    )
    base_kwargs.update(pipeline_params)
    new = ForecastPipeline(**base_kwargs)
    for stage in prototype._stages:
        cloned = clone(stage.forecaster)
        if stage.name in stage_params:
            cloned.set_params(**stage_params[stage.name])
        new._register(stage.name, cloned)
    return new
