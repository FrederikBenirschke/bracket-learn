"""GridSearch over a model graph + WalkForward hyperparameters.

We do *not* reuse ``sklearn.model_selection.GridSearchCV`` because our CV is
time-aware (``expanding-window`` / ``rolling-window`` / ``kfold``, owned by
`WalkForward`). Sklearn's GridSearchCV would re-split with its own KFold,
destroying time ordering and silently inflating OOF metrics on sequential
data.

The search takes a **model graph** (a `Pipeline` / `Stacker` / list of them)
and a **WalkForward** template, kept separate the way the native surface keeps
model and CV separate. For each combination from ``param_grid`` it:

1. deep-copies the model graph and applies any ``node__field`` params to the
   graph node named ``node`` (sklearn ``__``-nested syntax — e.g.
   ``qreg__n_estimators=400`` routes into the stage owning ``n_estimators``);
2. clones the WalkForward template, overriding any CV-level params
   (``n_folds``, ``cv``, ``embargo``, ``rolling_window``, ``refit_on_full``,
   ``shuffle``, ``random_state``);
3. runs ``WalkForward.fit_predict`` and scores the chosen node with the chosen
   metric.

It returns the best params, the fitted winning ``WalkForward`` (ready for
``.predict`` when ``refit_on_full=True``), and a full results table.

Usage::

    model = Pipeline([QuantileReg()], name="qreg")
    wf = WalkForward(cv="kfold", n_folds=4, refit_on_full=True)
    grid = {"qreg__n_estimators": [50, 150, 400], "n_folds": [3, 5]}
    search = GridSearch(model, wf, param_grid=grid,
                        scoring="crps", refit_node="qreg")
    search.fit(X, y, ids=ids, timestamps=ts)
    print(search.best_params_)      # {"qreg__n_estimators": 400, "n_folds": 5}
    print(search.best_score_)       # mean CRPS at that combo
    preds = search.best_wf_.predict(X_new, ids=..., timestamps=...)
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np

from bracketlearn.compose import WalkForward, _flatten
from bracketlearn.pipeline import Pipeline

# CV-level params route to the WalkForward clone; everything else must be a
# ``node__field`` nested key.
_WF_KEYS = {
    "cv", "n_folds", "embargo", "refit_on_full", "shuffle", "random_state",
    "rolling_window",
}


class GridSearch:
    """Brute-force grid search over a model graph + `WalkForward` params.

    Args:
        model: prototype model graph (`Pipeline` / `Stacker` / list). Deep-copied
            per grid point; never mutated.
        wf: prototype `WalkForward`. Cloned per grid point; never mutated.
        param_grid: dict mapping param name to a list of candidate values. Keys
            are either a CV-level WalkForward arg (``n_folds`` etc.) or a
            ``node__field`` nested key routed into the graph node named ``node``.
        scoring: metric name passed to ``PipelineResult.score`` — one of
            ``crps``, ``log_score``, ``log_loss_bracket``, ``brier_bracket``.
            Lower is better for all four (this is a *loss*).
        refit_node: name of the node whose OOF metric is the objective. If
            ``None``, the *mean* across all nodes is used.
        edges: shared 1-D bracket ladder ``(B+1,)``; required if ``scoring``
            is a bracket metric.
        greater_is_better: defaults to ``False`` (built-in metrics are losses).
    """

    def __init__(
        self,
        model: Any,
        wf: WalkForward,
        *,
        param_grid: dict[str, Sequence[Any]],
        scoring: str = "crps",
        refit_node: str | None = None,
        edges: Any = None,
        greater_is_better: bool = False,
    ):
        if not param_grid:
            raise ValueError("param_grid is empty")
        _ALLOWED = {"crps", "log_score", "log_loss_bracket", "brier_bracket"}
        if scoring not in _ALLOWED:
            raise ValueError(f"scoring={scoring!r} not in {_ALLOWED}")
        if scoring in ("log_loss_bracket", "brier_bracket") and edges is None:
            raise ValueError(f"scoring={scoring!r} requires edges=...")
        self.model = model
        self.wf = wf
        self.param_grid = {k: list(v) for k, v in param_grid.items()}
        self.scoring = scoring
        self.refit_node = refit_node
        self.edges = edges
        self.greater_is_better = greater_is_better
        self.results_: list[dict[str, Any]] = []
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_model_: Any = None
        self.best_wf_: WalkForward | None = None

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
        groups: np.ndarray | None = None,
    ) -> GridSearch:
        self.results_ = []
        best_score = -np.inf if self.greater_is_better else np.inf
        best_params: dict[str, Any] | None = None
        best_model: Any = None
        best_wf: WalkForward | None = None

        for params in self._iter_grid():
            model, wf = _clone_with_params(self.model, self.wf, params)
            result = wf.fit_predict(
                model, X, y, ids=ids, timestamps=timestamps,
                sample_weight=sample_weight, groups=groups,
            )
            scores = result.score(y, metrics=[self.scoring], edges=self.edges)
            if self.refit_node is not None:
                if self.refit_node not in scores:
                    raise ValueError(
                        f"refit_node={self.refit_node!r} not in graph nodes "
                        f"{list(scores)}"
                    )
                metric_val = float(scores[self.refit_node][self.scoring])
            else:
                vals = [
                    s[self.scoring] for s in scores.values()
                    if self.scoring in s and not np.isnan(s[self.scoring])
                ]
                if not vals:
                    raise RuntimeError(
                        f"no node produced a finite {self.scoring} score; "
                        "set refit_node=... explicitly"
                    )
                metric_val = float(np.mean(vals))
            self.results_.append({"params": dict(params), self.scoring: metric_val})
            improved = (
                metric_val > best_score if self.greater_is_better
                else metric_val < best_score
            )
            if improved:
                best_score = metric_val
                best_params = dict(params)
                best_model = model
                best_wf = wf

        self.best_params_ = best_params
        self.best_score_ = float(best_score) if best_params is not None else None
        self.best_model_ = best_model
        self.best_wf_ = best_wf
        return self


def _clone_with_params(
    model: Any, wf: WalkForward, params: dict[str, Any],
) -> tuple[Any, WalkForward]:
    """Deep-copy ``model`` + clone ``wf``, then route each param to its level.

    ``node__field`` keys set ``field`` on the graph node named ``node`` (for a
    `Pipeline` node, on the single stage that owns ``field``); bare keys in
    ``_WF_KEYS`` override the WalkForward clone. Anything else raises.
    """
    wf_overrides: dict[str, Any] = {}
    node_params: dict[str, dict[str, Any]] = {}
    for k, v in params.items():
        if "__" in k:
            head, _, tail = k.partition("__")
            node_params.setdefault(head, {})[tail] = v
        elif k in _WF_KEYS:
            wf_overrides[k] = v
        else:
            raise ValueError(
                f"param {k!r} is neither a WalkForward arg ({sorted(_WF_KEYS)}) "
                f"nor a node-nested key (use 'node_name__field' form)"
            )

    new_model = copy.deepcopy(model)
    if node_params:
        nodes_by_name = {n["name"]: n["obj"] for n in _flatten(new_model)}
        for node_name, fields in node_params.items():
            if node_name not in nodes_by_name:
                raise ValueError(
                    f"param targets node {node_name!r}, not in graph nodes "
                    f"{sorted(nodes_by_name)}"
                )
            for field, value in fields.items():
                _set_node_field(nodes_by_name[node_name], node_name, field, value)

    wf_kwargs = dict(
        cv=wf.cv, n_folds=wf.n_folds, embargo=wf.embargo,
        refit_on_full=wf.refit_on_full, shuffle=wf.shuffle,
        random_state=wf.random_state, rolling_window=wf.rolling_window,
    )
    wf_kwargs.update(wf_overrides)
    return new_model, WalkForward(**wf_kwargs)


def _set_node_field(obj: Any, node_name: str, field: str, value: Any) -> None:
    """Set ``field=value`` on a graph node's underlying estimator.

    For a `Pipeline` node the field is routed to the single stage that exposes
    it (ambiguity or absence raises loud, per Rule #0.5). For a bare forecaster
    or a `Stacker` meta the field is set directly.
    """
    if isinstance(obj, Pipeline):
        owners = [s for s in obj.stages if field in s.get_params(deep=False)]
        if not owners:
            raise ValueError(
                f"node {node_name!r}: no stage exposes param {field!r} "
                f"(stages: {[type(s).__name__ for s in obj.stages]})"
            )
        if len(owners) > 1:
            raise ValueError(
                f"node {node_name!r}: param {field!r} is exposed by "
                f"{len(owners)} stages — ambiguous"
            )
        owners[0].set_params(**{field: value})
    else:
        obj.set_params(**{field: value})
