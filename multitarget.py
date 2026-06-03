"""Multi-target wrapper: a single (N, M) y becomes M independent models.

Mirrors sklearn's ``MultiOutputRegressor``: each target column is fit by a
separate deep-copy of the model graph, run under its own `WalkForward`. No
cross-target sharing — if joint modelling is desired, build a single trainer
that natively consumes (N, M) y.

Why a wrapper rather than threading M through every trainer:

- The (N, M) → (N, M) contract would multiply every backing's shape (e.g.
  ``DistributionForecast.params['mu']`` becomes (N, M)) and break every
  scoring rule. The blast radius is huge for a feature most users won't touch.
- ``MultiOutput`` keeps the single-target machinery unchanged and composes M
  times — readable, debuggable, and matches the expectation that "M targets =
  M models" unless a joint trainer is explicitly built.

Example::

    mt = MultiOutput(Pipeline([EMOS()], name="emos"), WalkForward(n_folds=5))
    result = mt.fit_predict(X, Y, ids=ids, timestamps=ts)   # Y shape (N, 2)
    print(result.score(Y, metrics=["crps"])["target_0"]["emos"]["crps"])
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bracketlearn.compose import WalkForward
from bracketlearn.pipeline import PipelineResult


@dataclass
class MultiOutputResult:
    """Per-target ``PipelineResult``. Indexed by target name (``target_0`` etc.,
    or user-supplied ``target_names``).

    ``score()`` returns ``{target_name: {node: {metric: value}}}``.
    """

    per_target: dict[str, PipelineResult]
    target_names: list[str]

    def __getitem__(self, name: str) -> PipelineResult:
        return self.per_target[name]

    def __iter__(self):
        return iter(self.per_target)

    def items(self):
        return self.per_target.items()

    @property
    def targets(self) -> list[str]:
        return list(self.target_names)

    def score(
        self,
        Y: np.ndarray,
        *,
        metrics: Sequence[str] = ("crps", "log_score", "pit"),
        edges: np.ndarray | None = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Score every target against its column of Y. Returns
        ``{target_name: {node: {metric: value}}}``."""
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (N, M); got shape {Y.shape}")
        if Y.shape[1] != len(self.target_names):
            raise ValueError(
                f"Y has {Y.shape[1]} columns but was fit with "
                f"{len(self.target_names)} targets"
            )
        out: dict[str, dict[str, dict[str, float]]] = {}
        for j, name in enumerate(self.target_names):
            out[name] = self.per_target[name].score(
                Y[:, j], metrics=metrics, edges=edges,
            )
        return out


class MultiOutput:
    """Fits ``M`` independent deep-copies of a model graph, one per target.

    The model graph is deep-copied per target so fitted state from one target
    cannot leak into another. The user-supplied ``model`` and ``wf`` are never
    mutated.

    Args:
        model: prototype model graph (`Pipeline` / `Stacker` / list), cloned
            per target.
        wf: prototype `WalkForward`, cloned per target. Use ``refit_on_full=True``
            to enable ``predict`` on unseen rows.
        target_names: optional list of M names; defaults to
            ``["target_0", "target_1", ...]``.
    """

    def __init__(
        self,
        model: Any,
        wf: WalkForward,
        *,
        target_names: Sequence[str] | None = None,
    ):
        self.model = model
        self.wf = wf
        self._target_names_init = (
            list(target_names) if target_names is not None else None
        )
        self._fitted: dict[str, WalkForward] = {}
        self._target_names: list[str] = []

    def _clone_wf(self) -> WalkForward:
        return WalkForward(
            cv=self.wf.cv, n_folds=self.wf.n_folds, embargo=self.wf.embargo,
            refit_on_full=self.wf.refit_on_full, shuffle=self.wf.shuffle,
            random_state=self.wf.random_state, rolling_window=self.wf.rolling_window,
        )

    def fit_predict(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> MultiOutputResult:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (N, M); got shape {Y.shape}")
        M = Y.shape[1]
        if self._target_names_init is not None:
            if len(self._target_names_init) != M:
                raise ValueError(
                    f"target_names has {len(self._target_names_init)} entries "
                    f"but Y has {M} columns"
                )
            self._target_names = list(self._target_names_init)
        else:
            self._target_names = [f"target_{j}" for j in range(M)]

        per_target: dict[str, PipelineResult] = {}
        self._fitted = {}
        for j, name in enumerate(self._target_names):
            model = copy.deepcopy(self.model)
            wf = self._clone_wf()
            result = wf.fit_predict(
                model, X, Y[:, j], ids=ids, timestamps=timestamps,
                sample_weight=sample_weight, groups=groups,
            )
            per_target[name] = result
            self._fitted[name] = wf
        return MultiOutputResult(
            per_target=per_target, target_names=self._target_names,
        )

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Predict per target on unseen X. Returns
        ``{target_name: {node_name: DistributionForecast}}``. Requires the
        prototype ``wf`` to have ``refit_on_full=True``."""
        if not self._fitted:
            raise RuntimeError(
                "predict() requires a prior fit_predict() with refit_on_full=True"
            )
        return {
            name: wf.predict(X, ids=ids, timestamps=timestamps, groups=groups)
            for name, wf in self._fitted.items()
        }
