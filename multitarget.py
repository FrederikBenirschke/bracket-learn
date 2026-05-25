"""Multi-target wrapper: a single (N, M) y becomes M independent pipelines.

Mirrors sklearn's ``MultiOutputRegressor``: each target column is fit by a
separate clone of the inner pipeline. No cross-target sharing — if joint
modelling is desired, the user builds a single trainer that natively
consumes (N, M) y and uses it inside an ordinary ``ForecastPipeline``.

Why a wrapper rather than threading M through every trainer:
- The (N, M) → (N, M) contract would multiply every backing's shape (e.g.
  ``DistributionForecast.params['mu']`` becomes (N, M)) and break every
  scoring rule. The blast radius is huge for a feature most users won't
  touch.
- ``MultiOutputForecastPipeline`` keeps the existing single-target machinery
  unchanged and composes M times — readable, debuggable, and matches
  user expectation that "M targets = M models" unless they explicitly
  build a joint trainer.

Usage::

    mt = MultiOutputForecastPipeline(
        ForecastPipeline(steps=[("emos", EMOS())], n_folds=5),
    )
    result = mt.fit_predict(X, Y, ids=ids, timestamps=ts)  # Y shape (N, 2)
    print(result.score(Y, metrics=["crps"])["target_0"]["emos"]["crps"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from bracketlearn.base import clone
from bracketlearn.pipeline import ForecastPipeline, PipelineResult


@dataclass
class MultiOutputPipelineResult:
    """Per-target ``PipelineResult``. Indexed by target name (``target_0`` etc.,
    or user-supplied ``target_names``).

    ``score()`` returns ``{target_name: {stage: {metric: value}}}``.
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
        ladder: Any = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Score every target against its column of Y. Returns
        ``{target_name: {stage: {metric: value}}}``."""
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (N, M); got shape {Y.shape}")
        if Y.shape[1] != len(self.target_names):
            raise ValueError(
                f"Y has {Y.shape[1]} columns but pipeline was fit with "
                f"{len(self.target_names)} targets"
            )
        out: dict[str, dict[str, dict[str, float]]] = {}
        for j, name in enumerate(self.target_names):
            out[name] = self.per_target[name].score(
                Y[:, j], metrics=metrics, ladder=ladder,
            )
        return out


class MultiOutputForecastPipeline:
    """Fits ``n_targets`` independent clones of a ``ForecastPipeline``.

    The inner pipeline is cloned per target so fitted state from one target
    cannot leak into another. The user-supplied pipeline is never mutated.

    Args:
        pipeline: the prototype ``ForecastPipeline`` to clone per target.
        target_names: optional list of M names; defaults to
            ``["target_0", "target_1", ...]``.
    """

    def __init__(
        self,
        pipeline: ForecastPipeline,
        *,
        target_names: Sequence[str] | None = None,
    ):
        self.pipeline = pipeline
        self._target_names_init = (
            list(target_names) if target_names is not None else None
        )
        self._fitted: dict[str, ForecastPipeline] = {}
        self._target_names: list[str] = []

    def fit_predict(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> MultiOutputPipelineResult:
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
            p = _clone_pipeline(self.pipeline)
            result = p.fit_predict(
                X, Y[:, j], ids=ids, timestamps=timestamps,
                sample_weight=sample_weight,
            )
            per_target[name] = result
            self._fitted[name] = p
        return MultiOutputPipelineResult(
            per_target=per_target, target_names=self._target_names,
        )

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        """Predict per target on unseen X. Returns
        ``{target_name: {stage_name: DistributionForecast}}``."""
        if not self._fitted:
            raise RuntimeError(
                "predict() requires a prior fit_predict() with refit_on_full=True"
            )
        return {
            name: p.predict(X, ids=ids, timestamps=timestamps)
            for name, p in self._fitted.items()
        }


def _clone_pipeline(p: ForecastPipeline) -> ForecastPipeline:
    """Deep-clone a ForecastPipeline: each stage's forecaster gets cloned;
    constructor params are preserved."""
    new = ForecastPipeline(
        cv=p.cv, n_folds=p.n_folds, embargo=p.embargo,
        calibration_fraction=p.calibration_fraction,
        refit_on_full=p.refit_on_full,
        shuffle=p.shuffle, random_state=p.random_state,
        rolling_window=p.rolling_window,
    )
    for stage in p._stages:
        new._register(stage.name, clone(stage.forecaster))
    return new
