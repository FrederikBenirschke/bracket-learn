"""Protocol definitions (§4).

Two leaf forecaster protocols (PointForecaster, DistForecaster) + a
StepLearner mixin for genuine online learners + two transformer protocols
(Lifter, Calibrator). Five named concepts total.

The WalkForward driver lives in composite.py — it's a pipeline-level
wrapper, not a protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from bracketlearn.forecast import (
        DistributionForecast,
        PointForecast,
    )


# ---------------------------------------------------------------------------
# Parent Forecaster (§4.1).
# ---------------------------------------------------------------------------


@runtime_checkable
class Forecaster(Protocol):
    """Anything that can be added to a ForecastPipeline.

    `depends_on` names upstream pipeline nodes whose OOF this forecaster
    consumes. Pipeline topo-sorts on this at build time and raises loudly
    on missing deps (Rule #0.5). Default empty tuple = no deps.
    """

    name: str
    depends_on: tuple[str, ...]


# ---------------------------------------------------------------------------
# PointForecaster (§4.2).
# ---------------------------------------------------------------------------


@runtime_checkable
class PointForecaster(Forecaster, Protocol):
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        ...

    def predict(
        self,
        X: Any,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> "PointForecast":
        ...


# ---------------------------------------------------------------------------
# DistForecaster (§4.3).
# ---------------------------------------------------------------------------


@runtime_checkable
class DistForecaster(Forecaster, Protocol):
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        ...

    def predict_dist(
        self,
        X: Any,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> "DistributionForecast":
        ...


# ---------------------------------------------------------------------------
# StepLearner mixin (§4.4) — for genuine online forecasters.
# ---------------------------------------------------------------------------


class StepLearner:
    """Mixin for online forecasters whose internal state updates per
    observation. Compose with PointForecaster or DistForecaster.

    `step` emits a forecast for X_t. `observe` ingests labels that may
    arrive later (e.g. weather settlement lagging predictions by hours).
    `observe` must be idempotent — calling it twice with the same labels
    is a no-op.

    forecast.provenance.fold_idx is set to "prequential" by convention.
    """

    def step(
        self,
        X_t: Any,
        *,
        ids_t: np.ndarray,
        timestamp_t: Any,
    ) -> Any:
        """Emit forecast for the current observation."""
        raise NotImplementedError

    def observe(self, labels: dict[Any, float]) -> None:
        """Ingest realized y for past predictions. Idempotent.

        keys = ids or (id, timestamp) tuples; values = realized y.
        """
        raise NotImplementedError

    def pending_ids(self) -> set[Any]:
        """ids of predictions made but not yet observed."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lifter (§4.6) — Point → Dist transformer.
# ---------------------------------------------------------------------------


@runtime_checkable
class Lifter(Protocol):
    """Point → Distribution lift.

    requires_X: if True, fit needs raw X (e.g. ConditionalVariance fits
    σ̂ = f(X) on log r²). Pipeline raises loudly if requires_X and X is
    not supplied.

    Pipeline supplies point_oof from its fold structure; standalone
    callers compute it themselves (sklearn.cross_val_predict).
    """

    requires_X: bool

    def fit(
        self,
        point_oof: "PointForecast",
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        ...

    def lift(self, point: "PointForecast") -> "DistributionForecast":
        ...


# ---------------------------------------------------------------------------
# Calibrator (§4.7) — Dist → Dist transformer.
# ---------------------------------------------------------------------------


@runtime_checkable
class Calibrator(Protocol):
    """Reliability-targeted Dist → Dist correction.

    Examples: Isotonic, Platt, Beta, VennAbers, BinwiseCDF,
    ConformalCalibrate (per-τ coverage).

    Pipeline supplies dist_oof from a held-out calibration fold.
    """

    def fit(
        self,
        dist_oof: "DistributionForecast",
        y: np.ndarray,
    ) -> Self:
        ...

    def transform(
        self,
        dist: "DistributionForecast",
    ) -> "DistributionForecast":
        ...
