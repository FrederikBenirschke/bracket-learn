"""Protocol definitions (§4).

Two leaf forecaster protocols (PointForecaster, DistForecaster), two
forecast→forecast transformer protocols (Lifter, Calibrator), and one
input/target Transformer (standardizer composed as a Pipeline's leading
stage). Five named concepts.

The WalkForward driver lives in composite.py — it's a pipeline-level
wrapper, not a protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

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
    """Anything that can be used as a stage in a `Pipeline` / `Stacker`.

    `depends_on` is retained for back-compat (default empty tuple = no deps);
    under the object-graph surface the dependency IS the `Stacker` nesting, so
    upstream forecasts arrive positionally via ``upstream=[...]`` rather than
    by name.
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
    ) -> PointForecast:
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
    ) -> DistributionForecast:
        ...


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
        point_oof: PointForecast,
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        ...

    def lift(self, point: PointForecast) -> DistributionForecast:
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
        dist_oof: DistributionForecast,
        y: np.ndarray,
    ) -> Self:
        ...

    def transform(
        self,
        dist: DistributionForecast,
    ) -> DistributionForecast:
        ...


# ---------------------------------------------------------------------------
# Transformer (§4.8) — feature/target standardizer that a Pipeline runs as
# its first stage(s). Distinct from Lifter/Calibrator (which transform a
# *forecast*): a Transformer transforms the model's *inputs* (X), the
# *target* (y) at fit, and inverts the resulting *distribution* back to the
# original scale at predict. sklearn-`TransformerMixin`-compatible: a plain
# X-only transformer is the degenerate case (identity target + inverse_dist).
# ---------------------------------------------------------------------------


@runtime_checkable
class Transformer(Protocol):
    """Input/target standardizer composed as the leading stage of a Pipeline.

    The per-row map may be data-driven and keyed on ``ids`` (group) and an
    optional per-row ``center`` array (e.g. seasonal climatology), which the
    Pipeline threads through. Contract:

    - ``fit(X, y, *, ids, center=None, **kw)`` learns the per-group scale (and
      any state) from the training rows; returns self.
    - ``transform(X, *, ids, center=None)`` maps features to standardized
      space and **stamps** the per-row ``(center, scale)`` it used, so that
      ``transform_target`` / ``inverse_dist`` need no re-derivation. Called
      once per fit batch and once per predict batch (the stamp reflects the
      most recent call — mirror the Lifter/Calibrator stateful pattern).
    - ``transform_target(y)`` maps the target by the stamped ``(center, scale)``.
    - ``inverse_dist(dist)`` maps a forecast back to the original scale via
      ``DistributionForecast.affine(shift=center, scale=scale)`` using the
      stamp from the most recent ``transform``.
    """

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        center: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Self:
        ...

    def transform(
        self,
        X: Any,
        *,
        ids: np.ndarray,
        center: np.ndarray | None = None,
    ) -> np.ndarray:
        ...

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        ...

    def inverse_dist(
        self,
        dist: DistributionForecast,
    ) -> DistributionForecast:
        ...
