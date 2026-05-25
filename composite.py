"""Composite forecasters: combine simple Forecasters into richer ones.

- LiftedForecaster:     PointForecaster + Lifter      → DistForecaster.
- CalibratedForecaster: DistForecaster  + Calibrator  → DistForecaster.

Both are flat wrappers — the pipeline keeps a `[(name, forecaster)]` list
without special slots for lifters/calibrators.

Planned for v0.2 (see README):
- Bootstrap (B-fold resample → empirical-backed DistForecast)
- WalkForward refit-each-cycle driver
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.protocols import Calibrator, Lifter, PointForecaster

if TYPE_CHECKING:
    from bracketlearn.forecast import DistributionForecast, PointForecast


# ---------------------------------------------------------------------------
# LiftedForecaster — first-class composite (§4.8).
# ---------------------------------------------------------------------------


class LiftedForecaster(BaseEstimator):
    """PointForecaster + Lifter, exposed as a DistForecaster.

    fit signature: fit(X, y, *, base_oof: PointForecast).
    Pipeline supplies base_oof from its fold structure. Standalone callers
    compute OOF themselves (cross_val_predict → PointForecast → .fit).

    No hidden inner CV. No secret pipeline-state coupling. Rule #0.5.
    """

    def __init__(
        self,
        base: PointForecaster,
        lifter: Lifter,
        *,
        name: str | None = None,
    ):
        self.base = base
        self.lifter = lifter
        self.name = name or f"{base.name}+{type(lifter).__name__}"
        self.depends_on = base.depends_on

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        base_oof: "PointForecast",
        deps_oof: dict[str, Any] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        self.base.fit(
            X, y,
            sample_weight=sample_weight,
            deps_oof=deps_oof,
        )
        if self.lifter.requires_X:
            self.lifter.fit(base_oof, y, X=X)
        else:
            self.lifter.fit(base_oof, y)
        return self

    def predict_dist(
        self,
        X: Any,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> "DistributionForecast":
        point = self.base.predict(X, ids=ids, timestamps=timestamps)
        return self.lifter.lift(point)


# ---------------------------------------------------------------------------
# CalibratedForecaster — DistForecaster + Calibrator → DistForecaster.
# ---------------------------------------------------------------------------


class CalibratedForecaster(BaseEstimator):
    """Wraps a DistForecaster with a Calibrator. Pipeline fits the calibrator
    on a held-out tail of each training fold (see ForecastPipeline).

    Mirrors LiftedForecaster: the wrapped trainer stays a plain DistForecaster
    so the pipeline keeps a flat list of (name, forecaster) pairs.
    """

    def __init__(
        self,
        forecaster: Any,
        calibrator: Calibrator,
        *,
        name: str | None = None,
    ):
        self.forecaster = forecaster
        self.calibrator = calibrator
        self.name = name or f"{getattr(forecaster, 'name', type(forecaster).__name__)}+{type(calibrator).__name__}"
        self.depends_on = tuple(getattr(forecaster, "depends_on", ()))

    def fit(self, X: Any, y: np.ndarray, **kwargs: Any) -> Self:
        self.forecaster.fit(X, y, **kwargs)
        return self

    def predict_dist(self, X: Any, **kwargs: Any) -> "DistributionForecast":
        dist = self.forecaster.predict_dist(X, **kwargs)
        if getattr(self.calibrator, "fitted_", True):
            return self.calibrator.transform(dist)
        return dist


