"""Composite forecasters and drivers.

- LiftedForecaster: PointForecaster + Lifter → DistForecaster (§4.8).
  Pipeline injects base_oof via the same mechanism as deps_oof.
- WalkForward: refit-each-cycle driver for batch forecasters (§4.5).
- Bootstrap: DistForecaster that takes a base in __init__ (§3, end).

Bootstrap is intentionally NOT a Lifter — it needs to refit the base B
times on resampled subsets, which the (point_oof, y) Lifter signature
can't express. Same pattern would handle Jackknife+ / CV+ later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from bracketlearn.protocols import (
    Calibrator,
    DistForecaster,
    Forecaster,
    Lifter,
    PointForecaster,
)

if TYPE_CHECKING:
    from bracketlearn.forecast import DistributionForecast, PointForecast


# ---------------------------------------------------------------------------
# LiftedForecaster — first-class composite (§4.8).
# ---------------------------------------------------------------------------


class LiftedForecaster:
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


class CalibratedForecaster:
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


# ---------------------------------------------------------------------------
# WalkForward — refit-each-cycle driver (§4.5).
# ---------------------------------------------------------------------------


@dataclass
class WalkForward:
    """Wraps a batch forecaster into a refit-each-cycle pipeline.

    Pipeline-level concept — not a Forecaster protocol member. Inspect
    .forecaster to reach the wrapped object.

    refit_every: "1d" / "1w" / N rows / a Splitter.
    expanding=True grows the training window each cycle (default);
    expanding=False uses a rolling window of fixed length.
    """

    forecaster: Forecaster
    refit_every: str | int
    expanding: bool = True
    window_length: str | int | None = None    # required if not expanding


# ---------------------------------------------------------------------------
# Bootstrap — DistForecaster that refits a base B times on resampled data.
# ---------------------------------------------------------------------------


class Bootstrap:
    """DistForecaster wrapping a PointForecaster. Refits B copies of the
    base on bootstrap-resampled subsets; predict_dist returns an
    empirical-backed DistributionForecast over the B member predictions.

    NOT a Lifter — Lifter takes a fitted PointForecast in; Bootstrap takes
    a recipe (forecaster object) and refits it. Same pattern would handle
    Jackknife+ / CV+ in v0.2.
    """

    def __init__(
        self,
        base: PointForecaster,
        n_bags: int = 50,
        *,
        random_seed: int | None = None,
        name: str | None = None,
    ):
        self.base = base
        self.n_bags = n_bags
        self.random_seed = random_seed
        self.name = name or f"Bootstrap({base.name},B={n_bags})"
        self.depends_on = base.depends_on
        self._bags: list[PointForecaster] = []

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
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
