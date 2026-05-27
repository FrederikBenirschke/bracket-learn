"""Convenience builders: pre-wrap common (forecaster, lifter/calibrator) combos."""

from __future__ import annotations

from typing import Any

import numpy as np

from bracketlearn.trainers.parametric import EMOS
from bracketlearn.trainers.point import SklearnPoint


def ridge(
    *,
    alphas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    name: str = "ridge",
) -> Any:
    """RidgeCV + GlobalResidual wrapped in LiftedForecaster.

    Picks α from `alphas` via leave-one-out CV on the inner-fit slice.
    Returns a LiftedForecaster ready to register with a ForecastPipeline.
    """
    from sklearn.linear_model import RidgeCV

    from bracketlearn.lift import GlobalResidual
    from bracketlearn.pipeline import LiftedForecaster

    return LiftedForecaster(
        base=SklearnPoint(RidgeCV(alphas=np.asarray(alphas))),
        lifter=GlobalResidual(),
        name=name,
    )


def emos_calibrated(*, edges: np.ndarray, name: str = "emos_calibrated") -> Any:
    """EMOS wrapped with Isotonic on the given bracket ladder.

    `edges` defines the ladder used for isotonic calibration. The pipeline
    fits the isotonic on a held-out tail of each training fold and applies
    it to the test fold.
    """
    from bracketlearn.lift import Isotonic
    from bracketlearn.pipeline import CalibratedForecaster

    return CalibratedForecaster(
        forecaster=EMOS(),
        calibrator=Isotonic(pre_integrate_edges=np.asarray(edges, dtype=float)),
        name=name,
    )


