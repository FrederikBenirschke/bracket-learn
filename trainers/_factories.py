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


def market_ols(*, name: str = "market_ols") -> Any:
    """Plain OLS + GlobalResidual. Mirrors market_ols Q2 (target = realized).

    Q1 (target = market_implied) is the same model fit with a different
    target — out of scope for bracketlearn (the target choice is a calling
    convention, not a model class).
    """
    from sklearn.linear_model import LinearRegression

    from bracketlearn.lift import GlobalResidual
    from bracketlearn.pipeline import LiftedForecaster

    return LiftedForecaster(
        base=SklearnPoint(LinearRegression()),
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
        calibrator=Isotonic(edges=np.asarray(edges, dtype=float)),
        name=name,
    )


