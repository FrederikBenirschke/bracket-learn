"""Convenience builders: pre-wrap common (forecaster, lifter/calibrator) combos.

Each returns a self-contained `Pipeline` chain. Drop it straight into
`WalkForward.fit_predict` (or nest it in a `Stacker`); ``name`` labels the
leaderboard row.
"""

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
    """RidgeCV point forecaster lifted to a Normal via `GlobalResidual`.

    Picks α from `alphas` via leave-one-out CV on the inner-fit slice.
    Returns a `Pipeline([SklearnPoint(RidgeCV), GlobalResidual()])` — a
    self-contained `DistForecaster`.
    """
    from sklearn.linear_model import RidgeCV

    from bracketlearn.lift import GlobalResidual
    from bracketlearn.pipeline import Pipeline

    return Pipeline(
        [SklearnPoint(RidgeCV(alphas=np.asarray(alphas))), GlobalResidual()],
        name=name,
    )


def emos_calibrated(*, edges: np.ndarray, name: str = "emos_calibrated") -> Any:
    """EMOS calibrated with Isotonic on the given bracket ladder.

    `edges` defines the ladder used for isotonic calibration. The pipeline
    fits the isotonic on a held-out tail of each training fold and applies
    it to the test fold. Returns a `Pipeline([EMOS(), Isotonic(edges)])`.
    """
    from bracketlearn.lift import Isotonic
    from bracketlearn.pipeline import Pipeline

    return Pipeline(
        [EMOS(), Isotonic(pre_integrate_edges=np.asarray(edges, dtype=float))],
        name=name,
    )
