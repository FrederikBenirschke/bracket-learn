"""Trainers grouped by output shape / mechanism.

Public API is re-exported here so callers keep using
``from bracketlearn.trainers import EMOS`` etc.

Layout
------

- ``bracketlearn.trainers.point`` — SklearnPoint, OnlineAggregator, RNNHourly.
- ``bracketlearn.trainers.parametric`` — EMOS, NGBoostNormal, MixtureNormals,
  StackedParametric (legacy alias ``Stacking``).
- ``bracketlearn.trainers.quantile`` — QuantileReg, QuantileForest.
- ``bracketlearn.trainers.bracket`` — CumulativeBinary, TailSpecialist, CDFBoostBracket, BracketClassifier, BracketRegressor, LinearPoolDist.
- ``bracketlearn.trainers.meta`` — DistAsFeatures.

Convenience builders (``ridge``, ``emos_calibrated``) live in
``bracketlearn.trainers._factories`` and are re-exported below.
"""

from __future__ import annotations

from bracketlearn.trainers._factories import emos_calibrated, ridge
from bracketlearn.trainers.bracket import (
    BracketClassifier,
    BracketRegressor,
    CDFBoostBracket,
    CumulativeBinary,
    LinearPoolDist,
    TailSpecialist,
)
from bracketlearn.trainers.meta import BracketStacking, DistAsFeatures
from bracketlearn.trainers.parametric import (
    EMOS,
    BayesianRidge,
    BMAStacking,
    HierarchicalNormal,
    MixtureNormals,
    NGBoostNormal,
    StackedParametric,
    Stacking,
)
from bracketlearn.trainers.point import OnlineAggregator, RNNHourly, SklearnPoint
from bracketlearn.trainers.quantile import QuantileForest, QuantileReg

__all__ = [
    # point
    "OnlineAggregator",
    "RNNHourly",
    "SklearnPoint",
    # parametric
    "BMAStacking",
    "BayesianRidge",
    "EMOS",
    "HierarchicalNormal",
    "MixtureNormals",
    "NGBoostNormal",
    "StackedParametric",
    "Stacking",  # legacy alias for StackedParametric
    # quantile
    "QuantileForest",
    "QuantileReg",
    # bracket
    "BracketClassifier",
    "BracketRegressor",
    "CDFBoostBracket",
    "CumulativeBinary",
    "LinearPoolDist",
    "TailSpecialist",
    # meta
    "BracketStacking",
    "DistAsFeatures",
    # factories
    "emos_calibrated",
    "ridge",
]
