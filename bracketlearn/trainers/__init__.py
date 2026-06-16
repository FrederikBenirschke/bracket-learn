"""Trainers grouped by output shape / mechanism.

Public API is re-exported here so callers keep using
``from bracketlearn.trainers import EMOS`` etc.

Layout (by what a trainer models)
---------------------------------

- ``bracketlearn.trainers.point`` — SklearnPoint, OnlineAggregator, RNNHourly.
- ``bracketlearn.trainers.parametric`` — EMOS, HeteroscedasticNormal,
  NGBoostNormal, MixtureNormals, BayesianRidge, HierarchicalNormal.
- ``bracketlearn.trainers.quantile`` — QuantileReg, QuantileForest.
- ``bracketlearn.trainers.bracket`` — CumulativeBinary (bracket-native).
- ``bracketlearn.trainers.combiners`` — trainers that combine *upstream*
  forecasts: StackedParametric, BMAStacking, DistAsFeatures, BracketStacking,
  LinearPoolDist, TailSpecialist, CDFBoostBracket.

Convenience builders (``ridge``, ``emos_calibrated``) live in
``bracketlearn.trainers._factories`` and are re-exported below.
"""

from __future__ import annotations

from bracketlearn.trainers._factories import emos_calibrated, ridge
from bracketlearn.trainers.bracket import CumulativeBinary
from bracketlearn.trainers.combiners import (
    BMAStacking,
    BracketStacking,
    CDFBoostBracket,
    DistAsFeatures,
    LinearPoolDist,
    StackedParametric,
    TailSpecialist,
)
from bracketlearn.trainers.parametric import (
    EMOS,
    BayesianRidge,
    HeteroscedasticNormal,
    HierarchicalNormal,
    MixtureNormals,
    NGBoostNormal,
)
from bracketlearn.trainers.point import OnlineAggregator, RNNHourly, SklearnPoint
from bracketlearn.trainers.quantile import QuantileForest, QuantileReg

__all__ = [
    # point
    "OnlineAggregator",
    "RNNHourly",
    "SklearnPoint",
    # parametric
    "BayesianRidge",
    "EMOS",
    "HeteroscedasticNormal",
    "HierarchicalNormal",
    "MixtureNormals",
    "NGBoostNormal",
    # quantile
    "QuantileForest",
    "QuantileReg",
    # bracket
    "CumulativeBinary",
    # combiners
    "BMAStacking",
    "BracketStacking",
    "CDFBoostBracket",
    "DistAsFeatures",
    "LinearPoolDist",
    "StackedParametric",
    "TailSpecialist",
    # factories
    "emos_calibrated",
    "ridge",
]
