"""bracketlearn — forecasting + contract pricing framework.

sklearn-style API for forecasts that get priced against tradeable contracts.
See /tmp/bracketcast_concept_v0.2.md for the design document.

v0.1 stubs — signatures only, no implementations.
"""

from bracketlearn.baselines import EmpiricalDistribution, Persistence
from bracketlearn.forecast import (
    ContractForecast,
    DistributionForecast,
    PointForecast,
    ProvenanceMeta,
)
from bracketlearn.protocols import (
    Calibrator,
    DistForecaster,
    Forecaster,
    Lifter,
    PointForecaster,
    StepLearner,
)
from bracketlearn.tail import TailPolicy, TailRule
from bracketlearn.trainers import CDFBoostBracket, DistAsFeatures, LinearPoolDist

__all__ = [
    "CDFBoostBracket",
    "Calibrator",
    "ContractForecast",
    "DistAsFeatures",
    "DistForecaster",
    "DistributionForecast",
    "EmpiricalDistribution",
    "Forecaster",
    "Lifter",
    "LinearPoolDist",
    "Persistence",
    "PointForecast",
    "PointForecaster",
    "ProvenanceMeta",
    "StepLearner",
    "TailPolicy",
    "TailRule",
]
