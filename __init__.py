"""bracketlearn — sklearn-style probabilistic-forecasting and bracket-contract pricing.

Most probabilistic-forecasting libraries stop at "predict a distribution."
bracketlearn keeps going: every forecast is a typed `DistributionForecast`
that converts to a bracket ladder and prices the resulting contracts.
Calibration, conformal correction, and tail specialisation are first-class
transformer stages.

Quick start::

    from bracketlearn import (
        ForecastPipeline, BracketLadder,
        EMOS, QuantileReg, SklearnPoint,
        LiftedForecaster, GlobalResidual,
    )

Less commonly used symbols live in their submodules:
- ``bracketlearn.protocols`` — Forecaster, PointForecaster, DistForecaster,
  Lifter, Calibrator (for users writing custom stages).
- ``bracketlearn.adapters.ContractAdapter`` — the contract-pricing protocol.
- ``bracketlearn.forecast`` — ContractSpec, ProvenanceMeta, TailPolicyError.
- ``bracketlearn.trainers`` — ``ridge``, ``emos_calibrated`` convenience
  factories.
"""

from __future__ import annotations

__version__ = "0.3.0"

from bracketlearn.adapters import (
    BinaryAbove,
    BinaryBelow,
    BracketLadder,
    ThresholdLadder,
    Twin,
)
from bracketlearn.base import BaseEstimator, clone
from bracketlearn.baselines import EmpiricalDistribution, Persistence
from bracketlearn.forecast import (
    Backing,
    BracketForecast,
    ContractForecast,
    DistributionForecast,
    MixtureNormalForecast,
    NormalForecast,
    ParametricFamily,
    PointForecast,
    QuantileForecast,
    StudentTForecast,
    TailPolicy,
    TailRule,
    normalize_bracket_probs,
)
from bracketlearn.lift import (
    ConformalCalibrate,
    GARCHResidual,
    GlobalResidual,
    Isotonic,
    StudentTResidual,
)
from bracketlearn.multitarget import (
    MultiOutputForecastPipeline,
    MultiOutputPipelineResult,
)
from bracketlearn.pipeline import (
    CalibratedForecaster,
    ForecastPipeline,
    LiftedForecaster,
    PipelineResult,
)
from bracketlearn.restrict import BracketMask
from bracketlearn.search import GridSearch
from bracketlearn.trainers import (
    EMOS,
    CDFBoostBracket,
    CumulativeBinary,
    DistAsFeatures,
    LinearPoolDist,
    MixtureNormals,
    NGBoostNormal,
    OnlineAggregator,
    QuantileForest,
    QuantileReg,
    RNNHourly,
    SklearnPoint,
    Stacking,
    TailSpecialist,
)

__all__ = [
    "__version__",
    # base
    "BaseEstimator",
    "clone",
    # data
    "Backing",
    "BracketForecast",
    "ContractForecast",
    "DistributionForecast",
    "MixtureNormalForecast",
    "NormalForecast",
    "ParametricFamily",
    "PointForecast",
    "QuantileForecast",
    "StudentTForecast",
    "TailPolicy",
    "TailRule",
    "normalize_bracket_probs",
    # adapters
    "BinaryAbove",
    "BinaryBelow",
    "BracketLadder",
    "ThresholdLadder",
    "Twin",
    # baselines
    "EmpiricalDistribution",
    "Persistence",
    # trainers
    "CDFBoostBracket",
    "CumulativeBinary",
    "DistAsFeatures",
    "EMOS",
    "LinearPoolDist",
    "MixtureNormals",
    "NGBoostNormal",
    "OnlineAggregator",
    "QuantileForest",
    "QuantileReg",
    "RNNHourly",
    "SklearnPoint",
    "Stacking",
    "TailSpecialist",
    # restriction
    "BracketMask",
    # lifters / calibrators
    "ConformalCalibrate",
    "GARCHResidual",
    "GlobalResidual",
    "Isotonic",
    "StudentTResidual",
    # composites
    "CalibratedForecaster",
    "LiftedForecaster",
    # pipeline
    "ForecastPipeline",
    "PipelineResult",
    "GridSearch",
    "MultiOutputForecastPipeline",
    "MultiOutputPipelineResult",
]
