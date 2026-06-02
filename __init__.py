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

__version__ = "0.5.0"

from bracketlearn.adapters import (
    BinaryAbove,
    BinaryBelow,
    BracketLadder,
    ThresholdLadder,
    Twin,
)
from bracketlearn.base import BaseEstimator, clone
from bracketlearn.baselines import EmpiricalDistribution, Persistence, PersistenceDist
from bracketlearn.forecast import (
    BracketForecast,
    ContractForecast,
    DistributionForecast,
    MixtureNormalForecast,
    NormalForecast,
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
    PITCalibrate,
    StudentTResidual,
)
from bracketlearn.multitarget import (
    MultiOutputForecastPipeline,
    MultiOutputPipelineResult,
)
from bracketlearn.compose import Stacker, WalkForward
from bracketlearn.pipeline import (
    CalibratedForecaster,
    ForecastPipeline,
    LiftedForecaster,
    Pipeline,
    PipelineResult,
)
from bracketlearn.restrict import BracketMask
from bracketlearn.search import GridSearch
from bracketlearn.trainers import (
    EMOS,
    BayesianRidge,
    BMAStacking,
    BracketStacking,
    CDFBoostBracket,
    CumulativeBinary,
    DistAsFeatures,
    HierarchicalNormal,
    LinearPoolDist,
    MixtureNormals,
    NGBoostNormal,
    OnlineAggregator,
    QuantileForest,
    QuantileReg,
    RNNHourly,
    SklearnPoint,
    StackedParametric,
    Stacking,
    TailSpecialist,
)
from bracketlearn.transform import GroupByZScore, IdentityTransformer
from bracketlearn.transformers import BracketExpander

__all__ = [
    "__version__",
    # base
    "BaseEstimator",
    "clone",
    # data
    "BracketForecast",
    "ContractForecast",
    "DistributionForecast",
    "MixtureNormalForecast",
    "NormalForecast",
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
    "PersistenceDist",
    # trainers
    "BMAStacking",
    "BayesianRidge",
    "BracketStacking",
    "HierarchicalNormal",
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
    "StackedParametric",
    "Stacking",  # legacy alias for StackedParametric
    "TailSpecialist",
    # transformers
    "BracketExpander",
    "GroupByZScore",
    "IdentityTransformer",
    # restriction
    "BracketMask",
    # lifters / calibrators
    "ConformalCalibrate",
    "GARCHResidual",
    "GlobalResidual",
    "Isotonic",
    "PITCalibrate",
    "StudentTResidual",
    # composites
    "CalibratedForecaster",
    "LiftedForecaster",
    # pipeline
    "ForecastPipeline",
    "Pipeline",
    "PipelineResult",
    "Stacker",
    "WalkForward",
    "GridSearch",
    "MultiOutputForecastPipeline",
    "MultiOutputPipelineResult",
]
