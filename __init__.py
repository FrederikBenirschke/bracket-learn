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
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base machinery
# ---------------------------------------------------------------------------

from bracketlearn.base import BaseEstimator, clone

# ---------------------------------------------------------------------------
# Core data objects
# ---------------------------------------------------------------------------

from bracketlearn.forecast import (
    Backing,
    ContractForecast,
    ContractSpec,
    DistributionForecast,
    ParametricFamily,
    PointForecast,
    ProvenanceMeta,
)

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

from bracketlearn.protocols import (
    Calibrator,
    DistForecaster,
    Forecaster,
    Lifter,
    PointForecaster,
    StepLearner,
)

# ---------------------------------------------------------------------------
# Tail policy
# ---------------------------------------------------------------------------

from bracketlearn.tail import TailPolicy, TailPolicyError, TailRule

# ---------------------------------------------------------------------------
# Contract adapters
# ---------------------------------------------------------------------------

from bracketlearn.adapters import (
    BinaryAbove,
    BinaryBelow,
    BracketEdges,
    BracketLadder,
    ContractAdapter,
    PerRowBracketLadder,
    ThresholdLadder,
    Twin,
)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

from bracketlearn.baselines import EmpiricalDistribution, Persistence

# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

from bracketlearn.trainers import (
    CDFBoostBracket,
    CumulativeBinary,
    DistAsFeatures,
    EMOS,
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

# Optional convenience builders (these import lazily; OK at top level too).
try:
    from bracketlearn.trainers import emos_calibrated, market_ols, ridge
except ImportError:  # pragma: no cover
    emos_calibrated = market_ols = ridge = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Lifters / calibrators
# ---------------------------------------------------------------------------

from bracketlearn.lift import (
    ConformalCalibrate,
    GARCHResidual,
    GlobalResidual,
    Isotonic,
    StudentTResidual,
)

# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

from bracketlearn.composite import CalibratedForecaster, LiftedForecaster

# ---------------------------------------------------------------------------
# Pipeline + search + multi-target
# ---------------------------------------------------------------------------

from bracketlearn.pipeline import ForecastPipeline, PipelineResult
from bracketlearn.search import GridSearch
from bracketlearn.multitarget import (
    MultiOutputForecastPipeline,
    MultiOutputPipelineResult,
)


__all__ = [
    # base
    "BaseEstimator",
    "clone",
    # data
    "Backing",
    "ContractForecast",
    "ContractSpec",
    "DistributionForecast",
    "ParametricFamily",
    "PointForecast",
    "ProvenanceMeta",
    # protocols
    "Calibrator",
    "DistForecaster",
    "Forecaster",
    "Lifter",
    "PointForecaster",
    "StepLearner",
    # tail
    "TailPolicy",
    "TailPolicyError",
    "TailRule",
    # adapters
    "BinaryAbove",
    "BinaryBelow",
    "BracketEdges",
    "BracketLadder",
    "ContractAdapter",
    "PerRowBracketLadder",
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
    "emos_calibrated",
    "market_ols",
    "ridge",
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
