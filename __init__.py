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

__version__ = "0.2.0"

from bracketlearn.adapters import (
    BinaryAbove,
    BinaryBelow,
    BracketLadder,
    ContractAdapter,
    PerRowBracketLadder,
    ThresholdLadder,
    Twin,
)
from bracketlearn.base import BaseEstimator, clone
from bracketlearn.baselines import EmpiricalDistribution, Persistence
from bracketlearn.pipeline import CalibratedForecaster, LiftedForecaster
from bracketlearn.forecast import (
    Backing,
    ContractForecast,
    ContractSpec,
    DistributionForecast,
    ParametricFamily,
    PointForecast,
    ProvenanceMeta,
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
from bracketlearn.pipeline import ForecastPipeline, PipelineResult
from bracketlearn.restrict import BracketMask
from bracketlearn.protocols import (
    Calibrator,
    DistForecaster,
    Forecaster,
    Lifter,
    PointForecaster,
)
from bracketlearn.search import GridSearch
from bracketlearn.forecast import TailPolicy, TailPolicyError, TailRule
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

# Optional convenience builders (these import lazily; OK at top level too).
try:
    from bracketlearn.trainers import emos_calibrated, market_ols, ridge
except ImportError:  # pragma: no cover
    emos_calibrated = market_ols = ridge = None  # type: ignore[assignment]

__all__ = [
    "__version__",
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
    "normalize_bracket_probs",
    # protocols
    "Calibrator",
    "DistForecaster",
    "Forecaster",
    "Lifter",
    "PointForecaster",
    # tail
    "TailPolicy",
    "TailPolicyError",
    "TailRule",
    # adapters
    "BinaryAbove",
    "BinaryBelow",
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
