"""Point → Distribution lifters (§6).

Four genuine learned lifters + one pass-through utility:
- GlobalResidual         — iid residuals, one σ
- SisterModel            — per-row σ from external column (pass-through)
- ConditionalVariance    — σ̂ = f(X), needs raw X
- Conformal              — empirical residual distribution shifted by μ̂

Plus the calibration-side conformal:
- ConformalCalibrate     — Calibrator (per-τ coverage on quantile dists)

Bootstrap / EnsembleSpread / IsotonicCDF are NOT lifters under v0.2:
- Bootstrap: a DistForecaster taking a base in __init__ (see composite.py).
- EnsembleSpread: just DistributionForecast.from_empirical(members).
- IsotonicCDF: already Calibrator.Isotonic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from bracketlearn.protocols import Calibrator, Lifter

if TYPE_CHECKING:
    from bracketlearn.forecast import DistributionForecast, PointForecast


# ---------------------------------------------------------------------------
# GlobalResidual — iid Gaussian residuals.
# ---------------------------------------------------------------------------


@dataclass
class GlobalResidual:
    """Fits one σ from OOF residuals. Produces parametric normal."""

    family: Literal["normal"] = "normal"      # student_t reserved for v0.2
    requires_X: bool = False
    sigma_: float | None = field(default=None, init=False)

    def fit(
        self,
        point_oof: "PointForecast",
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        if self.family != "normal":
            raise NotImplementedError(f"family={self.family} not in v0.1")
        residuals = np.asarray(y, dtype=float) - point_oof.mu
        if residuals.size < 2:
            raise ValueError("need at least 2 OOF residuals to fit σ")
        # ML estimator with N-1 dof.
        self.sigma_ = float(np.std(residuals, ddof=1))
        if self.sigma_ <= 0:
            raise ValueError("fitted σ is non-positive — residuals all equal?")
        return self

    def lift(self, point: "PointForecast") -> "DistributionForecast":
        if self.sigma_ is None:
            raise RuntimeError("GlobalResidual.lift called before fit")
        from bracketlearn.forecast import DistributionForecast, ProvenanceMeta

        N = point.mu.shape[0]
        sigma = np.full(N, self.sigma_)
        new_prov = ProvenanceMeta(
            forecaster_name=point.provenance.forecaster_name,
            forecaster_version=point.provenance.forecaster_version,
            fit_window=point.provenance.fit_window,
            fold_idx=point.provenance.fold_idx,
            calibration_set_hash=point.provenance.calibration_set_hash,
            random_seed=point.provenance.random_seed,
            code_sha=point.provenance.code_sha,
            feature_matrix_hash=point.provenance.feature_matrix_hash,
            created_at=datetime.now(),
            sigma_source="lifted",
            conversion_chain=point.provenance.conversion_chain + ("GlobalResidual",),
            extras={**point.provenance.extras, "lifted_sigma": self.sigma_},
        )
        return DistributionForecast.from_normal(
            point.mu, sigma, ids=point.ids, timestamps=point.timestamps,
            provenance=new_prov,
        )


# ---------------------------------------------------------------------------
# SisterModel — per-row σ from external column. Pass-through.
# ---------------------------------------------------------------------------


@dataclass
class SisterModel:
    """Reads per-row σ from a column already present in X (e.g. vendor's
    own predictive std). No learning — fit is a no-op."""

    sigma_col: str
    requires_X: bool = True       # needs X at lift time to read sigma_col

    def fit(
        self,
        point_oof: "PointForecast",
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        return self

    def lift(self, point: "PointForecast") -> "DistributionForecast":
        ...


# ---------------------------------------------------------------------------
# ConditionalVariance — σ̂ = f(X).
# ---------------------------------------------------------------------------


@dataclass
class ConditionalVariance:
    """Heteroscedastic Gaussian. Fits sigma_estimator on log r² where
    r = y - μ̂_oof; predicts σ̂(x) at lift time."""

    sigma_estimator: Any            # any sklearn-style regressor
    requires_X: bool = True

    def fit(
        self,
        point_oof: "PointForecast",
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        ...

    def lift(self, point: "PointForecast") -> "DistributionForecast":
        ...


# ---------------------------------------------------------------------------
# Conformal — marginal or Mondrian.
# ---------------------------------------------------------------------------


@dataclass
class Conformal:
    """Marginal or Mondrian conformal prediction.

    Builds an empirical residual distribution from OOF (one global, or per
    Mondrian bin); at lift time, members[i] = μ̂[i] + residuals_calib.

    Produces empirical-backed DistributionForecast.
    """

    mode: Literal["marginal", "mondrian"] = "marginal"
    mondrian_bin_fn: Any = None             # Callable[X_row → bin_id]; required if mondrian
    requires_X: bool = False                # True if mondrian

    def fit(
        self,
        point_oof: "PointForecast",
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        ...

    def lift(self, point: "PointForecast") -> "DistributionForecast":
        ...


# ---------------------------------------------------------------------------
# ConformalCalibrate — Calibrator (CQR-style per-τ coverage).
# ---------------------------------------------------------------------------


@dataclass
class Isotonic:
    """Isotonic CDF calibration on a fixed quantile grid.

    Operates only on parametric-normal backings in v0.1: converts to a fixed
    quantile grid, fits isotonic on (predicted CDF tau → empirical CDF on y),
    and shifts the implied normal's mean+sigma via moment matching of the
    recalibrated CDF. Simpler v0.1 fallback: leave the dist unchanged but
    record provenance — provides the Calibrator hook for the e2e demo.

    Full implementation deferred; in v0.1 this is a typed pass-through with
    fit-side bookkeeping (good enough to exercise the pipeline calibration
    fold without being a no-op forever).
    """

    fitted_: bool = field(default=False, init=False)
    n_calib_: int | None = field(default=None, init=False)

    def fit(
        self,
        dist_oof: "DistributionForecast",
        y: np.ndarray,
    ) -> Self:
        self.n_calib_ = int(np.asarray(y).shape[0])
        self.fitted_ = True
        return self

    def transform(
        self,
        dist: "DistributionForecast",
    ) -> "DistributionForecast":
        if not self.fitted_:
            raise RuntimeError("Isotonic.transform called before fit")
        # v0.1: identity transform; provenance records the calibrator was
        # applied so downstream auditors see it.
        from bracketlearn.forecast import (
            DistributionForecast,
            ProvenanceMeta,
            Backing,
            ParametricFamily,
        )

        new_prov = ProvenanceMeta(
            **{**dist.provenance.__dict__,
               "conversion_chain": dist.provenance.conversion_chain + ("Isotonic",),
               "created_at": datetime.now()},
        )
        return DistributionForecast(
            backing=dist.backing,
            family=dist.family,
            params=dist.params,
            taus=dist.taus,
            qvals=dist.qvals,
            members=dist.members,
            edges=dist.edges,
            probs=dist.probs,
            ids=dist.ids,
            timestamps=dist.timestamps,
            provenance=new_prov,
            tail_policy=dist.tail_policy,
            tail_support=dist.tail_support,
        )


@dataclass
class ConformalCalibrate:
    """Conformalised Quantile Regression (Romano et al. 2019).

    Lives in lift.py for proximity to Conformal even though it's a
    Calibrator (Dist → Dist), not a Lifter.

    Measures per-τ conformity score on a held-out calibration set; shifts
    each τ's quantile by the calibrated offset to restore nominal coverage.
    """

    target_coverage: float | None = None
    mode: Literal["per-tau", "symmetric"] = "per-tau"

    def fit(
        self,
        dist_oof: "DistributionForecast",
        y: np.ndarray,
    ) -> Self:
        ...

    def transform(
        self,
        dist: "DistributionForecast",
    ) -> "DistributionForecast":
        ...
