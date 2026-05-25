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
    """Isotonic calibration on bracket probabilities.

    Mirrors prediction_market_weather/ml/trainers/emos_calibrated.py:

    1. Discretise the dist onto `edges` via dist.cdf(edges).
    2. Flatten to long-form (p_pred, y_hit) pairs across (rows × brackets).
    3. Fit sklearn.IsotonicRegression on (p_pred, y_hit) with [0, 1] clipping.
    4. transform(): apply isotonic to bracket probs, renormalise per row,
       return a bracket-backed DistributionForecast.

    Calibrated output is bracket-backed regardless of input backing — the
    isotonic correction is meaningful only relative to the chosen ladder.

    `edges` is required (Rule #0.5: no default ladder).
    """

    edges: np.ndarray
    iso_: Any = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)
    n_calib_: int | None = field(default=None, init=False)

    def fit(
        self,
        dist_oof: "DistributionForecast",
        y: np.ndarray,
    ) -> Self:
        from sklearn.isotonic import IsotonicRegression

        from bracketlearn.forecast import Backing, DistributionForecast

        y = np.asarray(y, dtype=float)
        edges = np.asarray(self.edges, dtype=float)
        B = edges.shape[0] - 1
        # Per-row bracket probs from the dist's CDF.
        probs = _bracket_probs_from_dist(dist_oof, edges)        # (N, B)
        # Realized bin per row.
        bin_idx = np.searchsorted(edges, y, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, B - 1)
        # One-hot the realized bins.
        onehot = np.zeros_like(probs)
        onehot[np.arange(probs.shape[0]), bin_idx] = 1.0
        p_long = probs.reshape(-1)
        y_long = onehot.reshape(-1)
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso_.fit(p_long, y_long)
        self.n_calib_ = int(y.shape[0])
        self.fitted_ = True
        return self

    def transform(
        self,
        dist: "DistributionForecast",
    ) -> "DistributionForecast":
        if not self.fitted_:
            raise RuntimeError("Isotonic.transform called before fit")
        from bracketlearn.forecast import DistributionForecast, ProvenanceMeta

        edges = np.asarray(self.edges, dtype=float)
        probs = _bracket_probs_from_dist(dist, edges)
        cal = self.iso_.predict(probs.reshape(-1)).reshape(probs.shape)
        row_sum = cal.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        cal = cal / row_sum
        new_prov = ProvenanceMeta(
            **{**dist.provenance.__dict__,
               "conversion_chain": dist.provenance.conversion_chain + ("Isotonic",),
               "created_at": datetime.now()},
        )
        return DistributionForecast.from_brackets(
            edges=edges, probs=cal,
            ids=dist.ids, timestamps=dist.timestamps,
            provenance=new_prov,
        )


def _bracket_probs_from_dist(
    dist: "DistributionForecast", edges: np.ndarray,
) -> np.ndarray:
    """Per-row bracket probabilities from any dist that supports cdf()."""
    cdf_hi = dist.cdf(edges[1:])
    cdf_lo = dist.cdf(edges[:-1])
    probs = cdf_hi - cdf_lo
    # Numerical clip — Σ may drift slightly from 1 for parametric backings
    # because we're discretising an unbounded distribution.
    probs = np.clip(probs, 0.0, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return probs / row_sum


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
