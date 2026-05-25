"""Point → Distribution lifters (§6) + Dist → Dist calibrators.

v0.1 ships:
- GlobalResidual      — Lifter: iid Gaussian residuals, one σ.
- Isotonic            — Calibrator: per-bracket isotonic calibration.
- ConformalCalibrate  — Calibrator: per-τ conformal coverage on quantile dists.

Planned for v0.2 (see README "Not yet" section):
- SisterModel, ConditionalVariance, Conformal lifters
- Bootstrap, IsotonicCDF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from bracketlearn.base import BaseEstimator

if TYPE_CHECKING:
    from bracketlearn.forecast import DistributionForecast, PointForecast


# ---------------------------------------------------------------------------
# GlobalResidual — iid Gaussian residuals.
# ---------------------------------------------------------------------------


@dataclass
class GlobalResidual(BaseEstimator):
    """Fits one σ from OOF residuals. Produces parametric normal."""

    family: Literal["normal"] = "normal"      # student_t reserved for v0.2
    requires_X: bool = False
    sigma_: float | None = field(default=None, init=False)

    def fit(
        self,
        point_oof: PointForecast,
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

    def lift(self, point: PointForecast) -> DistributionForecast:
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
# Isotonic — Calibrator (bracket-probability isotonic regression).
# ---------------------------------------------------------------------------


@dataclass
class Isotonic(BaseEstimator):
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
        dist_oof: DistributionForecast,
        y: np.ndarray,
    ) -> Self:
        from sklearn.isotonic import IsotonicRegression


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
        dist: DistributionForecast,
    ) -> DistributionForecast:
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
    dist: DistributionForecast, edges: np.ndarray,
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
class ConformalCalibrate(BaseEstimator):
    """Conformalised Quantile Regression (Romano et al. 2019).

    Calibrator (Dist → Dist) for quantile-backed forecasts. For each τ,
    learns an offset δ_τ on the held-out calibration set such that
    q̂_τ - δ_τ covers (1-τ) of the calibration rows from below.

    Under exchangeability, the (1-τ)-coverage guarantee carries to test.
    Operates only on quantile backings (rejects others loudly per Rule #0.5).
    """

    mode: Literal["per-tau", "symmetric"] = "per-tau"
    offsets_: np.ndarray | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        dist_oof: DistributionForecast,
        y: np.ndarray,
    ) -> Self:
        from bracketlearn.forecast import Backing

        if dist_oof.backing != Backing.QUANTILE:
            raise NotImplementedError(
                f"ConformalCalibrate expects quantile backing; got {dist_oof.backing}"
            )
        y = np.asarray(y, dtype=float)
        taus = dist_oof.taus
        qvals = dist_oof.qvals      # (N, Q)
        if self.mode != "per-tau":
            raise NotImplementedError(f"mode={self.mode!r} not in tier-2")
        # δ_τ = quantile of (q̂_τ - y) at level (1-τ).
        residuals = qvals - y[:, None]      # (N, Q)
        offsets = np.zeros(taus.shape[0])
        for j, tau in enumerate(taus):
            offsets[j] = float(np.quantile(residuals[:, j], 1.0 - tau))
        self.offsets_ = offsets
        self.fitted_ = True
        return self

    def transform(
        self,
        dist: DistributionForecast,
    ) -> DistributionForecast:
        from bracketlearn.forecast import Backing, DistributionForecast, ProvenanceMeta

        if not self.fitted_:
            raise RuntimeError("ConformalCalibrate.transform called before fit")
        if dist.backing != Backing.QUANTILE:
            raise NotImplementedError(
                f"ConformalCalibrate expects quantile backing; got {dist.backing}"
            )
        if not np.array_equal(dist.taus, np.arange(self.offsets_.shape[0])) and \
           dist.taus.shape[0] != self.offsets_.shape[0]:
            raise ValueError(
                f"ConformalCalibrate: shape mismatch — calibrated for Q={self.offsets_.shape[0]} "
                f"taus, dist has Q={dist.taus.shape[0]}"
            )
        # Apply δ_τ shift; isotonic-repair afterwards to keep monotonicity.
        qvals = dist.qvals - self.offsets_[None, :]
        qvals = np.maximum.accumulate(qvals, axis=1)
        new_prov = ProvenanceMeta(
            **{**dist.provenance.__dict__,
               "conversion_chain": dist.provenance.conversion_chain + ("ConformalCalibrate",),
               "created_at": datetime.now()},
        )
        return DistributionForecast.from_quantiles(
            taus=dist.taus, qvals=qvals,
            tail_policy=dist.tail_policy,
            ids=dist.ids, timestamps=dist.timestamps,
            provenance=new_prov,
        )
