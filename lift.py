"""Point → Distribution lifters (§6) + Dist → Dist calibrators.

v0.1 ships:
- GlobalResidual      — Lifter: iid Gaussian residuals, one σ.
- StudentTResidual    — Lifter: iid Student-t residuals, MLE (σ, ν).
- GARCHResidual       — Lifter: time-varying σ from GARCH(1,1), one-step.
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

    requires_X: bool = False
    sigma_: float | None = field(default=None, init=False)

    def fit(
        self,
        point_oof: PointForecast,
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
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
# StudentTResidual — iid Student-t residuals (MLE).
# ---------------------------------------------------------------------------


@dataclass
class StudentTResidual(BaseEstimator):
    """Fits (σ, ν) from OOF residuals via MLE. Produces parametric student_t.

    Use when residuals are fat-tailed relative to Gaussian (sports margins,
    short-horizon returns). ν is constrained to (2.1, df_max) so the marginal
    variance is finite; ν close to the lower bound indicates heavy tails.
    """

    df_min: float = 2.1
    df_max: float = 200.0
    requires_X: bool = False
    sigma_: float | None = field(default=None, init=False)
    df_: float | None = field(default=None, init=False)

    def fit(
        self,
        point_oof: PointForecast,
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        from scipy import stats as _stats

        residuals = np.asarray(y, dtype=float) - point_oof.mu
        if residuals.size < 10:
            raise ValueError("need at least 10 OOF residuals to fit (σ, ν)")
        # scipy MLE with loc fixed at 0 — point forecast assumed unbiased.
        df, _loc, scale = _stats.t.fit(residuals, floc=0.0)
        # Clip ν into the configured range; if df_min is hit we still raise
        # (silent clipping would mask a degenerate fit).
        if not (self.df_min < df < self.df_max):
            # Out of range — if too low, residuals have infinite variance per
            # MLE; refuse rather than ship a finite-variance lie.
            if df <= self.df_min:
                raise ValueError(
                    f"MLE df={df:.2f} ≤ df_min={self.df_min}; residuals are "
                    f"too heavy-tailed for finite-variance Student-t. "
                    f"Reduce df_min only if you know what you're doing."
                )
            # If too high, the t is indistinguishable from Gaussian — clip up
            # to df_max (the cap is a numerical convenience, not a statement).
            df = self.df_max
        if scale <= 0:
            raise ValueError("fitted σ is non-positive — residuals all equal?")
        self.df_ = float(df)
        self.sigma_ = float(scale)
        return self

    def lift(self, point: PointForecast) -> DistributionForecast:
        if self.sigma_ is None or self.df_ is None:
            raise RuntimeError("StudentTResidual.lift called before fit")
        from bracketlearn.forecast import DistributionForecast, ProvenanceMeta

        N = point.mu.shape[0]
        sigma = np.full(N, self.sigma_)
        df = np.full(N, self.df_)
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
            conversion_chain=point.provenance.conversion_chain + ("StudentTResidual",),
            extras={
                **point.provenance.extras,
                "lifted_sigma": self.sigma_,
                "lifted_df": self.df_,
            },
        )
        return DistributionForecast.from_student_t(
            point.mu, sigma, df,
            ids=point.ids, timestamps=point.timestamps,
            provenance=new_prov,
        )


# ---------------------------------------------------------------------------
# GARCHResidual — time-varying σ from GARCH(1,1), one-step ahead.
# ---------------------------------------------------------------------------


@dataclass
class GARCHResidual(BaseEstimator):
    """Fits GARCH(1,1) on OOF residuals; lifts to per-row σ.

    Volatility recursion: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}, with the
    residual mean assumed zero (point forecast unbiased).

    One-step semantics (per user choice): every lift() row receives the
    forecasted σ for the next observation given the fitted residual history,
    i.e. σ̂² = ω + α·r²_T + β·σ²_T where T is the last fit-residual index.
    Multi-horizon mean-reversion is not implemented — pass timestamps that
    match the one-step convention.

    family="normal" (default) produces a parametric normal output; "student_t"
    additionally fits a Student-t df on the standardised residuals
    (r_t / σ_t) and produces a parametric student_t output.
    """

    family: Literal["normal", "student_t"] = "normal"
    requires_X: bool = False
    omega_: float | None = field(default=None, init=False)
    alpha_: float | None = field(default=None, init=False)
    beta_: float | None = field(default=None, init=False)
    sigma2_next_: float | None = field(default=None, init=False)
    df_: float | None = field(default=None, init=False)

    def fit(
        self,
        point_oof: PointForecast,
        y: np.ndarray,
        *,
        X: Any | None = None,
    ) -> Self:
        from scipy.optimize import minimize

        residuals = np.asarray(y, dtype=float) - point_oof.mu
        T = residuals.shape[0]
        if T < 30:
            raise ValueError("need at least 30 OOF residuals to fit GARCH(1,1)")

        r2 = residuals ** 2
        var_uncond = float(np.var(residuals, ddof=1))
        if var_uncond <= 0:
            raise ValueError("residual variance is non-positive — all equal?")

        def _recurse(omega: float, alpha: float, beta: float) -> np.ndarray:
            sigma2 = np.empty(T)
            sigma2[0] = var_uncond
            for t in range(1, T):
                sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
            return sigma2

        def _neg_loglik(theta: np.ndarray) -> float:
            omega, alpha, beta = theta
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
                return 1e12
            sigma2 = _recurse(omega, alpha, beta)
            if np.any(sigma2 <= 0):
                return 1e12
            # Gaussian log-likelihood on the residual series.
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r2 / sigma2)
            return -ll

        # Initial guess: targeting unconditional variance with α=0.05, β=0.9.
        alpha0, beta0 = 0.05, 0.90
        omega0 = var_uncond * (1.0 - alpha0 - beta0)
        result = minimize(
            _neg_loglik,
            x0=np.array([omega0, alpha0, beta0]),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
        )
        if not result.success:
            raise RuntimeError(f"GARCH MLE failed: {result.message}")
        omega, alpha, beta = result.x
        if alpha + beta >= 0.999:
            raise ValueError(
                f"GARCH near-IGARCH: α+β={alpha+beta:.4f}; refusing to ship a "
                f"non-stationary fit"
            )
        sigma2 = _recurse(omega, alpha, beta)
        # One-step σ² for the *next* obs given history up to T-1.
        sigma2_next = omega + alpha * r2[-1] + beta * sigma2[-1]
        self.omega_ = float(omega)
        self.alpha_ = float(alpha)
        self.beta_ = float(beta)
        self.sigma2_next_ = float(sigma2_next)

        if self.family == "student_t":
            from scipy import stats as _stats
            z = residuals / np.sqrt(sigma2)
            df, _loc, _scale = _stats.t.fit(z, floc=0.0, fscale=1.0)
            if df <= 2.1:
                raise ValueError(
                    f"GARCH-t standardised residuals have df={df:.2f} ≤ 2.1 "
                    f"(infinite variance)"
                )
            self.df_ = float(df)
        return self

    def lift(self, point: PointForecast) -> DistributionForecast:
        if self.sigma2_next_ is None:
            raise RuntimeError("GARCHResidual.lift called before fit")
        from bracketlearn.forecast import DistributionForecast, ProvenanceMeta

        N = point.mu.shape[0]
        sigma_next = float(np.sqrt(self.sigma2_next_))
        sigma = np.full(N, sigma_next)
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
            conversion_chain=point.provenance.conversion_chain + ("GARCHResidual",),
            extras={
                **point.provenance.extras,
                "garch_omega": self.omega_,
                "garch_alpha": self.alpha_,
                "garch_beta": self.beta_,
                "garch_sigma_next": sigma_next,
                **({"garch_df": self.df_} if self.df_ is not None else {}),
            },
        )
        if self.family == "student_t":
            df = np.full(N, self.df_)
            return DistributionForecast.from_student_t(
                point.mu, sigma, df,
                ids=point.ids, timestamps=point.timestamps,
                provenance=new_prov,
            )
        return DistributionForecast.from_normal(
            point.mu, sigma,
            ids=point.ids, timestamps=point.timestamps,
            provenance=new_prov,
        )


# ---------------------------------------------------------------------------
# Isotonic — Calibrator (bracket-probability isotonic regression).
# ---------------------------------------------------------------------------


@dataclass
class Isotonic(BaseEstimator):
    """Isotonic calibration on bracket probabilities.

    Inputs and outputs are :class:`BracketForecast` (per-row brackets).
    A single 1-D isotonic curve is fit on (predicted bracket prob,
    realized hit) pairs flattened across (rows × brackets). At
    transform time it's applied independently to every (row, bracket)
    cell, then each row is renormalised back to sum-to-1.

    v0.3 — drops the ``edges`` constructor arg. Callers that have a
    non-bracket dist should ``.integrate(edges_per_row)`` first, which
    works on any subclass and accepts per-row grids natively. The
    calibrator itself is grid-agnostic because the single isotonic
    curve maps (predicted-prob → calibrated-prob) without referencing
    the underlying bracket edges.

    Convenience: pass ``pre_integrate_edges`` (1-D shared, 2-D dense,
    or ragged sequence) to have Isotonic auto-integrate non-bracket
    inputs internally — useful in factories that wrap a parametric
    forecaster with bracket-prob calibration on a known ladder.
    """

    pre_integrate_edges: Any = None
    iso_: Any = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)
    n_calib_: int | None = field(default=None, init=False)

    def _maybe_integrate(self, dist: DistributionForecast) -> DistributionForecast:
        from bracketlearn.forecast import BracketForecast

        if isinstance(dist, BracketForecast):
            return dist
        if self.pre_integrate_edges is None:
            raise TypeError(
                f"Isotonic expects a BracketForecast input; got "
                f"{type(dist).__name__}. Either call dist.integrate(edges_per_row) "
                f"first, or construct Isotonic(pre_integrate_edges=...) so it "
                f"integrates internally."
            )
        return dist.integrate(self.pre_integrate_edges)

    def fit(
        self,
        dist_oof: DistributionForecast,
        y: np.ndarray,
    ) -> Self:
        from sklearn.isotonic import IsotonicRegression

        dist_oof = self._maybe_integrate(dist_oof)
        y = np.asarray(y, dtype=float)
        probs = dist_oof.probs                  # (N, B_max), NaN-padded for ragged rows
        bin_idx = dist_oof.realized_bin(y)      # (N,)
        N, B_max = probs.shape
        # One-hot the realized bin only over the row's valid prefix.
        # NaN-padded probs positions stay NaN in onehot, get filtered out
        # before passing to sklearn.
        onehot = np.full_like(probs, np.nan)
        valid_mask = ~np.isnan(probs)
        onehot[valid_mask] = 0.0
        onehot[np.arange(N), bin_idx] = 1.0
        # Flatten and drop NaN positions before fitting.
        finite_mask = valid_mask.ravel()
        p_long = probs.ravel()[finite_mask]
        y_long = onehot.ravel()[finite_mask]
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
        from bracketlearn.forecast import BracketForecast, ProvenanceMeta

        dist = self._maybe_integrate(dist)
        probs = dist.probs                      # (N, B_max), NaN-padded
        finite_mask = ~np.isnan(probs)
        cal = np.full_like(probs, np.nan)
        cal[finite_mask] = self.iso_.predict(probs[finite_mask])
        # Row-wise renorm over the row's valid prefix.
        row_sum = np.nansum(cal, axis=1, keepdims=True)
        if np.any(row_sum.ravel() <= 0):
            n_bad = int((row_sum.ravel() <= 0).sum())
            raise ValueError(
                f"Isotonic.transform: {n_bad}/{cal.shape[0]} rows have zero "
                f"calibrated mass — isotonic fit is degenerate (check fit data)."
            )
        with np.errstate(invalid="ignore"):
            cal = cal / row_sum
        new_prov = ProvenanceMeta(
            **{**dist.provenance.__dict__,
               "conversion_chain": dist.provenance.conversion_chain + ("Isotonic",),
               "created_at": datetime.now()},
        )
        return BracketForecast.from_arrays(
            edges=dist.edges, probs=cal,
            ids=dist.ids, timestamps=dist.timestamps,
            provenance=new_prov,
        )


@dataclass
class ConformalCalibrate(BaseEstimator):
    """Conformalised Quantile Regression (Romano et al. 2019).

    Calibrator (Dist → Dist) for quantile-backed forecasts. For each τ,
    learns an offset δ_τ on the held-out calibration set such that
    q̂_τ - δ_τ covers (1-τ) of the calibration rows from below.

    Under exchangeability, the (1-τ)-coverage guarantee carries to test.
    Operates only on quantile backings (rejects others loudly).
    """

    offsets_: np.ndarray | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        dist_oof: DistributionForecast,
        y: np.ndarray,
    ) -> Self:
        from bracketlearn.forecast import Backing

        if dist_oof.backing != Backing.QUANTILE:
            raise ValueError(
                f"ConformalCalibrate expects quantile backing; got {dist_oof.backing}"
            )
        y = np.asarray(y, dtype=float)
        taus = dist_oof.taus
        qvals = dist_oof.qvals      # (N, Q)
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
            raise ValueError(
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
