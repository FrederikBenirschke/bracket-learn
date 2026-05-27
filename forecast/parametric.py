"""Parametric DistributionForecast subclasses.

- ``_ParametricMixin`` — shared scipy-backed math for single-rv dists
  (Normal, Student-t). Subclasses declare ``_rv`` and override
  ``_per_row_params``.
- ``NormalForecast`` — single-rv Gaussian, (N,) μ and σ.
- ``StudentTForecast`` — single-rv Student-t, (N,) μ, σ, ν.
- ``MixtureNormalForecast`` — (N, K) weighted normals; standalone math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import stats as _stats

from bracketlearn.forecast._helpers import _quantile_via_brentq
from bracketlearn.forecast._meta import Backing, ParametricFamily, ProvenanceMeta
from bracketlearn.forecast.base import DistributionForecast


# ---------------------------------------------------------------------------
# _ParametricMixin — shared math for single-rv scipy-backed dists.
#
# Normal and Student-t differ only in which scipy.stats distribution they
# delegate to and which kwargs they pass. Both inherit cdf/cdf_at/cdf_at_grid/
# ppf/pdf from this mixin; subclasses declare ``_rv`` (the scipy distribution)
# and override ``_per_row_params`` to return the 1-D shape parameter arrays.
#
# Mixture stays standalone — its cdf needs a weighted sum over K components,
# which doesn't fit the single-rv shape.
# ---------------------------------------------------------------------------


class _ParametricMixin:
    _rv: ClassVar[Any]

    def _per_row_params(self) -> dict[str, np.ndarray]:
        """Per-row 1-D shape params for self._rv (e.g. {'loc': mu, 'scale': sigma})."""
        raise NotImplementedError

    def _bcast_params(self) -> dict[str, np.ndarray]:
        """_per_row_params broadcast to (N, 1) for (N, M)-shaped outputs."""
        return {k: v[:, None] for k, v in self._per_row_params().items()}

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        out = self._rv.cdf(x_arr[None, :], **self._bcast_params())
        return out[:, 0] if scalar else out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] != self.ids.shape[0]:
            raise ValueError(
                f"cdf_at: y has {y_arr.shape[0]} rows, dist has {self.ids.shape[0]}"
            )
        return self._rv.cdf(y_arr, **self._per_row_params())

    def cdf_at_grid(self, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError(f"cdf_at_grid: y must be 2-D (N, M); got shape {y_arr.shape}")
        if y_arr.shape[0] != self.ids.shape[0]:
            raise ValueError(
                f"cdf_at_grid: y has {y_arr.shape[0]} rows, dist has {self.ids.shape[0]}"
            )
        nan_mask = np.isnan(y_arr)
        y_safe = np.where(nan_mask, 0.0, y_arr)
        out = self._rv.cdf(y_safe, **self._bcast_params())
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        out = self._rv.ppf(tau_arr[None, :], **self._bcast_params())
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        out = self._rv.pdf(x_arr[None, :], **self._bcast_params())
        return out[:, 0] if scalar else out


# ---------------------------------------------------------------------------
# NormalForecast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalForecast(_ParametricMixin, DistributionForecast):
    mu: np.ndarray              # (N,)
    sigma: np.ndarray           # (N,)

    _rv: ClassVar = _stats.norm

    @classmethod
    def from_arrays(
        cls,
        *,
        mu: np.ndarray,
        sigma: np.ndarray,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> NormalForecast:
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if mu.shape != sigma.shape or mu.shape != ids.shape:
            raise ValueError(
                f"shape mismatch: mu={mu.shape} sigma={sigma.shape} ids={ids.shape}"
            )
        if np.any(sigma <= 0):
            raise ValueError("sigma must be strictly positive")
        return cls(
            ids=ids, timestamps=timestamps, provenance=provenance,
            mu=mu, sigma=sigma,
        )

    def _per_row_params(self) -> dict[str, np.ndarray]:
        return {"loc": self.mu, "scale": self.sigma}

    # compat: backing/family + params dict.
    @property
    def backing(self) -> Backing:
        return Backing.PARAMETRIC

    @property
    def family(self) -> ParametricFamily:
        return ParametricFamily.NORMAL

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"mu": self.mu, "sigma": self.sigma}

    @property
    def tail_policy(self):
        return None

    @property
    def tail_support(self) -> str:
        return "full"

    def mean(self):
        return self.mu.copy()

    def variance(self):
        return self.sigma ** 2

    def sample(self, n, rng):
        mu = self.mu[:, None]
        sigma = self.sigma[:, None]
        return rng.normal(loc=mu, scale=sigma, size=(mu.shape[0], n))

    def crps(self, y):
        from bracketlearn.score import crps_gaussian
        return crps_gaussian(self, y)

    def log_score(self, y):
        from bracketlearn.score import log_score_gaussian
        return log_score_gaussian(self, y)

    def to_point(self, *, how: str = "mean"):
        if how not in ("mean", "median", "mode"):
            raise ValueError(f"how={how!r} not in 'mean'/'median'/'mode'")
        return np.asarray(self.mu, dtype=float)

    @classmethod
    def stitch(cls, folds, *, timestamps, provenance):
        all_rows = np.concatenate([rows for rows, _ in folds])
        order = np.argsort(all_rows, kind="stable")
        ids_sorted = all_rows[order]
        ts_sorted = timestamps[all_rows][order]
        mu = np.concatenate([d.mu for _, d in folds])[order]
        sigma = np.concatenate([d.sigma for _, d in folds])[order]
        return cls.from_arrays(
            mu=mu, sigma=sigma,
            ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
        )


# ---------------------------------------------------------------------------
# StudentTForecast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentTForecast(_ParametricMixin, DistributionForecast):
    mu: np.ndarray
    sigma: np.ndarray
    df: np.ndarray

    _rv: ClassVar = _stats.t

    @classmethod
    def from_arrays(
        cls,
        *,
        mu: np.ndarray,
        sigma: np.ndarray,
        df: np.ndarray,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> StudentTForecast:
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        df = np.asarray(df, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if df.ndim == 0:
            df = np.full(mu.shape, float(df))
        if mu.shape != sigma.shape or mu.shape != ids.shape or mu.shape != df.shape:
            raise ValueError(
                f"shape mismatch: mu={mu.shape} sigma={sigma.shape} "
                f"df={df.shape} ids={ids.shape}"
            )
        if np.any(sigma <= 0):
            raise ValueError("sigma must be strictly positive")
        if np.any(df <= 2.0):
            raise ValueError("df must be > 2 (finite variance required)")
        return cls(
            ids=ids, timestamps=timestamps, provenance=provenance,
            mu=mu, sigma=sigma, df=df,
        )

    def _per_row_params(self) -> dict[str, np.ndarray]:
        return {"df": self.df, "loc": self.mu, "scale": self.sigma}

    @property
    def backing(self) -> Backing:
        return Backing.PARAMETRIC

    @property
    def family(self) -> ParametricFamily:
        return ParametricFamily.STUDENT_T

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"mu": self.mu, "sigma": self.sigma, "df": self.df}

    @property
    def tail_policy(self):
        return None

    @property
    def tail_support(self) -> str:
        return "full"

    def mean(self):
        return self.mu.copy()

    def variance(self):
        return self.sigma ** 2 * self.df / (self.df - 2.0)

    def sample(self, n, rng):
        mu = self.mu[:, None]
        sigma = self.sigma[:, None]
        N = mu.shape[0]
        out = np.empty((N, n))
        df_flat = self.df
        for i in range(N):
            out[i] = rng.standard_t(df_flat[i], size=n)
        return mu + sigma * out

    def crps(self, y):
        raise NotImplementedError(
            "StudentTForecast.crps not implemented — no closed-form CRPS for "
            "Student-t in score.py yet. Use MC via sample()."
        )

    def log_score(self, y):
        y_arr = np.asarray(y, dtype=float)
        return -_stats.t.logpdf(y_arr, df=self.df, loc=self.mu, scale=self.sigma)

    def to_point(self, *, how: str = "mean"):
        if how not in ("mean", "median", "mode"):
            raise ValueError(f"how={how!r} not in 'mean'/'median'/'mode'")
        return np.asarray(self.mu, dtype=float)

    @classmethod
    def stitch(cls, folds, *, timestamps, provenance):
        all_rows = np.concatenate([rows for rows, _ in folds])
        order = np.argsort(all_rows, kind="stable")
        ids_sorted = all_rows[order]
        ts_sorted = timestamps[all_rows][order]
        mu = np.concatenate([d.mu for _, d in folds])[order]
        sigma = np.concatenate([d.sigma for _, d in folds])[order]
        df = np.concatenate([d.df for _, d in folds])[order]
        return cls.from_arrays(
            mu=mu, sigma=sigma, df=df,
            ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
        )


# ---------------------------------------------------------------------------
# MixtureNormalForecast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureNormalForecast(DistributionForecast):
    weights: np.ndarray          # (N, K)
    mus: np.ndarray              # (N, K)
    sigmas: np.ndarray           # (N, K)

    @classmethod
    def from_arrays(
        cls,
        *,
        weights: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> MixtureNormalForecast:
        weights = np.asarray(weights, dtype=float)
        mus = np.asarray(mus, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if weights.shape != mus.shape or weights.shape != sigmas.shape:
            raise ValueError(
                f"shape mismatch: weights={weights.shape} mus={mus.shape} sigmas={sigmas.shape}"
            )
        if weights.shape[0] != ids.shape[0]:
            raise ValueError(
                f"N mismatch: weights={weights.shape[0]} ids={ids.shape[0]}"
            )
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("weights must sum to 1 per row")
        if np.any(sigmas <= 0):
            raise ValueError("sigmas must be strictly positive (components carry mass)")
        return cls(
            ids=ids, timestamps=timestamps, provenance=provenance,
            weights=weights, mus=mus, sigmas=sigmas,
        )

    @property
    def backing(self) -> Backing:
        return Backing.PARAMETRIC

    @property
    def family(self) -> ParametricFamily:
        return ParametricFamily.MIXTURE_NORMAL

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights, "mus": self.mus, "sigmas": self.sigmas}

    @property
    def tail_policy(self):
        return None

    @property
    def tail_support(self) -> str:
        return "full"

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        w = self.weights[:, :, None]
        mus = self.mus[:, :, None]
        sigmas = self.sigmas[:, :, None]
        cdfs = _stats.norm.cdf(x_arr[None, None, :], loc=mus, scale=sigmas)
        out = (w * cdfs).sum(axis=1)
        return out[:, 0] if scalar else out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] != self.ids.shape[0]:
            raise ValueError(
                f"cdf_at: y has {y_arr.shape[0]} rows, dist has {self.ids.shape[0]}"
            )
        cdfs = _stats.norm.cdf(y_arr[:, None], loc=self.mus, scale=self.sigmas)
        return (self.weights * cdfs).sum(axis=1)

    def cdf_at_grid(self, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError(f"cdf_at_grid: y must be 2-D (N, M); got shape {y_arr.shape}")
        if y_arr.shape[0] != self.ids.shape[0]:
            raise ValueError(
                f"cdf_at_grid: y has {y_arr.shape[0]} rows, dist has {self.ids.shape[0]}"
            )
        nan_mask = np.isnan(y_arr)
        y_safe = np.where(nan_mask, 0.0, y_arr)
        w = self.weights[:, :, None]
        mus = self.mus[:, :, None]
        sigmas = self.sigmas[:, :, None]
        cdfs = _stats.norm.cdf(y_safe[:, None, :], loc=mus, scale=sigmas)
        out = (w * cdfs).sum(axis=1)
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        # Numeric inverse via vectorised per-row bisection.
        w = self.weights
        mus = self.mus
        sigmas = self.sigmas
        N = w.shape[0]
        lo_full = (mus - 8.0 * sigmas).min(axis=1)
        hi_full = (mus + 8.0 * sigmas).max(axis=1)

        def _row_cdf(x_per_row):
            z = (x_per_row[:, None] - mus) / sigmas
            return (w * _stats.norm.cdf(z)).sum(axis=1)

        out = np.empty((N, tau_arr.shape[0]))
        for j, t in enumerate(tau_arr):
            if t <= 0:
                out[:, j] = lo_full
                continue
            if t >= 1:
                out[:, j] = hi_full
                continue
            lo = lo_full.copy()
            hi = hi_full.copy()
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                go_right = _row_cdf(mid) < t
                lo = np.where(go_right, mid, lo)
                hi = np.where(go_right, hi, mid)
            out[:, j] = 0.5 * (lo + hi)
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        w = self.weights[:, :, None]
        mus = self.mus[:, :, None]
        sigmas = self.sigmas[:, :, None]
        pdfs = _stats.norm.pdf(x_arr[None, None, :], loc=mus, scale=sigmas)
        out = (w * pdfs).sum(axis=1)
        return out[:, 0] if scalar else out

    def mean(self):
        return (self.weights * self.mus).sum(axis=1)

    def variance(self):
        mean = (self.weights * self.mus).sum(axis=1)
        ex2 = (self.weights * (self.mus ** 2 + self.sigmas ** 2)).sum(axis=1)
        return ex2 - mean ** 2

    def crps(self, y):
        from bracketlearn.score import crps_mixture_normal
        return crps_mixture_normal(self, y)

    def log_score(self, y):
        from bracketlearn.score import log_score_mixture_normal
        return log_score_mixture_normal(self, y)

    def to_point(self, *, how: str = "mean"):
        if how not in ("mean", "median", "mode"):
            raise ValueError(f"how={how!r} not in 'mean'/'median'/'mode'")
        if how == "mean":
            return (self.weights * self.mus).sum(axis=1)
        if how == "mode":
            best = np.argmax(self.weights, axis=1)
            return self.mus[np.arange(self.mus.shape[0]), best]
        # median: numerical CDF inversion
        return _quantile_via_brentq(self, 0.5)

    @classmethod
    def stitch(cls, folds, *, timestamps, provenance):
        all_rows = np.concatenate([rows for rows, _ in folds])
        order = np.argsort(all_rows, kind="stable")
        ids_sorted = all_rows[order]
        ts_sorted = timestamps[all_rows][order]
        weights = np.concatenate([d.weights for _, d in folds], axis=0)[order]
        mus = np.concatenate([d.mus for _, d in folds], axis=0)[order]
        sigmas = np.concatenate([d.sigmas for _, d in folds], axis=0)[order]
        return cls.from_arrays(
            weights=weights, mus=mus, sigmas=sigmas,
            ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
        )
