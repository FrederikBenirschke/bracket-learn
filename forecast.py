"""Data objects: PointForecast, DistributionForecast, ContractForecast, ProvenanceMeta.

All frozen dataclasses. ndarrays inside are set read-only in __post_init__
to make immutability real (frozen=True alone only freezes attribute binding,
not buffer contents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats as _stats

if TYPE_CHECKING:
    from bracketlearn.tail import TailPolicy


# ---------------------------------------------------------------------------
# ProvenanceMeta — audit/reproducibility schema (§5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceMeta:
    """Identifies the exact (code, data, seed, fold) that produced a forecast.

    Tested invariant: two forecasts with identical
    (forecaster_name, forecaster_version, fit_window, fold_idx,
    calibration_set_hash, feature_matrix_hash, random_seed) are bit-identical.
    """

    forecaster_name: str
    forecaster_version: str
    fit_window: tuple[datetime, datetime]
    fold_idx: int | Literal["prequential"] | None
    calibration_set_hash: str | None
    random_seed: int | None
    code_sha: str
    feature_matrix_hash: str
    created_at: datetime
    sigma_source: Literal["native", "lifted", "none"] = "none"
    conversion_chain: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PointForecast (§5.1) — no σ. Native-σ forecasters return DistributionForecast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointForecast:
    mu: np.ndarray              # (N,)
    ids: np.ndarray             # (N,)
    timestamps: np.ndarray      # (N,)
    provenance: ProvenanceMeta

    def __post_init__(self) -> None:
        for arr in (self.mu, self.ids, self.timestamps):
            arr.setflags(write=False)
        if not (len(self.mu) == len(self.ids) == len(self.timestamps)):
            raise ValueError(
                f"length mismatch: mu={len(self.mu)} ids={len(self.ids)} "
                f"timestamps={len(self.timestamps)}"
            )


# ---------------------------------------------------------------------------
# DistributionForecast (§5.2) — four backings, lazy conversions, tail policy.
# ---------------------------------------------------------------------------


class Backing(StrEnum):
    PARAMETRIC = "parametric"
    QUANTILE = "quantile"
    EMPIRICAL = "empirical"
    BRACKET = "bracket"


class ParametricFamily(StrEnum):
    NORMAL = "normal"
    STUDENT_T = "student_t"
    MIXTURE_NORMAL = "mixture_normal"


@dataclass(frozen=True)
class DistributionForecast:
    """The load-bearing object. One of four backings; conversions are lazy.

    Construction is via the from_* classmethods so each backing's storage
    invariants can be enforced. Do not instantiate directly.
    """

    backing: Backing
    ids: np.ndarray             # (N,)
    timestamps: np.ndarray      # (N,)
    provenance: ProvenanceMeta

    # Backing-specific storage (only one set populated per instance).
    # parametric:
    family: ParametricFamily | None = None
    params: dict[str, np.ndarray] | None = None     # e.g. {"mu": (N,), "sigma": (N,)}
    # quantile:
    taus: np.ndarray | None = None                  # (Q,)
    qvals: np.ndarray | None = None                 # (N, Q)
    # empirical:
    members: np.ndarray | None = None               # (N, K)
    # bracket:
    edges: np.ndarray | None = None                 # (B+1,)
    probs: np.ndarray | None = None                 # (N, B)

    # Tail policy — required for finite-support backings; None for full-support
    # parametric (normal, student_t with infinite support).
    tail_policy: TailPolicy | None = None
    tail_support: Literal["full", "bounded", "finite-quantile"] = "full"

    # ------------------------------------------------------------------ ctor

    @classmethod
    def from_normal(
        cls,
        mu: np.ndarray,
        sigma: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> DistributionForecast:
        """Native parametric normal. No tail policy needed."""
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        if mu.shape != sigma.shape or mu.shape != ids.shape:
            raise ValueError(
                f"shape mismatch: mu={mu.shape} sigma={sigma.shape} ids={ids.shape}"
            )
        if np.any(sigma <= 0):
            raise ValueError("sigma must be strictly positive")
        return cls(
            backing=Backing.PARAMETRIC,
            family=ParametricFamily.NORMAL,
            params={"mu": mu, "sigma": sigma},
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=provenance,
            tail_support="full",
        )

    @classmethod
    def from_student_t(
        cls,
        mu: np.ndarray,
        sigma: np.ndarray,
        df: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> DistributionForecast:
        """Native parametric Student-t. sigma is the scale parameter; the
        marginal variance is sigma² · df / (df − 2) and requires df > 2.

        df may be a scalar (broadcast) or per-row array.
        """
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        df = np.asarray(df, dtype=float)
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
            backing=Backing.PARAMETRIC,
            family=ParametricFamily.STUDENT_T,
            params={"mu": mu, "sigma": sigma, "df": df},
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=provenance,
            tail_support="full",
        )

    @classmethod
    def from_mixture_normal(
        cls,
        weights: np.ndarray,    # (N, K)
        mus: np.ndarray,        # (N, K)
        sigmas: np.ndarray,     # (N, K)
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> DistributionForecast:
        """Per-row mixture of K Gaussians. Components with zero weight are
        permitted (e.g. missing vendor in mixture_normals); the row's
        weights must still sum to 1 (renormalise upstream)."""
        weights = np.asarray(weights, dtype=float)
        mus = np.asarray(mus, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
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
            backing=Backing.PARAMETRIC,
            family=ParametricFamily.MIXTURE_NORMAL,
            params={"weights": weights, "mus": mus, "sigmas": sigmas},
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=provenance,
            tail_support="full",
        )

    @classmethod
    def from_quantiles(
        cls,
        taus: np.ndarray,
        qvals: np.ndarray,
        *,
        tail_policy: TailPolicy,      # REQUIRED — no default
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> DistributionForecast:
        """Quantile-backed. tail_policy is required — silent linear
        extrapolation would mask upstream config bugs.

        taus: (Q,) sorted ascending in (0, 1).
        qvals: (N, Q) per-row quantile values, monotone non-decreasing along Q.
        """
        taus = np.asarray(taus, dtype=float)
        qvals = np.asarray(qvals, dtype=float)
        if taus.ndim != 1 or taus.shape[0] < 2:
            raise ValueError(f"taus must be 1-D with ≥2 entries; got {taus.shape}")
        if qvals.ndim != 2 or qvals.shape[1] != taus.shape[0]:
            raise ValueError(
                f"qvals shape {qvals.shape} incompatible with taus {taus.shape}"
            )
        if qvals.shape[0] != ids.shape[0]:
            raise ValueError(f"N mismatch: qvals N={qvals.shape[0]} ids N={ids.shape[0]}")
        if np.any(np.diff(taus) <= 0):
            raise ValueError("taus must be strictly increasing")
        if np.any((taus <= 0) | (taus >= 1)):
            raise ValueError("taus must lie strictly in (0, 1)")
        # Quantiles must be monotone non-decreasing along Q. Small numerical
        # crossings (≲ 1e-9 of the row range) are repaired silently; anything
        # bigger is a real bug upstream and raises.
        diffs = np.diff(qvals, axis=1)
        if np.any(diffs < 0):
            row_range = qvals.max(axis=1, keepdims=True) - qvals.min(axis=1, keepdims=True)
            tol = np.maximum(1e-9 * row_range, 1e-12)
            worst = float(diffs.min())
            if np.any(diffs < -tol):
                raise ValueError(
                    f"qvals must be monotone non-decreasing along Q; "
                    f"worst crossing = {worst:.6g} (use isotonic-repair upstream)"
                )
            qvals = np.maximum.accumulate(qvals, axis=1)
        return cls(
            backing=Backing.QUANTILE,
            taus=taus,
            qvals=qvals,
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=provenance,
            tail_policy=tail_policy,
            tail_support="finite-quantile",
        )

    @classmethod
    def from_empirical(
        cls,
        members: np.ndarray,            # (N, K)
        *,
        tail_policy: TailPolicy,      # REQUIRED
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> DistributionForecast:
        raise NotImplementedError(
            "DistributionForecast.from_empirical — empirical backing not "
            "yet implemented. Use from_quantiles or from_brackets."
        )

    @classmethod
    def from_brackets(
        cls,
        edges: np.ndarray,              # (B+1,)
        probs: np.ndarray,              # (N, B)
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> DistributionForecast:
        """Bracket-backed. Bounded by construction; no tail policy."""
        edges = np.asarray(edges, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if edges.ndim != 1 or edges.shape[0] < 2:
            raise ValueError(f"edges must be 1-D with at least 2 entries; got {edges.shape}")
        if probs.ndim != 2 or probs.shape[1] != edges.shape[0] - 1:
            raise ValueError(
                f"probs shape {probs.shape} incompatible with edges {edges.shape}"
            )
        if probs.shape[0] != ids.shape[0]:
            raise ValueError(f"N mismatch: probs N={probs.shape[0]} ids N={ids.shape[0]}")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be monotone strictly increasing")
        if np.any(probs < 0):
            raise ValueError("probs must be nonnegative")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("probs must sum to 1 per row")
        return cls(
            backing=Backing.BRACKET,
            edges=edges,
            probs=probs,
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=provenance,
            tail_support="bounded",
        )

    # ----------------------------------------------------------- accessors

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        """P(X ≤ x). Scalar x → (N,); array x → (N, len(x))."""
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)

        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            out = _stats.norm.cdf(x_arr[None, :], loc=mu, scale=sigma)
        elif self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.STUDENT_T:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            df = self.params["df"][:, None]
            out = _stats.t.cdf(x_arr[None, :], df=df, loc=mu, scale=sigma)
        elif self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.MIXTURE_NORMAL:
            # weights/mus/sigmas: (N, K). Compute Σ_k w_k Φ((x - μ_k)/σ_k).
            w = self.params["weights"][:, :, None]      # (N, K, 1)
            mus = self.params["mus"][:, :, None]
            sigmas = self.params["sigmas"][:, :, None]
            cdfs = _stats.norm.cdf(x_arr[None, None, :], loc=mus, scale=sigmas)  # (N, K, M)
            out = (w * cdfs).sum(axis=1)                 # (N, M)
        elif self.backing == Backing.BRACKET:
            # P(X ≤ x) = sum of probs in fully-included bins + partial last bin.
            # Assume uniform within each bin (default interpolation).
            edges = self.edges
            probs = self.probs       # (N, B)
            B = probs.shape[1]
            out = np.zeros((probs.shape[0], x_arr.shape[0]))
            cum = np.concatenate(
                [np.zeros((probs.shape[0], 1)), np.cumsum(probs, axis=1)], axis=1
            )  # (N, B+1)
            for j, xv in enumerate(x_arr):
                if xv <= edges[0]:
                    out[:, j] = 0.0
                elif xv >= edges[-1]:
                    out[:, j] = 1.0
                else:
                    # find bin index k such that edges[k] <= xv < edges[k+1]
                    k = int(np.searchsorted(edges, xv, side="right") - 1)
                    k = max(0, min(k, B - 1))
                    width = edges[k + 1] - edges[k]
                    frac = (xv - edges[k]) / width if width > 0 else 0.0
                    out[:, j] = cum[:, k] + frac * probs[:, k]
        elif self.backing == Backing.QUANTILE:
            # Piecewise-linear CDF through (qvals, taus). Outside qvals[0]/qvals[-1]
            # apply tail policy. Tier 2: only "clip" implemented (0 / 1 outside).
            taus = self.taus
            qvals = self.qvals                 # (N, Q)
            N, Q = qvals.shape
            out = np.zeros((N, x_arr.shape[0]))
            # Per-row interp: for each row i, x → F(x) by interpolating taus on qvals[i].
            # Use np.interp row-by-row (no good vectorized version for non-shared knots).
            left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
            for i in range(N):
                xq = qvals[i]
                # ensure monotone increasing (defensive — caller should ensure)
                if np.any(np.diff(xq) < 0):
                    xq = np.maximum.accumulate(xq)
                interp = np.interp(x_arr, xq, taus,
                                   left=0.0 if left_rule == "clip" else np.nan,
                                   right=1.0 if right_rule == "clip" else np.nan)
                if np.any(np.isnan(interp)):
                    raise NotImplementedError(
                        f"tail policy {left_rule!r}/{right_rule!r} not implemented yet"
                    )
                out[i] = interp
        else:
            raise NotImplementedError(
                f"cdf not implemented for backing={self.backing} family={self.family}"
            )

        return out[:, 0] if scalar else out

    def ppf(self, tau: np.ndarray | float) -> np.ndarray:
        """Quantile function. Scalar tau → (N,); array tau → (N, len(tau))."""
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")

        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            out = _stats.norm.ppf(tau_arr[None, :], loc=mu, scale=sigma)
        elif self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.STUDENT_T:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            df = self.params["df"][:, None]
            out = _stats.t.ppf(tau_arr[None, :], df=df, loc=mu, scale=sigma)
        elif self.backing == Backing.QUANTILE:
            # Per-row piecewise-linear interp: tau → q. Outside [taus[0], taus[-1]]
            # we clip to the outermost stored quantile (matches TailRule.clip
            # semantics; any non-clip policy raises rather than silently
            # extrapolating).
            taus = self.taus
            qvals = self.qvals
            N, _ = qvals.shape
            out = np.empty((N, tau_arr.shape[0]))
            left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
            if left_rule != "clip" or right_rule != "clip":
                raise NotImplementedError(
                    f"ppf on quantile backing only supports clip tails; "
                    f"got left={left_rule!r} right={right_rule!r}"
                )
            for i in range(N):
                out[i] = np.interp(tau_arr, taus, qvals[i])
        elif self.backing == Backing.BRACKET:
            # Invert piecewise-linear CDF: cum-probs at edge k is sum of
            # probs[0..k-1]. For tau in [cum[k], cum[k+1]] we land in bin k
            # and linearly interpolate within [edges[k], edges[k+1]].
            edges = self.edges
            probs = self.probs                   # (N, B)
            N, B = probs.shape
            cum = np.concatenate(
                [np.zeros((N, 1)), np.cumsum(probs, axis=1)], axis=1
            )                                    # (N, B+1)
            out = np.empty((N, tau_arr.shape[0]))
            for i in range(N):
                for j, t in enumerate(tau_arr):
                    if t <= 0:
                        out[i, j] = edges[0]
                    elif t >= 1:
                        out[i, j] = edges[-1]
                    else:
                        k = int(np.searchsorted(cum[i], t, side="right") - 1)
                        k = max(0, min(k, B - 1))
                        width_p = cum[i, k + 1] - cum[i, k]
                        if width_p <= 0:
                            out[i, j] = edges[k]
                        else:
                            frac = (t - cum[i, k]) / width_p
                            out[i, j] = edges[k] + frac * (edges[k + 1] - edges[k])
        elif self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.MIXTURE_NORMAL:
            # Numeric inverse via vectorised per-row bisection. Search range:
            # μ ± 8·σ across components — wide enough that mixture CDF is
            # ~0/1 at endpoints.
            w = self.params["weights"]            # (N, K)
            mus = self.params["mus"]              # (N, K)
            sigmas = self.params["sigmas"]        # (N, K)
            N = w.shape[0]
            lo_full = (mus - 8.0 * sigmas).min(axis=1)    # (N,)
            hi_full = (mus + 8.0 * sigmas).max(axis=1)    # (N,)

            def _row_cdf(x_per_row: np.ndarray) -> np.ndarray:
                """Mixture CDF evaluated at one x per row. Returns (N,)."""
                # P(X ≤ x) = Σ_k w_k Φ((x - μ_k) / σ_k), with x broadcast over K.
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
                for _ in range(60):                       # ~1e-18 on 16-decade range
                    mid = 0.5 * (lo + hi)
                    go_right = _row_cdf(mid) < t
                    lo = np.where(go_right, mid, lo)
                    hi = np.where(go_right, hi, mid)
                out[:, j] = 0.5 * (lo + hi)
        else:
            raise NotImplementedError(f"ppf not implemented for backing={self.backing}")

        return out[:, 0] if scalar else out

    def pdf(
        self,
        x: np.ndarray | float,
        *,
        density_method: Literal["step", "kde:scott", "kde:silverman"] | None = None,
    ) -> np.ndarray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)

        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            out = _stats.norm.pdf(x_arr[None, :], loc=mu, scale=sigma)
        elif self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.STUDENT_T:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            df = self.params["df"][:, None]
            out = _stats.t.pdf(x_arr[None, :], df=df, loc=mu, scale=sigma)
        elif self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.MIXTURE_NORMAL:
            w = self.params["weights"][:, :, None]
            mus = self.params["mus"][:, :, None]
            sigmas = self.params["sigmas"][:, :, None]
            pdfs = _stats.norm.pdf(x_arr[None, None, :], loc=mus, scale=sigmas)
            out = (w * pdfs).sum(axis=1)
        elif self.backing == Backing.BRACKET:
            if density_method != "step":
                raise ValueError(
                    "pdf on bracket backing requires density_method='step' "
                    "(no silent KDE bandwidth)"
                )
            # density inside bin k = probs[:, k] / (edges[k+1] - edges[k])
            widths = np.diff(self.edges)         # (B,)
            density = self.probs / widths[None, :]  # (N, B)
            out = np.zeros((self.probs.shape[0], x_arr.shape[0]))
            for j, xv in enumerate(x_arr):
                if xv < self.edges[0] or xv >= self.edges[-1]:
                    out[:, j] = 0.0
                else:
                    k = int(np.searchsorted(self.edges, xv, side="right") - 1)
                    k = max(0, min(k, density.shape[1] - 1))
                    out[:, j] = density[:, k]
        else:
            raise NotImplementedError(f"pdf not implemented for backing={self.backing}")

        return out[:, 0] if scalar else out

    def mean(self) -> np.ndarray:
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            return self.params["mu"].copy()
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.STUDENT_T:
            return self.params["mu"].copy()
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.MIXTURE_NORMAL:
            return (self.params["weights"] * self.params["mus"]).sum(axis=1)
        if self.backing == Backing.BRACKET:
            mids = 0.5 * (self.edges[1:] + self.edges[:-1])    # (B,)
            return (self.probs * mids[None, :]).sum(axis=1)
        if self.backing == Backing.QUANTILE:
            # Trapezoidal: E[X] ≈ Σ ((τ_{k+1}-τ_{k-1})/2) · q_k, with end taus
            # set so the first/last contribute 0 mass beyond the grid (clip).
            taus = self.taus
            qvals = self.qvals
            # central differences with 0 tail mass
            d = np.empty_like(taus)
            d[0] = (taus[1] - 0.0) * 0.5      # approximate left mass at τ_0 from clip(0)
            d[-1] = (1.0 - taus[-2]) * 0.5    # right mass beyond τ_-1 from clip(1)
            d[1:-1] = (taus[2:] - taus[:-2]) * 0.5
            return (qvals * d[None, :]).sum(axis=1)
        raise NotImplementedError(f"mean not implemented for backing={self.backing}")

    def variance(self) -> np.ndarray:
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            return self.params["sigma"] ** 2
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.STUDENT_T:
            # Var = σ² · df / (df − 2); df > 2 is enforced at construction.
            sigma = self.params["sigma"]
            df = self.params["df"]
            return sigma ** 2 * df / (df - 2.0)
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.MIXTURE_NORMAL:
            # E[X²] − E[X]², with E[X²] = Σ w_k (μ_k² + σ_k²).
            w = self.params["weights"]
            mus = self.params["mus"]
            sigmas = self.params["sigmas"]
            mean = (w * mus).sum(axis=1)
            ex2 = (w * (mus ** 2 + sigmas ** 2)).sum(axis=1)
            return ex2 - mean ** 2
        if self.backing == Backing.BRACKET:
            mids = 0.5 * (self.edges[1:] + self.edges[:-1])
            m = (self.probs * mids[None, :]).sum(axis=1)
            ex2 = (self.probs * (mids ** 2)[None, :]).sum(axis=1)
            return ex2 - m ** 2
        raise NotImplementedError(f"variance not implemented for backing={self.backing}")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            return rng.normal(loc=mu, scale=sigma, size=(mu.shape[0], n))
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.STUDENT_T:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            df = self.params["df"][:, None]
            # rng.standard_t(df) does not broadcast df over rows; sample per-row.
            N = mu.shape[0]
            out = np.empty((N, n))
            df_flat = df[:, 0]
            for i in range(N):
                out[i] = rng.standard_t(df_flat[i], size=n)
            return mu + sigma * out
        raise NotImplementedError(f"sample not implemented for backing={self.backing}")

    # NOTE: expected_payoff is intentionally NOT here. Each ContractAdapter
    # owns its own price() per backing (§5/§8). MC lives only in the
    # `Custom` adapter.

    # ----------------------------------------------------------- conversions

    def to_quantiles(self, taus: np.ndarray) -> DistributionForecast:
        """Returns new dist with quantile backing. Records lossy conversion
        into provenance.conversion_chain."""
        raise NotImplementedError("DistributionForecast.to_quantiles — not yet implemented")

    def to_brackets(self, edges: np.ndarray) -> DistributionForecast:
        raise NotImplementedError("DistributionForecast.to_brackets — not yet implemented")

    def to_normal(self) -> DistributionForecast:
        """Moment match. Lossy for fat-tailed or skewed inputs."""
        raise NotImplementedError("DistributionForecast.to_normal — not yet implemented")

    def is_lossless_to(self, target_backing: Backing) -> bool:
        """True iff conversion to target_backing preserves all information."""
        raise NotImplementedError("DistributionForecast.is_lossless_to — not yet implemented")


def _resolve_tail_kinds(tail_policy) -> tuple[str, str]:
    """Return (left_kind, right_kind) for the tail policy.

    None policy is allowed for parametric backings only — quantile callers
    must pass one. Tier-2 only implements 'clip'; other kinds raise.
    """
    if tail_policy is None:
        raise NotImplementedError("quantile-backed cdf requires a TailPolicy")
    return tail_policy.left.kind, tail_policy.right.kind


# ---------------------------------------------------------------------------
# ContractForecast (§5.3) — output of ContractAdapter.price().
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractSpec:
    """Typed serialisable spec for an adapter. Replaces v0.1's dict.

    Subclasses (BracketSpec, BinarySpec, ...) carry kind-specific fields.
    schema_version supports forward-migration of stored parquet/JSON.
    """

    kind: str                       # discriminator: "binary_above", "bracket_ladder", ...
    schema_version: int = 1


@dataclass(frozen=True)
class ContractForecast:
    contract_ids: np.ndarray        # (M,)
    entity_ids: np.ndarray          # (M,) — matches one DistributionForecast row
    timestamps: np.ndarray          # (M,)
    fair_price: np.ndarray          # (M,) — payoff-natural units (see §8.3)
    group_id: np.ndarray            # (M,) — paired/laddered rows share a group
    contract_spec: ContractSpec
    provenance: ProvenanceMeta

    def __post_init__(self) -> None:
        for arr in (self.contract_ids, self.entity_ids, self.timestamps,
                    self.fair_price, self.group_id):
            arr.setflags(write=False)

    def calibrate(
        self,
        method: Literal["platt", "isotonic", "beta"],
        *,
        realized: np.ndarray,
    ) -> ContractForecast:
        """Contract-space recalibration (§8.4). Distinct from
        DistributionForecast-level calibration."""
        raise NotImplementedError("ContractForecast.calibrate — not yet implemented")
