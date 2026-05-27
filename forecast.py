"""Data objects: PointForecast, DistributionForecast hierarchy, ContractForecast.

All frozen dataclasses. ndarrays inside are set read-only in __post_init__
to make immutability real (frozen=True alone only freezes attribute binding,
not buffer contents).

v0.3.0 refactor (Session 1)
---------------------------
``DistributionForecast`` is now an ``abc.ABC`` base with concrete subclasses
per backing:

    DistributionForecast (abstract)
    ├── NormalForecast
    ├── StudentTForecast
    ├── MixtureNormalForecast
    ├── QuantileForecast
    └── BracketForecast

Each subclass owns its storage (typed, no optional fields) and its math
(no if/elif/else on backing). The ``Backing`` and ``ParametricFamily``
enums survive this session as compat shims exposed via ``@property`` on
each subclass — so existing consumers (``score.py``, ``pipeline.py``,
``lift.py``, ``restrict.py``, tests) keep working unchanged. They will
be retired in a later session that switches consumers to isinstance
dispatch.

``DistributionForecast.from_*`` classmethods are preserved as thin
construction shims that route to the correct subclass.

Per-row ``BracketForecast`` storage and the ``integrate()`` lift land in
Session 2 — Session 1 only relocates code.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from scipy import stats as _stats

# ---------------------------------------------------------------------------
# Tail extrapolation policy (§7).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailRule:
    """One side of a tail policy."""

    kind: Literal["clip"]
    params: dict

    @staticmethod
    def clip() -> TailRule:
        return TailRule(kind="clip", params={})


@dataclass(frozen=True)
class TailPolicy:
    left: TailRule
    right: TailRule

    @classmethod
    def same(cls, rule: TailRule) -> TailPolicy:
        return cls(left=rule, right=rule)


class TailPolicyError(ValueError):
    """Raised when a tail policy is incompatible with the backing."""


# ---------------------------------------------------------------------------
# Shared helpers (unchanged from v0.2).
# ---------------------------------------------------------------------------


def normalize_bracket_probs(
    raw: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    """Normalise raw per-bracket weights into a valid distribution. See v0.2 docstring."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim not in (1, 2):
        raise ValueError(
            f"{source}: normalize_bracket_probs expects 1-D or 2-D "
            f"input; got shape {raw.shape}."
        )
    if np.any(raw < 0):
        raise ValueError(
            f"{source}: normalize_bracket_probs received negative "
            f"weights. Refusing to clip silently — upstream produced "
            f"invalid data."
        )
    if raw.ndim == 1:
        s = float(raw.sum())
        if s <= 0:
            raise ValueError(
                f"{source}: normalize_bracket_probs got total weight "
                f"{s:.6g} ≤ 0 across K={raw.shape[0]} brackets. Refusing "
                f"to fabricate a uniform distribution."
            )
        return raw / s
    row_sum = raw.sum(axis=1, keepdims=True)
    if np.any(row_sum.ravel() <= 0):
        bad = np.where(row_sum.ravel() <= 0)[0]
        raise ValueError(
            f"{source}: normalize_bracket_probs got {bad.size} row(s) "
            f"with total weight ≤ 0. First offending row indices: "
            f"{bad[:5].tolist()}."
        )
    return raw / row_sum


def bracket_probs_from_cdf_at_edges(
    cdf_at_edges: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    probs = np.diff(cdf_at_edges, axis=1)
    probs = np.clip(probs, 0.0, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    if np.any(row_sum.ravel() <= 0):
        n_bad = int((row_sum.ravel() <= 0).sum())
        raise ValueError(
            f"{source}: {n_bad}/{probs.shape[0]} rows produced zero total "
            f"bracket mass. Refusing to substitute a uniform distribution."
        )
    return probs / row_sum


# ---------------------------------------------------------------------------
# ProvenanceMeta — audit/reproducibility schema (§5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceMeta:
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

    @classmethod
    def placeholder(
        cls,
        forecaster_name: str,
        *,
        sigma_source: Literal["native", "lifted", "none"] = "native",
        random_seed: int | None = None,
    ) -> ProvenanceMeta:
        now = datetime.now()
        return cls(
            forecaster_name=forecaster_name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), now),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=random_seed,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=now,
            sigma_source=sigma_source,
        )


# ---------------------------------------------------------------------------
# PointForecast (§5.1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointForecast:
    mu: np.ndarray
    ids: np.ndarray
    timestamps: np.ndarray
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
# Backing + ParametricFamily enums — compat shims, see module docstring.
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


# ---------------------------------------------------------------------------
# DistributionForecast — abstract base.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionForecast(abc.ABC):
    """Abstract probabilistic-forecast object. One row per market/event.

    Concrete subclasses: ``NormalForecast``, ``StudentTForecast``,
    ``MixtureNormalForecast``, ``QuantileForecast``, ``BracketForecast``.
    Construct via subclass directly or via the ``from_*`` classmethods on
    this base (which route to the correct subclass).
    """

    ids: np.ndarray
    timestamps: np.ndarray
    provenance: ProvenanceMeta

    # ---------- compat: backing/family discriminator (subclasses override) ----------

    @property
    def backing(self) -> Backing:
        raise NotImplementedError

    @property
    def family(self) -> ParametricFamily | None:
        return None

    # ---------- abstract math ----------

    @abc.abstractmethod
    def cdf(self, x: np.ndarray | float) -> np.ndarray: ...

    @abc.abstractmethod
    def cdf_at(self, y: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def cdf_at_grid(self, y: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def ppf(self, tau: np.ndarray | float) -> np.ndarray: ...

    @abc.abstractmethod
    def pdf(
        self,
        x: np.ndarray | float,
        *,
        density_method: str | None = None,
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def mean(self) -> np.ndarray: ...

    @abc.abstractmethod
    def variance(self) -> np.ndarray: ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__}.sample is not implemented")

    # ---------- v0.2 construction shims (route to subclass) ----------

    @classmethod
    def from_normal(
        cls,
        mu: np.ndarray,
        sigma: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> NormalForecast:
        return NormalForecast.from_arrays(
            mu=mu, sigma=sigma, ids=ids, timestamps=timestamps, provenance=provenance,
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
    ) -> StudentTForecast:
        return StudentTForecast.from_arrays(
            mu=mu, sigma=sigma, df=df,
            ids=ids, timestamps=timestamps, provenance=provenance,
        )

    @classmethod
    def from_mixture_normal(
        cls,
        weights: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> MixtureNormalForecast:
        return MixtureNormalForecast.from_arrays(
            weights=weights, mus=mus, sigmas=sigmas,
            ids=ids, timestamps=timestamps, provenance=provenance,
        )

    @classmethod
    def from_quantiles(
        cls,
        taus: np.ndarray,
        qvals: np.ndarray,
        *,
        tail_policy: TailPolicy,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> QuantileForecast:
        return QuantileForecast.from_arrays(
            taus=taus, qvals=qvals, tail_policy=tail_policy,
            ids=ids, timestamps=timestamps, provenance=provenance,
        )

    @classmethod
    def from_brackets(
        cls,
        edges: np.ndarray,
        probs: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> BracketForecast:
        return BracketForecast.from_arrays(
            edges=edges, probs=probs,
            ids=ids, timestamps=timestamps, provenance=provenance,
        )

    # ---------- v0.2 storage compat (subclasses populate) ----------
    #
    # Consumers in score.py / pipeline.py / lift.py / restrict.py and tests
    # still read ``dist.params["mu"]``, ``dist.taus``, ``dist.qvals``,
    # ``dist.edges``, ``dist.probs``, ``dist.tail_policy``. Subclasses keep
    # those attributes so the dispatch tables don't need to change in this
    # session. Removal is a follow-up.


# ---------------------------------------------------------------------------
# Tail-policy helper (unchanged).
# ---------------------------------------------------------------------------


def _resolve_tail_kinds(tail_policy) -> tuple[str, str]:
    if tail_policy is None:
        raise NotImplementedError("quantile-backed cdf requires a TailPolicy")
    return tail_policy.left.kind, tail_policy.right.kind


# ---------------------------------------------------------------------------
# NormalForecast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalForecast(DistributionForecast):
    mu: np.ndarray              # (N,)
    sigma: np.ndarray           # (N,)

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

    # math.
    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        out = _stats.norm.cdf(x_arr[None, :], loc=self.mu[:, None], scale=self.sigma[:, None])
        return out[:, 0] if scalar else out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] != self.ids.shape[0]:
            raise ValueError(
                f"cdf_at: y has {y_arr.shape[0]} rows, dist has {self.ids.shape[0]}"
            )
        return _stats.norm.cdf(y_arr, loc=self.mu, scale=self.sigma)

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
        out = _stats.norm.cdf(y_safe, loc=self.mu[:, None], scale=self.sigma[:, None])
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        out = _stats.norm.ppf(tau_arr[None, :], loc=self.mu[:, None], scale=self.sigma[:, None])
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        out = _stats.norm.pdf(x_arr[None, :], loc=self.mu[:, None], scale=self.sigma[:, None])
        return out[:, 0] if scalar else out

    def mean(self):
        return self.mu.copy()

    def variance(self):
        return self.sigma ** 2

    def sample(self, n, rng):
        mu = self.mu[:, None]
        sigma = self.sigma[:, None]
        return rng.normal(loc=mu, scale=sigma, size=(mu.shape[0], n))


# ---------------------------------------------------------------------------
# StudentTForecast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentTForecast(DistributionForecast):
    mu: np.ndarray
    sigma: np.ndarray
    df: np.ndarray

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

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        out = _stats.t.cdf(
            x_arr[None, :], df=self.df[:, None],
            loc=self.mu[:, None], scale=self.sigma[:, None],
        )
        return out[:, 0] if scalar else out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] != self.ids.shape[0]:
            raise ValueError(
                f"cdf_at: y has {y_arr.shape[0]} rows, dist has {self.ids.shape[0]}"
            )
        return _stats.t.cdf(y_arr, df=self.df, loc=self.mu, scale=self.sigma)

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
        out = _stats.t.cdf(
            y_safe, df=self.df[:, None],
            loc=self.mu[:, None], scale=self.sigma[:, None],
        )
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        out = _stats.t.ppf(
            tau_arr[None, :], df=self.df[:, None],
            loc=self.mu[:, None], scale=self.sigma[:, None],
        )
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        out = _stats.t.pdf(
            x_arr[None, :], df=self.df[:, None],
            loc=self.mu[:, None], scale=self.sigma[:, None],
        )
        return out[:, 0] if scalar else out

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


# ---------------------------------------------------------------------------
# QuantileForecast.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantileForecast(DistributionForecast):
    taus: np.ndarray             # (Q,)
    qvals: np.ndarray            # (N, Q)
    tail_policy: TailPolicy

    @classmethod
    def from_arrays(
        cls,
        *,
        taus: np.ndarray,
        qvals: np.ndarray,
        tail_policy: TailPolicy,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> QuantileForecast:
        taus = np.asarray(taus, dtype=float)
        qvals = np.asarray(qvals, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
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
            ids=ids, timestamps=timestamps, provenance=provenance,
            taus=taus, qvals=qvals, tail_policy=tail_policy,
        )

    @property
    def backing(self) -> Backing:
        return Backing.QUANTILE

    @property
    def family(self) -> ParametricFamily | None:
        return None

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"taus": self.taus, "qvals": self.qvals}

    @property
    def tail_support(self) -> str:
        return "finite-quantile"

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        taus = self.taus
        qvals = self.qvals
        N, _ = qvals.shape
        out = np.zeros((N, x_arr.shape[0]))
        left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
        for i in range(N):
            interp = np.interp(
                x_arr, qvals[i], taus,
                left=0.0 if left_rule == "clip" else np.nan,
                right=1.0 if right_rule == "clip" else np.nan,
            )
            if np.any(np.isnan(interp)):
                raise NotImplementedError(
                    f"tail policy {left_rule!r}/{right_rule!r} not implemented yet"
                )
            out[i] = interp
        return out[:, 0] if scalar else out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        N_rows = self.ids.shape[0]
        if y_arr.shape[0] != N_rows:
            raise ValueError(
                f"cdf_at: y has {y_arr.shape[0]} rows, dist has {N_rows}"
            )
        taus = self.taus
        qvals = self.qvals
        left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
        out = np.empty(N_rows, dtype=float)
        for i in range(N_rows):
            v = np.interp(
                y_arr[i:i + 1], qvals[i], taus,
                left=0.0 if left_rule == "clip" else np.nan,
                right=1.0 if right_rule == "clip" else np.nan,
            )
            if np.isnan(v).any():
                raise NotImplementedError(
                    f"tail policy {left_rule!r}/{right_rule!r} not implemented yet"
                )
            out[i] = v[0]
        return out

    def cdf_at_grid(self, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError(f"cdf_at_grid: y must be 2-D (N, M); got shape {y_arr.shape}")
        N_rows = self.ids.shape[0]
        if y_arr.shape[0] != N_rows:
            raise ValueError(
                f"cdf_at_grid: y has {y_arr.shape[0]} rows, dist has {N_rows}"
            )
        M = y_arr.shape[1]
        nan_mask = np.isnan(y_arr)
        y_safe = np.where(nan_mask, 0.0, y_arr)
        taus = self.taus
        qvals = self.qvals
        left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
        out = np.empty((N_rows, M), dtype=float)
        for i in range(N_rows):
            v = np.interp(
                y_safe[i], qvals[i], taus,
                left=0.0 if left_rule == "clip" else np.nan,
                right=1.0 if right_rule == "clip" else np.nan,
            )
            if np.isnan(v).any():
                raise NotImplementedError(
                    f"tail policy {left_rule!r}/{right_rule!r} not implemented yet"
                )
            out[i] = v
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
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
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        raise NotImplementedError(
            "pdf on quantile backing not implemented in v0.2; use bracket integration."
        )

    def mean(self):
        taus = self.taus
        qvals = self.qvals
        d = np.empty_like(taus)
        d[0] = (taus[1] - 0.0) * 0.5
        d[-1] = (1.0 - taus[-2]) * 0.5
        d[1:-1] = (taus[2:] - taus[:-2]) * 0.5
        return (qvals * d[None, :]).sum(axis=1)

    def variance(self):
        raise NotImplementedError("variance on quantile backing not implemented in v0.2")


# ---------------------------------------------------------------------------
# BracketForecast (1-D edges still — per-row storage lands in Session 2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BracketForecast(DistributionForecast):
    edges: np.ndarray            # (B+1,)
    probs: np.ndarray            # (N, B)

    @classmethod
    def from_arrays(
        cls,
        *,
        edges: np.ndarray,
        probs: np.ndarray,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> BracketForecast:
        edges = np.asarray(edges, dtype=float)
        probs = np.asarray(probs, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
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
            ids=ids, timestamps=timestamps, provenance=provenance,
            edges=edges, probs=probs,
        )

    @property
    def backing(self) -> Backing:
        return Backing.BRACKET

    @property
    def family(self) -> ParametricFamily | None:
        return None

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"edges": self.edges, "probs": self.probs}

    @property
    def tail_policy(self):
        return None

    @property
    def tail_support(self) -> str:
        return "bounded"

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        edges = self.edges
        probs = self.probs
        N_rows, B = probs.shape
        cum = np.concatenate(
            [np.zeros((N_rows, 1)), np.cumsum(probs, axis=1)], axis=1
        )
        k = np.searchsorted(edges, x_arr, side="right") - 1
        below = x_arr <= edges[0]
        above = x_arr >= edges[-1]
        k_clipped = np.clip(k, 0, B - 1)
        widths = edges[k_clipped + 1] - edges[k_clipped]
        safe_w = np.where(widths > 0, widths, 1.0)
        frac = np.where(widths > 0, (x_arr - edges[k_clipped]) / safe_w, 0.0)
        out = cum[:, k_clipped] + frac[None, :] * probs[:, k_clipped]
        if below.any():
            out[:, below] = 0.0
        if above.any():
            out[:, above] = 1.0
        return out[:, 0] if scalar else out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        N_rows = self.ids.shape[0]
        if y_arr.shape[0] != N_rows:
            raise ValueError(
                f"cdf_at: y has {y_arr.shape[0]} rows, dist has {N_rows}"
            )
        edges = self.edges
        probs = self.probs
        B = probs.shape[1]
        cum = np.concatenate(
            [np.zeros((N_rows, 1)), np.cumsum(probs, axis=1)], axis=1
        )
        k = np.searchsorted(edges, y_arr, side="right") - 1
        below = y_arr <= edges[0]
        above = y_arr >= edges[-1]
        k_clipped = np.clip(k, 0, B - 1)
        rows = np.arange(N_rows)
        widths = edges[k_clipped + 1] - edges[k_clipped]
        safe_w = np.where(widths > 0, widths, 1.0)
        frac = np.where(widths > 0, (y_arr - edges[k_clipped]) / safe_w, 0.0)
        out = cum[rows, k_clipped] + frac * probs[rows, k_clipped]
        out = np.where(below, 0.0, out)
        out = np.where(above, 1.0, out)
        return out

    def cdf_at_grid(self, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError(f"cdf_at_grid: y must be 2-D (N, M); got shape {y_arr.shape}")
        N_rows = self.ids.shape[0]
        if y_arr.shape[0] != N_rows:
            raise ValueError(
                f"cdf_at_grid: y has {y_arr.shape[0]} rows, dist has {N_rows}"
            )
        M = y_arr.shape[1]
        nan_mask = np.isnan(y_arr)
        y_safe = np.where(nan_mask, 0.0, y_arr)
        edges = self.edges
        probs = self.probs
        B = probs.shape[1]
        cum = np.concatenate(
            [np.zeros((N_rows, 1)), np.cumsum(probs, axis=1)], axis=1
        )
        k = np.searchsorted(edges, y_safe.ravel(), side="right").reshape(N_rows, M) - 1
        below = y_safe <= edges[0]
        above = y_safe >= edges[-1]
        k_clipped = np.clip(k, 0, B - 1)
        widths = edges[k_clipped + 1] - edges[k_clipped]
        safe_w = np.where(widths > 0, widths, 1.0)
        frac = np.where(widths > 0, (y_safe - edges[k_clipped]) / safe_w, 0.0)
        rows_idx = np.arange(N_rows)[:, None]
        out = cum[rows_idx, k_clipped] + frac * probs[rows_idx, k_clipped]
        out = np.where(below, 0.0, out)
        out = np.where(above, 1.0, out)
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        edges = self.edges
        probs = self.probs
        N, B = probs.shape
        cum = np.concatenate(
            [np.zeros((N, 1)), np.cumsum(probs, axis=1)], axis=1
        )
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
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        if density_method != "step":
            raise ValueError(
                "pdf on bracket backing requires density_method='step' "
                "(no silent KDE bandwidth)"
            )
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        widths = np.diff(self.edges)
        density = self.probs / widths[None, :]
        B = density.shape[1]
        k = np.searchsorted(self.edges, x_arr, side="right") - 1
        inside = (x_arr >= self.edges[0]) & (x_arr < self.edges[-1])
        k_clipped = np.clip(k, 0, B - 1)
        out = density[:, k_clipped]
        if (~inside).any():
            out[:, ~inside] = 0.0
        return out[:, 0] if scalar else out

    def mean(self):
        mids = 0.5 * (self.edges[1:] + self.edges[:-1])
        return (self.probs * mids[None, :]).sum(axis=1)

    def variance(self):
        mids = 0.5 * (self.edges[1:] + self.edges[:-1])
        m = (self.probs * mids[None, :]).sum(axis=1)
        ex2 = (self.probs * (mids ** 2)[None, :]).sum(axis=1)
        return ex2 - m ** 2


# ---------------------------------------------------------------------------
# ContractForecast (§5.3) — output of ContractAdapter.price(). Unchanged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractSpec:
    """Typed serialisable spec for an adapter."""

    kind: str
    schema_version: int = 1


@dataclass(frozen=True)
class ContractForecast:
    contract_ids: np.ndarray
    entity_ids: np.ndarray
    timestamps: np.ndarray
    fair_price: np.ndarray
    group_id: np.ndarray
    contract_spec: ContractSpec
    provenance: ProvenanceMeta

    def __post_init__(self) -> None:
        for arr in (self.contract_ids, self.entity_ids, self.timestamps,
                    self.fair_price, self.group_id):
            arr.setflags(write=False)
