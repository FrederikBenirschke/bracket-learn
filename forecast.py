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

    # ---------- per-row bracket projection ----------

    def integrate(self, edges_per_row) -> BracketForecast:
        """Project this distribution onto a per-row bracket grid.

        ``edges_per_row`` may be:
          - 1-D ``(B+1,)`` shared across all rows,
          - 2-D ``(N, B+1)`` dense per-row grid,
          - sequence of length N with each entry a 1-D edge vector
            (ragged; NaN-padded into a dense (N, B_max+1) array).

        Default implementation: ``cdf_at_grid`` on the dense edges then
        ``np.diff`` along the bin axis. Subclasses may override for a
        faster closed-form path.
        """
        edges_dense = _to_dense_2d(edges_per_row, n_rows=self.ids.shape[0])
        cdf_at_edges = self.cdf_at_grid(edges_dense)
        probs = np.diff(cdf_at_edges, axis=1)
        probs = _clip_tiny_negatives(probs)
        # Re-normalise per row so any cumulative tiny clip doesn't drift
        # the row away from sum-to-1 (BracketForecast.from_arrays enforces
        # sum-to-1 with atol=1e-6).
        row_sum = np.nansum(probs, axis=1, keepdims=True)
        if np.any(row_sum.ravel() <= 0):
            n_bad = int((row_sum.ravel() <= 0).sum())
            raise ValueError(
                f"integrate: {n_bad} row(s) have zero total mass on the "
                f"requested bracket grid. The grid likely lies outside the "
                f"distribution's support."
            )
        # Where probs is NaN (ragged tail), preserve NaN; renormalise
        # finite entries.
        with np.errstate(invalid="ignore"):
            probs = probs / row_sum
        return BracketForecast.from_arrays(
            edges=edges_dense, probs=probs,
            ids=self.ids, timestamps=self.timestamps,
            provenance=self.provenance,
        )

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


def _to_dense_2d(edges_per_row, *, n_rows: int) -> np.ndarray:
    """Normalise heterogeneous edge inputs to a dense (N, B_max+1) array
    with NaN padding for ragged rows.

    Accepts: 1-D shared (B+1,), 2-D dense (N, B+1), or a length-N
    sequence of 1-D arrays.
    """
    if isinstance(edges_per_row, np.ndarray):
        if edges_per_row.ndim == 1:
            if edges_per_row.shape[0] < 2:
                raise ValueError(
                    f"shared edges must have ≥2 entries; got {edges_per_row.shape}"
                )
            return np.broadcast_to(
                edges_per_row[None, :].astype(float),
                (n_rows, edges_per_row.shape[0]),
            ).copy()
        if edges_per_row.ndim == 2:
            if edges_per_row.shape[0] != n_rows:
                raise ValueError(
                    f"edges_per_row N={edges_per_row.shape[0]} != dist N={n_rows}"
                )
            return edges_per_row.astype(float)
        raise ValueError(f"edges_per_row ndarray must be 1-D or 2-D; got shape {edges_per_row.shape}")
    # Sequence path.
    rows = list(edges_per_row)
    if len(rows) != n_rows:
        raise ValueError(
            f"edges_per_row has length {len(rows)}; dist has {n_rows} rows"
        )
    rows_arr = [np.asarray(r, dtype=float) for r in rows]
    if any(r.ndim != 1 or r.shape[0] < 2 for r in rows_arr):
        bad = [i for i, r in enumerate(rows_arr) if r.ndim != 1 or r.shape[0] < 2]
        raise ValueError(
            f"edges_per_row entries must be 1-D with ≥2 entries; bad row(s): {bad[:5]}"
        )
    B_max1 = max(r.shape[0] for r in rows_arr)
    out = np.full((n_rows, B_max1), np.nan, dtype=float)
    for i, r in enumerate(rows_arr):
        out[i, : r.shape[0]] = r
    return out


def _clip_tiny_negatives(probs: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Clip small numerical-noise negative entries in a probs array to 0.
    Larger negatives raise — they indicate a real upstream bug rather
    than rounding."""
    if np.any(probs < -atol):
        worst = float(np.nanmin(probs))
        raise ValueError(
            f"integrate: probs contain negative entries beyond tolerance "
            f"(min={worst:.6g}); upstream cdf_at_grid is non-monotone."
        )
    return np.where(probs < 0, 0.0, probs)


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
    """Per-row bracket-backed distribution.

    Storage is always 2-D:
      ``edges``: (N, B_max + 1) — row i's bracket boundaries live in
        the first ``B_i + 1`` columns; trailing columns are NaN.
      ``probs``: (N, B_max)     — row i's bracket probabilities live
        in the first ``B_i`` columns; trailing columns are NaN.

    All math is per-row. Ragged-row support is via NaN padding: a row's
    valid prefix is everything before the first NaN in ``edges`` (and
    the matching one-shorter prefix in ``probs``). All accessor methods
    mask NaN-padded positions out and return finite results for valid
    rows.

    The ``shared_edges`` helper returns the 1-D edge vector iff every
    row has identical edges (and no NaN padding); otherwise it raises.
    Use it from legacy callers that still assume a shared ladder.
    """

    edges: np.ndarray            # (N, B+1) with NaN padding for ragged rows
    probs: np.ndarray            # (N, B) with NaN padding for ragged rows

    @classmethod
    def from_arrays(
        cls,
        *,
        edges: np.ndarray,           # 1-D (B+1,) broadcast to all rows, OR 2-D (N, B+1)
        probs: np.ndarray,           # (N, B)
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> BracketForecast:
        edges_in = np.asarray(edges, dtype=float)
        probs = np.asarray(probs, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if probs.ndim != 2:
            raise ValueError(f"probs must be 2-D (N, B); got shape {probs.shape}")
        if probs.shape[0] != ids.shape[0]:
            raise ValueError(f"N mismatch: probs N={probs.shape[0]} ids N={ids.shape[0]}")
        N = probs.shape[0]
        if edges_in.ndim == 1:
            if edges_in.shape[0] != probs.shape[1] + 1:
                raise ValueError(
                    f"probs shape {probs.shape} incompatible with shared edges {edges_in.shape}"
                )
            if np.any(np.diff(edges_in) <= 0):
                raise ValueError("edges must be monotone strictly increasing")
            edges = np.broadcast_to(edges_in[None, :], (N, edges_in.shape[0])).copy()
        elif edges_in.ndim == 2:
            if edges_in.shape[0] != N:
                raise ValueError(f"edges N={edges_in.shape[0]} != probs N={N}")
            if edges_in.shape[1] != probs.shape[1] + 1:
                raise ValueError(
                    f"edges shape {edges_in.shape} incompatible with probs {probs.shape}"
                )
            # Per-row monotonicity check, NaN-tolerant: for each row, the
            # finite prefix must be strictly increasing.
            edge_nan = np.isnan(edges_in)
            for i in range(N):
                row = edges_in[i]
                valid_len = int((~edge_nan[i]).sum())
                if valid_len < 2:
                    raise ValueError(
                        f"row {i}: edges must have ≥2 finite entries; got {valid_len}"
                    )
                prefix = row[:valid_len]
                if np.any(np.diff(prefix) <= 0):
                    raise ValueError(
                        f"row {i}: finite edge prefix must be monotone strictly increasing"
                    )
            edges = edges_in
        else:
            raise ValueError(f"edges must be 1-D or 2-D; got shape {edges_in.shape}")
        # Validate probs row-wise. NaN positions in probs must align with
        # the NaN-padded edge tail (row's B_i = valid_edges - 1).
        prob_nan = np.isnan(probs)
        edge_nan = np.isnan(edges)
        for i in range(N):
            valid_e = int((~edge_nan[i]).sum())
            valid_p = int((~prob_nan[i]).sum())
            if valid_p != valid_e - 1:
                raise ValueError(
                    f"row {i}: {valid_p} finite probs but {valid_e} finite edges — "
                    f"expected probs = edges - 1"
                )
            p_row = probs[i, :valid_p]
            if np.any(p_row < 0):
                raise ValueError(f"row {i}: probs must be nonnegative")
            if not np.isclose(p_row.sum(), 1.0, atol=1e-6):
                raise ValueError(
                    f"row {i}: probs must sum to 1; got {float(p_row.sum()):.6g}"
                )
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

    def shared_edges(self) -> np.ndarray:
        """Return the 1-D edge vector if every row shares the same edges.

        Raises if rows have ragged lengths (any NaN padding) or numerically
        distinct edge vectors. Use from legacy callers that still assume a
        shared ladder; new code should consume ``self.edges`` (2-D) directly.
        """
        edges = self.edges
        if np.isnan(edges).any():
            raise ValueError(
                "shared_edges: at least one row has NaN-padded edges — rows "
                "are ragged. Update caller to consume per-row edges."
            )
        first = edges[0]
        if not np.allclose(edges, first[None, :]):
            raise ValueError(
                "shared_edges: rows have distinct edge vectors. Update caller "
                "to consume per-row edges."
            )
        return first.copy()

    # ---------- internal helpers ----------

    def _row_valid_B(self) -> np.ndarray:
        """Per-row number of valid bins B_i. (N,) ints."""
        # B_i = count(non-NaN probs in row i) = count(non-NaN edges in row i) - 1.
        return (~np.isnan(self.probs)).sum(axis=1).astype(int)

    def _row_cum(self) -> np.ndarray:
        """(N, B_max + 1) cumulative probs. NaN-tolerant via nan-to-zero
        on probs; the cumulative tail beyond a row's valid prefix sits at
        the row's full mass (= 1) which is the right value for any query
        ≥ that row's right edge."""
        N, B_max = self.probs.shape
        p_clean = np.nan_to_num(self.probs, nan=0.0)
        cum = np.concatenate([np.zeros((N, 1)), np.cumsum(p_clean, axis=1)], axis=1)
        return cum

    def _per_row_search(self, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For a (N, M) query array, return (k_clipped, below, above) per
        (row, col). Uses each row's own edges (NaN-aware)."""
        N, M = vals.shape
        edges = self.edges
        B_per_row = self._row_valid_B()                # (N,)
        # left/right valid edge per row.
        left_edge = edges[:, 0]                         # (N,)
        right_edge = edges[np.arange(N), B_per_row]     # (N,) — edges[i, B_i]
        below = vals <= left_edge[:, None]
        above = vals >= right_edge[:, None]
        k = np.empty((N, M), dtype=int)
        # Per-row searchsorted on each row's finite prefix.
        for i in range(N):
            B_i = int(B_per_row[i])
            e_i = edges[i, :B_i + 1]
            ki = np.searchsorted(e_i, vals[i], side="right") - 1
            k[i] = np.clip(ki, 0, B_i - 1)
        return k, below, above

    # ---------- math ----------

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        N = self.ids.shape[0]
        M = x_arr.shape[0]
        # Broadcast x across all rows so each row evaluates the same M points.
        vals = np.broadcast_to(x_arr[None, :], (N, M)).copy()
        return self._cdf_2d(vals)[:, 0] if scalar else self._cdf_2d(vals)

    def _cdf_2d(self, vals: np.ndarray) -> np.ndarray:
        """Per-row CDF at vals[i, :]. Shape in/out: (N, M)."""
        N, M = vals.shape
        edges = self.edges
        probs_clean = np.nan_to_num(self.probs, nan=0.0)
        cum = self._row_cum()                           # (N, B_max+1)
        k, below, above = self._per_row_search(vals)    # (N, M) each
        rows_idx = np.arange(N)[:, None]                # (N, 1)
        # Gather per-row edge[k] and edge[k+1] and prob[k].
        lo = np.take_along_axis(edges, k, axis=1)       # (N, M)
        hi = np.take_along_axis(edges, k + 1, axis=1)   # (N, M)
        p_k = np.take_along_axis(probs_clean, k, axis=1)
        c_k = np.take_along_axis(cum, k, axis=1)
        widths = hi - lo
        safe_w = np.where(widths > 0, widths, 1.0)
        frac = np.where(widths > 0, (vals - lo) / safe_w, 0.0)
        out = c_k + frac * p_k
        out = np.where(below, 0.0, out)
        out = np.where(above, 1.0, out)
        return out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        N = self.ids.shape[0]
        if y_arr.shape[0] != N:
            raise ValueError(f"cdf_at: y has {y_arr.shape[0]} rows, dist has {N}")
        vals = y_arr.reshape(N, 1)
        return self._cdf_2d(vals)[:, 0]

    def cdf_at_grid(self, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError(f"cdf_at_grid: y must be 2-D (N, M); got shape {y_arr.shape}")
        N = self.ids.shape[0]
        if y_arr.shape[0] != N:
            raise ValueError(f"cdf_at_grid: y has {y_arr.shape[0]} rows, dist has {N}")
        nan_mask = np.isnan(y_arr)
        y_safe = np.where(nan_mask, 0.0, y_arr)
        out = self._cdf_2d(y_safe)
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        N = self.ids.shape[0]
        edges = self.edges
        probs_clean = np.nan_to_num(self.probs, nan=0.0)
        cum = self._row_cum()
        B_per_row = self._row_valid_B()
        out = np.empty((N, tau_arr.shape[0]))
        for i in range(N):
            B_i = int(B_per_row[i])
            e_i = edges[i, :B_i + 1]
            cum_i = cum[i, :B_i + 1]
            for j, t in enumerate(tau_arr):
                if t <= 0:
                    out[i, j] = e_i[0]
                elif t >= 1:
                    out[i, j] = e_i[-1]
                else:
                    k = int(np.searchsorted(cum_i, t, side="right") - 1)
                    k = max(0, min(k, B_i - 1))
                    width_p = cum_i[k + 1] - cum_i[k]
                    if width_p <= 0:
                        out[i, j] = e_i[k]
                    else:
                        frac = (t - cum_i[k]) / width_p
                        out[i, j] = e_i[k] + frac * (e_i[k + 1] - e_i[k])
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        if density_method != "step":
            raise ValueError(
                "pdf on bracket backing requires density_method='step' "
                "(no silent KDE bandwidth)"
            )
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        N = self.ids.shape[0]
        M = x_arr.shape[0]
        vals = np.broadcast_to(x_arr[None, :], (N, M)).copy()
        edges = self.edges
        probs_clean = np.nan_to_num(self.probs, nan=0.0)
        B_per_row = self._row_valid_B()
        k, below, above = self._per_row_search(vals)
        rows_idx = np.arange(N)[:, None]
        lo = np.take_along_axis(edges, k, axis=1)
        hi = np.take_along_axis(edges, k + 1, axis=1)
        p_k = np.take_along_axis(probs_clean, k, axis=1)
        widths = hi - lo
        safe_w = np.where(widths > 0, widths, 1.0)
        density = np.where(widths > 0, p_k / safe_w, 0.0)
        # Outside the row's support → 0 (matches v0.2 behaviour).
        inside = ~(below | above)
        out = np.where(inside, density, 0.0)
        return out[:, 0] if scalar else out

    def mean(self):
        # Per-row midpoints weighted by probs; NaN-tolerant via nansum.
        mids = 0.5 * (self.edges[:, 1:] + self.edges[:, :-1])      # (N, B_max)
        return np.nansum(self.probs * mids, axis=1)

    def variance(self):
        mids = 0.5 * (self.edges[:, 1:] + self.edges[:, :-1])
        m = np.nansum(self.probs * mids, axis=1)
        ex2 = np.nansum(self.probs * (mids ** 2), axis=1)
        return ex2 - m ** 2

    def realized_bin(self, y: np.ndarray) -> np.ndarray:
        """Per-row index of the bracket containing realized y. (N,) ints.

        y below row's leftmost edge → 0; y at/above rightmost edge → B_i - 1.
        """
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        N = self.ids.shape[0]
        if y_arr.shape[0] != N:
            raise ValueError(f"realized_bin: y has {y_arr.shape[0]} rows, dist has {N}")
        edges = self.edges
        B_per_row = self._row_valid_B()
        out = np.empty(N, dtype=int)
        for i in range(N):
            B_i = int(B_per_row[i])
            e_i = edges[i, :B_i + 1]
            k = int(np.searchsorted(e_i, y_arr[i], side="right") - 1)
            out[i] = max(0, min(k, B_i - 1))
        return out


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
