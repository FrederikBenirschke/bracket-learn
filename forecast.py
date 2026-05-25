"""Data objects: PointForecast, DistributionForecast, ContractForecast, ProvenanceMeta.

All frozen dataclasses. ndarrays inside are set read-only in __post_init__
to make immutability real (frozen=True alone only freezes attribute binding,
not buffer contents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Literal

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
    tail_policy: "TailPolicy | None" = None
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
    ) -> "DistributionForecast":
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
    ) -> "DistributionForecast":
        ...

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
    ) -> "DistributionForecast":
        ...

    @classmethod
    def from_quantiles(
        cls,
        taus: np.ndarray,
        qvals: np.ndarray,
        *,
        tail_policy: "TailPolicy",      # REQUIRED — no default
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> "DistributionForecast":
        """Quantile-backed. tail_policy is required (Rule #0.5: no silent
        linear extrapolation)."""
        ...

    @classmethod
    def from_empirical(
        cls,
        members: np.ndarray,            # (N, K)
        *,
        tail_policy: "TailPolicy",      # REQUIRED
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> "DistributionForecast":
        ...

    @classmethod
    def from_brackets(
        cls,
        edges: np.ndarray,              # (B+1,)
        probs: np.ndarray,              # (N, B)
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> "DistributionForecast":
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
        if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("probs must be nonnegative and sum to 1 per row")
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
        else:
            raise NotImplementedError(
                f"cdf not implemented for backing={self.backing} family={self.family}"
            )

        return out[:, 0] if scalar else out

    def ppf(self, tau: np.ndarray | float) -> np.ndarray:
        """Quantile function."""
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")

        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            mu = self.params["mu"][:, None]
            sigma = self.params["sigma"][:, None]
            out = _stats.norm.ppf(tau_arr[None, :], loc=mu, scale=sigma)
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
        elif self.backing == Backing.BRACKET:
            if density_method != "step":
                raise ValueError(
                    "pdf on bracket backing requires density_method='step' "
                    "(Rule #0.5: no silent KDE bandwidth)"
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
        if self.backing == Backing.BRACKET:
            # bin midpoint expectation under uniform-within-bin
            mids = 0.5 * (self.edges[1:] + self.edges[:-1])    # (B,)
            return (self.probs * mids[None, :]).sum(axis=1)
        raise NotImplementedError(f"mean not implemented for backing={self.backing}")

    def variance(self) -> np.ndarray:
        if self.backing == Backing.PARAMETRIC and self.family == ParametricFamily.NORMAL:
            return self.params["sigma"] ** 2
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
        raise NotImplementedError(f"sample not implemented for backing={self.backing}")

    # NOTE: expected_payoff is intentionally NOT here. Each ContractAdapter
    # owns its own price() per backing (§5/§8). MC lives only in the
    # `Custom` adapter.

    # ----------------------------------------------------------- conversions

    def to_quantiles(self, taus: np.ndarray) -> "DistributionForecast":
        """Returns new dist with quantile backing. Records lossy conversion
        into provenance.conversion_chain."""
        ...

    def to_brackets(self, edges: np.ndarray) -> "DistributionForecast":
        ...

    def to_normal(self) -> "DistributionForecast":
        """Moment match. Lossy for fat-tailed or skewed inputs."""
        ...

    def is_lossless_to(self, target_backing: Backing) -> bool:
        """True iff conversion to target_backing preserves all information."""
        ...


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
    ) -> "ContractForecast":
        """Contract-space recalibration (§8.4). Distinct from
        DistributionForecast-level calibration."""
        ...
