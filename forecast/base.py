"""DistributionForecast — abstract base for all distribution backings.

Concrete subclasses live in sibling modules:

- ``parametric.py`` — ``NormalForecast``, ``StudentTForecast``, ``MixtureNormalForecast``
- ``quantile.py``   — ``QuantileForecast``
- ``bracket.py``    — ``BracketForecast``

Subclass references in ``from_*`` classmethods and ``integrate`` use
local imports to keep base.py at the bottom of the dependency graph.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bracketlearn.forecast._helpers import _clip_tiny_negatives, _to_dense_2d
from bracketlearn.forecast._meta import Backing, ParametricFamily, ProvenanceMeta, TailPolicy

if TYPE_CHECKING:
    from bracketlearn.forecast.bracket import BracketForecast
    from bracketlearn.forecast.parametric import (
        MixtureNormalForecast,
        NormalForecast,
        StudentTForecast,
    )
    from bracketlearn.forecast.quantile import QuantileForecast


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

    @abc.abstractmethod
    def crps(self, y: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def log_score(self, y: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def to_point(self, *, how: str = "mean") -> np.ndarray: ...

    def pit(self, y: np.ndarray) -> np.ndarray:
        """Probability Integral Transform: F(y) per row. Uniform if calibrated."""
        return self.cdf_at(np.asarray(y, dtype=float))

    @classmethod
    @abc.abstractmethod
    def stitch(
        cls,
        folds: list[tuple[np.ndarray, DistributionForecast]],
        *,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> DistributionForecast:
        """Concatenate per-fold OOF dists into one whole-data OOF dist.

        ``folds`` is a list of ``(orig_row_indices, dist_for_that_fold)`` pairs.
        Output ids are the original row indices so ``y[ids]`` recovers the
        realized targets for OOF scoring."""
        ...

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
        from bracketlearn.forecast.bracket import BracketForecast

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
        from bracketlearn.forecast.parametric import NormalForecast

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
        from bracketlearn.forecast.parametric import StudentTForecast

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
        from bracketlearn.forecast.parametric import MixtureNormalForecast

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
        from bracketlearn.forecast.quantile import QuantileForecast

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
        from bracketlearn.forecast.bracket import BracketForecast

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
