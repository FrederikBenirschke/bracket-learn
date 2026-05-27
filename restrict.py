"""BracketMask — restrict a bracket forecast to a tradable subset.

Use case: a base forecaster (climatology, EMOS, mixture-normals,
stacking) emits probabilities over the full bracket grid, but at any
given timestamp some brackets may have no live quote (no bid, no ask,
no recent last) and cannot be touched. The original mass over those
brackets has nowhere to go — distributing it pro-rata across the
tradable subset is the maximum-entropy choice consistent with the
full forecast.

This is a stateless transformer: ``fit`` returns self, ``transform``
takes ``(dist, mask)`` and returns a new bracket-backed
``DistributionForecast`` whose probabilities are zero at masked-out
brackets and renormalised to sum to one over the surviving subset.

Per-row semantics: the mask varies row by row. At one timestamp every
bracket may be quoted; at another, only the body. Each row is
renormalised independently.

Per Rule #0.5: any row with an all-False mask, a zero-mass intersection,
or a shape/dtype mismatch raises loudly with the offending row indices.
There is no silent fill, no uniform fallback.

Input backing: bracket only. Other backings must be discretised first
(via the dist's ``cdf(edges)`` and
``bracket_probs_from_cdf_at_edges``). Keeping the entry point narrow
avoids ambiguity about which bracket grid the mask refers to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Self

import numpy as np

from bracketlearn.base import BaseEstimator

if TYPE_CHECKING:
    from bracketlearn.forecast import DistributionForecast


@dataclass
class BracketMask(BaseEstimator):
    """Per-row restriction of a bracket forecast to a tradable mask.

    Stateless — ``fit`` is a no-op, ``transform`` does the work. The
    mask is supplied at transform time because it varies per row and
    is not a property of the forecaster.
    """

    fitted_: bool = True  # stateless; satisfies sklearn check_is_fitted

    def fit(self, X=None, y=None) -> Self:  # noqa: ARG002 — stateless, sklearn shape
        return self

    def transform(
        self,
        dist: DistributionForecast,
        mask: np.ndarray,
    ) -> DistributionForecast:
        """Apply per-row mask; return bracket-backed dist with zeros at ~mask.

        Args:
            dist: bracket-backed forecast, ``dist.probs`` shape ``(N, K)``.
            mask: bool array shape ``(N, K)``. ``True`` = tradable.

        Returns:
            New ``DistributionForecast`` (bracket-backed, same edges,
            same ids/timestamps) with ``probs[i, ~mask[i]] == 0`` and
            ``probs[i, mask[i]]`` renormalised to sum to one.

        Raises:
            TypeError: input backing is not bracket; mask dtype is not bool.
            ValueError: mask shape mismatches ``dist.probs``; any row has
                an all-False mask; any row has zero forecast mass on its
                tradable subset.
        """
        from bracketlearn.forecast import (
            Backing,
            DistributionForecast,
            ProvenanceMeta,
        )

        if dist.backing != Backing.BRACKET:
            raise TypeError(
                f"BracketMask.transform: input backing is {dist.backing!r}; "
                f"only bracket backing is supported. Discretise first via "
                f"DistributionForecast.cdf(edges) + bracket_probs_from_cdf_at_edges."
            )

        probs = dist.probs
        if probs is None:
            raise ValueError(
                "BracketMask.transform: dist.probs is None despite "
                "backing=BRACKET — DistributionForecast invariant violated."
            )

        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError(
                f"BracketMask.transform: mask.dtype={mask.dtype!r} must be bool. "
                f"Refusing to coerce — a non-bool mask usually means caller "
                f"passed prices or counts by accident."
            )
        if mask.shape != probs.shape:
            raise ValueError(
                f"BracketMask.transform: mask shape {mask.shape} does not "
                f"match dist.probs shape {probs.shape}."
            )

        empty_rows = np.where(~mask.any(axis=1))[0]
        if empty_rows.size:
            raise ValueError(
                f"BracketMask.transform: {empty_rows.size} row(s) have "
                f"all-False mask (no tradable brackets). First offending "
                f"row indices: {empty_rows[:5].tolist()}."
            )

        masked = probs * mask
        row_sum = masked.sum(axis=1, keepdims=True)
        zero_mass_rows = np.where(row_sum.ravel() <= 0.0)[0]
        if zero_mass_rows.size:
            raise ValueError(
                f"BracketMask.transform: {zero_mass_rows.size} row(s) have "
                f"zero forecast mass on tradable brackets — the forecast "
                f"assigns no probability to any tradable bracket. First "
                f"offending row indices: {zero_mass_rows[:5].tolist()}."
            )

        out_probs = masked / row_sum

        new_prov = ProvenanceMeta(
            **{
                **dist.provenance.__dict__,
                "conversion_chain": (
                    dist.provenance.conversion_chain + ("BracketMask",)
                ),
                "created_at": datetime.now(),
            },
        )
        return DistributionForecast.from_brackets(
            edges=dist.edges,
            probs=out_probs,
            ids=dist.ids,
            timestamps=dist.timestamps,
            provenance=new_prov,
        )
