"""Contract adapters (§8).

Each adapter owns its price() method per backing. No central
expected_payoff dispatch (dropped in v0.2 per Tier A #5).

Adapters declare needs_left_tail / needs_right_tail so the framework can
warn when an unbounded payoff is paired with TailRule.clip() on the
relevant side.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from bracketlearn.forecast import (
    ContractForecast,
    ContractSpec,
    DistributionForecast,
    ProvenanceMeta,
)

# ---------------------------------------------------------------------------
# ContractAdapter protocol.
#
# Bracket math throughout uses closed-open semantics (lo ≤ X < hi), which
# matches CDF differences exactly for continuous distributions.
# ---------------------------------------------------------------------------


@runtime_checkable
class ContractAdapter(Protocol):
    name: str
    needs_left_tail: bool
    needs_right_tail: bool

    def price(
        self,
        dist: DistributionForecast,
    ) -> ContractForecast:
        ...


# ---------------------------------------------------------------------------
# Binary / bracket / ladder family — bounded payoffs, no tail needed.
# ---------------------------------------------------------------------------


def _provenance_for(dist: DistributionForecast, adapter_name: str) -> ProvenanceMeta:
    """Build a child ProvenanceMeta tagged with the adapter that produced it."""
    return ProvenanceMeta(
        forecaster_name=f"adapter:{adapter_name}",
        forecaster_version="0.1",
        fit_window=dist.provenance.fit_window,
        fold_idx=dist.provenance.fold_idx,
        calibration_set_hash=dist.provenance.calibration_set_hash,
        random_seed=None,
        code_sha=dist.provenance.code_sha,
        feature_matrix_hash=dist.provenance.feature_matrix_hash,
        created_at=datetime.now(),
    )


@dataclass
class BinaryAbove:
    """``P(X > k)`` priced as ``1 - dist.cdf(k)``.

    Maps to Kalshi / Polymarket single-threshold contracts: "high above 80°F",
    "S&P above 5000 by Friday", "candidate wins > 270 EV".
    """

    strike: float
    name: str = "binary_above"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        # dist.cdf returns shape (N,) for a scalar query.
        prob_above = 1.0 - dist.cdf(float(self.strike))
        prob_above = np.clip(prob_above, 0.0, 1.0)
        N = dist.ids.shape[0]
        return ContractForecast(
            contract_ids=np.zeros(N, dtype=int),
            entity_ids=np.asarray(dist.ids),
            timestamps=np.asarray(dist.timestamps),
            fair_price=prob_above.astype(float),
            group_id=np.asarray(dist.ids),
            contract_spec=ContractSpec(kind="binary_above"),
            provenance=_provenance_for(dist, self.name),
        )


@dataclass
class BinaryBelow:
    """``P(X ≤ k)`` priced as ``dist.cdf(k)``.

    Maps to Kalshi / Polymarket "below" contracts: "GDP below 2.5%",
    "low temperature below 32°F".
    """

    strike: float
    name: str = "binary_below"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        prob_below = dist.cdf(float(self.strike))
        prob_below = np.clip(prob_below, 0.0, 1.0)
        N = dist.ids.shape[0]
        return ContractForecast(
            contract_ids=np.zeros(N, dtype=int),
            entity_ids=np.asarray(dist.ids),
            timestamps=np.asarray(dist.timestamps),
            fair_price=prob_below.astype(float),
            group_id=np.asarray(dist.ids),
            contract_spec=ContractSpec(kind="binary_below"),
            provenance=_provenance_for(dist, self.name),
        )


@dataclass
class BracketLadder:
    """Bracket ladder with a per-row edge vector.

    Motivating venue: Kalshi temperature contracts list a different bracket
    grid each day (e.g. NYC max-temp brackets rotate daily). Each row gets
    its own ``edges_i``.

    Storage is ragged: ``edges_per_row`` is a Python list of length N, with
    ``edges_per_row[i]`` shape ``(B_i + 1,)``. Different rows may have
    different ``B_i`` (e.g. Kalshi sometimes adds an extra bracket for
    extreme-weather days).

    For the i.i.d. case where every row shares the same edges, pass
    ``edges_per_row=[edges] * N`` (cheap — the inner list holds N
    references to the same array). The old shared-edges shortcut was
    removed in v0.3.0 because every real-world venue this library targets
    has per-row edges, and keeping two adapters was API surface for a use
    case that never arose.

    Pricing uses :meth:`DistributionForecast.cdf_at_grid` on a NaN-padded
    dense matrix, so the inner CDF math runs vectorised for parametric
    backings rather than looping per row.

    Output is long-form: row ``(i, j)`` is the bracket-``j`` contract for
    entity ``i``. The flattened ``contract_ids`` index within each entity
    (0-based, 0..B_i-1 for interior buckets; with ``include_tail_buckets``,
    bucket -1 is "below edges[0]" and bucket B_i is "above edges[-1]" —
    those land at contract_id = -1 and B_i in the per-entity numbering).

    Args:
        edges_per_row: ragged ladder, len N.
        include_tail_buckets: when True, emit two extra rows per entity:
            ``cdf(edges[0])`` ("below") and ``1 - cdf(edges[-1])`` ("above").
            Mirrors Kalshi ladders that ship explicit "≤ X" and "> Y" rows.
            When False (default), only the B_i interior buckets are emitted
            and a coverage check warns/raises if the dist puts mass outside.
        strict: with ``include_tail_buckets=False``, raise on missed mass
            instead of warning.
        coverage_tol: missed-mass threshold for the coverage check. Ignored
            when ``include_tail_buckets=True`` (rows always sum to 1).
    """

    edges_per_row: list[np.ndarray]
    include_tail_buckets: bool = False
    name: str = "bracket_ladder"
    needs_left_tail: bool = False
    needs_right_tail: bool = False
    strict: bool = False
    coverage_tol: float = 1e-4

    def price(self, dist: DistributionForecast) -> ContractForecast:
        N = dist.ids.shape[0]
        if len(self.edges_per_row) != N:
            raise ValueError(
                f"edges_per_row has length {len(self.edges_per_row)}; "
                f"dist has {N} rows"
            )

        # Validate each row's edges and record B_i.
        B_per_row = np.empty(N, dtype=int)
        edges_clean: list[np.ndarray] = []
        for i, e in enumerate(self.edges_per_row):
            e_arr = np.asarray(e, dtype=float)
            if e_arr.ndim != 1 or e_arr.shape[0] < 2:
                raise ValueError(
                    f"edges_per_row[{i}] must be 1-D with ≥2 entries; "
                    f"got shape {e_arr.shape}"
                )
            if np.any(np.diff(e_arr) <= 0):
                raise ValueError(
                    f"edges_per_row[{i}] must be monotone strictly increasing"
                )
            B_per_row[i] = e_arr.shape[0] - 1
            edges_clean.append(e_arr)

        B_max = int(B_per_row.max())
        # Pad edges to (N, B_max+1) with NaN. The cdf_at_grid output's NaN
        # columns carry forward — we drop them when flattening to long form.
        edges_dense = np.full((N, B_max + 1), np.nan, dtype=float)
        for i, e_arr in enumerate(edges_clean):
            edges_dense[i, : e_arr.shape[0]] = e_arr

        # Per-row CDF at each edge: (N, B_max+1). NaN positions stay NaN.
        cdf_at_edges = dist.cdf_at_grid(edges_dense)
        # Bracket probs: diff along edges → (N, B_max). The last valid diff
        # for row i is at column B_per_row[i] - 1; columns ≥ B_per_row[i]
        # are NaN (one operand is NaN).
        probs = np.diff(cdf_at_edges, axis=1)
        # Clip tiny negative noise from numerical CDF differences. Real
        # negatives only arise from non-monotone CDF — bug upstream.
        valid_mask = ~np.isnan(probs)
        worst_neg = float(np.nanmin(probs)) if valid_mask.any() else 0.0
        if worst_neg < -1e-9:
            raise ValueError(
                f"BracketLadder: CDF non-monotone — worst diff "
                f"{worst_neg:.6g}. Indicates upstream bug."
            )
        probs = np.where(valid_mask, np.clip(probs, 0.0, 1.0), np.nan)

        # Coverage check (only meaningful when tail buckets are excluded).
        if not self.include_tail_buckets:
            row_sums = np.nansum(probs, axis=1)
            missed = 1.0 - row_sums
            worst_missed = float(missed.max())
            if worst_missed > self.coverage_tol:
                n_bad = int((missed > self.coverage_tol).sum())
                msg = (
                    f"ladder does not cover distribution support: "
                    f"worst row missed {worst_missed:.4f} of mass "
                    f"({n_bad}/{N} rows above coverage_tol={self.coverage_tol:g}). "
                    f"Set include_tail_buckets=True or widen the ladders."
                )
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)

        # Flatten to long form. Order: for entity i, emit (optional below),
        # then B_i interior buckets, then (optional above). contract_id is
        # within-entity: -1 = below, 0..B_i-1 = interior, B_i = above.
        contract_ids_list: list[int] = []
        entity_ids_list: list = []
        timestamps_list: list = []
        group_id_list: list = []
        fair_price_list: list[float] = []

        below_probs: np.ndarray | None = None
        above_probs: np.ndarray | None = None
        if self.include_tail_buckets:
            below_probs = cdf_at_edges[np.arange(N), 0]
            above_probs = 1.0 - cdf_at_edges[
                np.arange(N), B_per_row
            ]
            # Numerical clamp.
            below_probs = np.clip(below_probs, 0.0, 1.0)
            above_probs = np.clip(above_probs, 0.0, 1.0)

        ids = dist.ids
        ts = dist.timestamps
        for i in range(N):
            B_i = int(B_per_row[i])
            if self.include_tail_buckets:
                assert below_probs is not None   # set together with the flag above
                contract_ids_list.append(-1)
                entity_ids_list.append(ids[i])
                timestamps_list.append(ts[i])
                group_id_list.append(ids[i])
                fair_price_list.append(float(below_probs[i]))
            for j in range(B_i):
                contract_ids_list.append(j)
                entity_ids_list.append(ids[i])
                timestamps_list.append(ts[i])
                group_id_list.append(ids[i])
                fair_price_list.append(float(probs[i, j]))
            if self.include_tail_buckets:
                assert above_probs is not None   # set together with the flag above
                contract_ids_list.append(B_i)
                entity_ids_list.append(ids[i])
                timestamps_list.append(ts[i])
                group_id_list.append(ids[i])
                fair_price_list.append(float(above_probs[i]))

        return ContractForecast(
            contract_ids=np.asarray(contract_ids_list),
            entity_ids=np.asarray(entity_ids_list),
            timestamps=np.asarray(timestamps_list),
            fair_price=np.asarray(fair_price_list, dtype=float),
            group_id=np.asarray(group_id_list),
            contract_spec=ContractSpec(kind="bracket_ladder"),
            provenance=_provenance_for(dist, self.name),
        )


@dataclass
class ThresholdLadder:
    """One row per ``P(X > k_i)``. Shared group_id across the entity's row block.

    Maps to single-side Kalshi ladders ("high above 70°F", "high above 75°F",
    "high above 80°F" ...). Prices are *not* required to sum to 1 — they are
    survival-function values at increasing strikes, so they decrease monotonically.
    """

    strikes: np.ndarray              # (S,)
    name: str = "threshold_ladder"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        strikes = np.asarray(self.strikes, dtype=float)
        if strikes.ndim != 1 or strikes.shape[0] < 1:
            raise ValueError(
                f"strikes must be 1-D with ≥1 entries; got shape {strikes.shape}"
            )
        if np.any(np.diff(strikes) <= 0):
            raise ValueError("strikes must be monotone strictly increasing")
        # dist.cdf(strikes) → (N, S)
        cdf_at_k = dist.cdf(strikes)
        survival = np.clip(1.0 - cdf_at_k, 0.0, 1.0)   # (N, S)
        N, S = survival.shape
        contract_ids = np.tile(np.arange(S), N)
        entity_ids = np.repeat(dist.ids, S)
        timestamps = np.repeat(dist.timestamps, S)
        group_id = np.repeat(dist.ids, S)
        fair_price = survival.reshape(-1)
        return ContractForecast(
            contract_ids=contract_ids,
            entity_ids=entity_ids,
            timestamps=timestamps,
            fair_price=fair_price,
            group_id=group_id,
            contract_spec=ContractSpec(kind="threshold_ladder"),
            provenance=_provenance_for(dist, self.name),
        )


@dataclass
class Twin:
    """Paired YES / NO at a single strike. Two rows per entity sharing
    ``group_id`` (so calibrators can enforce ``p_yes + p_no = 1``).

    Maps to prediction-market spread / total contracts: "Eagles -3.5"
    (YES = covers), "Over 47.5 total points" (YES = goes over). Within
    each entity the two prices sum to exactly 1.0 by construction.

    Convention: ``contract_id=0`` is YES = ``P(X > k)``; ``contract_id=1``
    is NO = ``P(X ≤ k)``.
    """

    strike: float
    name: str = "twin"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        cdf_k = dist.cdf(float(self.strike))    # (N,)
        p_no = np.clip(cdf_k, 0.0, 1.0)
        p_yes = 1.0 - p_no
        N = dist.ids.shape[0]
        # Interleave (yes, no) per entity so contract_ids are 0,1,0,1,...
        contract_ids = np.tile(np.arange(2), N)
        entity_ids = np.repeat(dist.ids, 2)
        timestamps = np.repeat(dist.timestamps, 2)
        group_id = np.repeat(dist.ids, 2)
        fair_price = np.empty(2 * N, dtype=float)
        fair_price[0::2] = p_yes
        fair_price[1::2] = p_no
        return ContractForecast(
            contract_ids=contract_ids,
            entity_ids=entity_ids,
            timestamps=timestamps,
            fair_price=fair_price,
            group_id=group_id,
            contract_spec=ContractSpec(kind="twin"),
            provenance=_provenance_for(dist, self.name),
        )
