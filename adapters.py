"""Contract adapters (§8).

Each adapter owns its price() method per backing. No central
expected_payoff dispatch (dropped in v0.2 per Tier A #5).

Adapters declare needs_left_tail / needs_right_tail so the framework can
warn when an unbounded payoff is paired with TailRule.clip() on the
relevant side.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from bracketlearn.forecast import (
    ContractForecast,
    ContractSpec,
    DistributionForecast,
    ProvenanceMeta,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Edge semantics (§8.1 — replaces v0.1's stringly-typed "closed-open").
# ---------------------------------------------------------------------------


class BracketEdges(StrEnum):
    CLOSED_OPEN = "closed_open"     # lo ≤ X < hi (common ladder default)
    OPEN_CLOSED = "open_closed"     # lo < X ≤ hi
    CLOSED_CLOSED = "closed_closed" # lo ≤ X ≤ hi
    OPEN_OPEN = "open_open"         # lo < X < hi


# ---------------------------------------------------------------------------
# ContractAdapter protocol.
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


@dataclass
class BinaryAbove:
    """1[X > k]. Implementation: 1 - dist.cdf(k)."""

    strike: float
    name: str = "binary_above"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("BinaryAbove.price — not yet implemented")


@dataclass
class BinaryBelow:
    strike: float
    name: str = "binary_below"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("BinaryBelow.price — not yet implemented")


@dataclass
class Bracket:
    lo: float
    hi: float
    edges: BracketEdges = BracketEdges.CLOSED_OPEN
    name: str = "bracket"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("Bracket.price — not yet implemented")


@dataclass
class BracketLadder:
    """One row per (lo, hi) interval. Shared group_id across rows so
    downstream calibrators can enforce monotonicity / simplex constraints.

    ``strict`` controls coverage-failure behavior. By construction, the
    ladder's bracket probabilities (cdf(edges[-1]) - cdf(edges[0])) only
    capture mass that falls inside [edges[0], edges[-1]]. If the
    distribution places mass outside that range — common for quantile
    backings whose stored qvals extend past the ladder, or normal
    backings whose σ-tails exceed the outermost edges — row sums fall
    below 1.0 and contract prices are biased.

    - ``strict=False`` (default): warn loudly via UserWarning when any
      row's missed mass exceeds ``coverage_tol`` (default 1e-4). The
      warning reports the worst-row missed mass so callers can judge.
    - ``strict=True``: raise ValueError instead of warning. Use this
      when downstream code requires coherent ladder probabilities (e.g.
      simplex calibration, log-loss scoring).
    """

    edges: np.ndarray               # (B+1,)
    edge_semantics: BracketEdges = BracketEdges.CLOSED_OPEN
    name: str = "bracket_ladder"
    needs_left_tail: bool = False
    needs_right_tail: bool = False
    strict: bool = False
    coverage_tol: float = 1e-4

    def price(self, dist: DistributionForecast) -> ContractForecast:
        edges = np.asarray(self.edges, dtype=float)
        B = edges.shape[0] - 1
        N = dist.ids.shape[0]
        # P(lo ≤ X < hi) = cdf(hi) - cdf(lo) for closed_open semantics.
        cdf_hi = dist.cdf(edges[1:])    # (N, B)
        cdf_lo = dist.cdf(edges[:-1])
        probs = cdf_hi - cdf_lo         # (N, B)
        # Coverage check: row sums must be ~1.0. If not, ladder edges
        # don't cover the distribution's effective support and mass is
        # being silently dropped from contract prices.
        row_sums = probs.sum(axis=1)
        missed = 1.0 - row_sums
        worst_missed = float(missed.max())
        if worst_missed > self.coverage_tol:
            n_bad = int((missed > self.coverage_tol).sum())
            msg = (
                f"ladder does not cover distribution support: "
                f"worst row missed {worst_missed:.4f} of mass "
                f"({n_bad}/{N} rows above coverage_tol={self.coverage_tol:g}). "
                f"Extend ladder edges or set strict=False to silence."
            )
            if self.strict:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
        # Flatten to long form: (N*B,) rows.
        contract_ids = np.tile(np.arange(B), N)
        entity_ids = np.repeat(dist.ids, B)
        timestamps = np.repeat(dist.timestamps, B)
        group_id = np.repeat(dist.ids, B)   # one ladder per entity
        fair_price = probs.reshape(-1)
        return ContractForecast(
            contract_ids=contract_ids,
            entity_ids=entity_ids,
            timestamps=timestamps,
            fair_price=fair_price,
            group_id=group_id,
            contract_spec=ContractSpec(kind="bracket_ladder"),
            provenance=ProvenanceMeta(
                forecaster_name=f"adapter:{self.name}",
                forecaster_version="0.1",
                fit_window=dist.provenance.fit_window,
                fold_idx=dist.provenance.fold_idx,
                calibration_set_hash=dist.provenance.calibration_set_hash,
                random_seed=None,
                code_sha=dist.provenance.code_sha,
                feature_matrix_hash=dist.provenance.feature_matrix_hash,
                created_at=datetime.now(),
            ),
        )


@dataclass
class ThresholdLadder:
    """One row per 1[X > k_i]. Shared group_id."""

    strikes: np.ndarray
    name: str = "threshold_ladder"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("ThresholdLadder.price — not yet implemented")


@dataclass
class Twin:
    """YES = 1[X > k], NO = 1[X ≤ k]. Shared group_id (paired)."""

    strike: float
    name: str = "twin"
    needs_left_tail: bool = False
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("Twin.price — not yet implemented")


# ---------------------------------------------------------------------------
# Vanilla call / put — unbounded payoffs, side-specific tail.
# ---------------------------------------------------------------------------


@dataclass
class VanillaCall:
    """max(X - k, 0).

    Pricing per backing:
      - parametric normal → Bachelier closed form.
      - quantile          → trapezoid over quantile grid + right-tail integral.
      - empirical         → mean of max(members - k, 0).
      - bracket           → discrete sum, but with no info above edges[-1] →
                            tail policy mandatory and warned if clip.
    """

    strike: float
    name: str = "vanilla_call"
    needs_left_tail: bool = False
    needs_right_tail: bool = True

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("VanillaCall.price — not yet implemented")


@dataclass
class VanillaPut:
    """max(k - X, 0). Symmetric to VanillaCall on the left side."""

    strike: float
    name: str = "vanilla_put"
    needs_left_tail: bool = True
    needs_right_tail: bool = False

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("VanillaPut.price — not yet implemented")


# ---------------------------------------------------------------------------
# Composition primitives.
# ---------------------------------------------------------------------------


@dataclass
class LinearCombo:
    """Σ wᵢ · partᵢ. Single composition primitive for spreads.

    CallSpread, Butterfly, Condor, RatioSpread are factory functions
    returning a LinearCombo (no operator overloading).

    needs_*_tail is the OR of any part's needs_*_tail.
    """

    parts: list[tuple[float, ContractAdapter]]
    name: str = "linear_combo"

    @property
    def needs_left_tail(self) -> bool:
        return any(p.needs_left_tail for _, p in self.parts)

    @property
    def needs_right_tail(self) -> bool:
        return any(p.needs_right_tail for _, p in self.parts)

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("LinearCombo.price — not yet implemented")


def CallSpread(k1: float, k2: float) -> LinearCombo:
    """Long call at k1, short call at k2 (k1 < k2)."""
    return LinearCombo(parts=[(1.0, VanillaCall(k1)), (-1.0, VanillaCall(k2))],
                       name=f"call_spread({k1},{k2})")


def Butterfly(k1: float, k2: float, k3: float) -> LinearCombo:
    """k1 < k2 < k3, typically k2 = (k1+k3)/2."""
    return LinearCombo(parts=[(1.0, VanillaCall(k1)),
                              (-2.0, VanillaCall(k2)),
                              (1.0, VanillaCall(k3))],
                       name=f"butterfly({k1},{k2},{k3})")


def Condor(k1: float, k2: float, k3: float, k4: float) -> LinearCombo:
    """k1 < k2 < k3 < k4."""
    return LinearCombo(parts=[(1.0, VanillaCall(k1)),
                              (-1.0, VanillaCall(k2)),
                              (-1.0, VanillaCall(k3)),
                              (1.0, VanillaCall(k4))],
                       name=f"condor({k1},{k2},{k3},{k4})")


# ---------------------------------------------------------------------------
# PerRow — wrap a scalar-strike adapter into a per-row adapter (§8.2 / Q9).
# ---------------------------------------------------------------------------


@dataclass
class PerRow:
    """Per-row strike (or other scalar param) wrapper.

    Refuses the float | ndarray union footgun on adapter constructors.
    Each row of dist priced against its own param value.

    Usage:
        adapter = PerRow(BinaryAbove, strike=arr_per_row)
    """

    adapter_cls: type
    name: str = "per_row"
    needs_left_tail: bool = False
    needs_right_tail: bool = False
    per_row_kwargs: dict[str, np.ndarray] = field(default_factory=dict)

    def __init__(self, adapter_cls: type, **per_row_kwargs: np.ndarray):
        self.adapter_cls = adapter_cls
        self.per_row_kwargs = per_row_kwargs
        self.name = f"per_row({adapter_cls.__name__})"
        # needs_*_tail copied from a probe instance.
        probe_kwargs = {k: float(v[0]) for k, v in per_row_kwargs.items()}
        probe = adapter_cls(**probe_kwargs)
        self.needs_left_tail = probe.needs_left_tail
        self.needs_right_tail = probe.needs_right_tail

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("PerRow.price — not yet implemented")


# ---------------------------------------------------------------------------
# Custom — arbitrary user payoff, MC-priced.
# ---------------------------------------------------------------------------


@dataclass
class Custom:
    """Arbitrary user payoff. Priced by Monte Carlo (the only adapter that
    uses MC by design; needed for non-analytical payoffs).

    Support bounds are required (no silent assume-unbounded): pass
    support_lo=None to mean -∞, support_hi=None to mean +∞. needs_*_tail
    is inferred from the bounds.
    """

    payoff_fn: Callable[[np.ndarray], np.ndarray]
    support_lo: float | None
    support_hi: float | None
    n_samples: int = 10_000
    name: str = "custom"

    @property
    def needs_left_tail(self) -> bool:
        return self.support_lo is None

    @property
    def needs_right_tail(self) -> bool:
        return self.support_hi is None

    def price(self, dist: DistributionForecast) -> ContractForecast:
        raise NotImplementedError("Custom.price — not yet implemented")


# ---------------------------------------------------------------------------
# VenueSpec — units / multiplier / tick / min-size (§8.3).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VenueSpec:
    """Venue-specific quote translation. Multiplies adapter fair_price
    (payoff-natural units) into venue units."""

    venue: str                      # "venue_a", "exchange_b", ...
    ticker: str
    multiplier: float = 1.0         # $1 for binaries; $20 for CME HDD index pt
    tick_size: float = 0.01
    min_size: float = 1.0


def to_quote(
    contracts: ContractForecast,
    venue_spec: VenueSpec,
) -> ContractForecast:
    """Apply VenueSpec multiplier; return new ContractForecast with
    fair_price in venue units."""
    raise NotImplementedError("to_quote — not yet implemented")
