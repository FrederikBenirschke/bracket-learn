"""BracketLadder.price must return row sums == 1.0 (mass conservation).

Audit item B1 (AUDIT.md): qreg-backed ladder rows can sum to ~0.48 when
ladder edges fall inside the stored quantile range. Cause: clip tail
semantics drop mass below taus[0] and above taus[-1].

Invariant under test: for every (backing, tail_policy, ladder) combo
where the ladder edges cover the full support of the distribution,
sum_k (cdf(edges[k+1]) - cdf(edges[k])) == 1.

"Full support coverage" means:
  - parametric normal: edges span ≥ ±6 sigma around mu.
  - quantile (clip): edges span [qvals[0], qvals[-1]] (clip says no mass
    outside this range, so any edges containing the stored range MUST
    capture all advertised mass).
  - bracket: edges == dist.edges (lossless).
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.adapters import BracketLadder
from bracketlearn.forecast import DistributionForecast
from bracketlearn.tail import TailPolicy, TailRule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_sums(dist: DistributionForecast, edges: np.ndarray) -> np.ndarray:
    ladder = BracketLadder(edges=edges)
    contracts = ladder.price(dist)
    B = edges.shape[0] - 1
    return contracts.fair_price.reshape(-1, B).sum(axis=1)


# ---------------------------------------------------------------------------
# Parametric normal — full-support backing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 5, 20])
def test_normal_ladder_sums_to_one_when_edges_cover_support(prov, ids_ts, B):
    """Normal dist with edges spanning ±6σ → sums ≈ 1.0."""
    N = 10
    ids, ts = ids_ts(N)
    mu = np.linspace(-1.0, 1.0, N)
    sigma = np.full(N, 0.5)
    dist = DistributionForecast.from_normal(
        mu=mu, sigma=sigma, ids=ids, timestamps=ts, provenance=prov,
    )
    # ±6σ covers ~1 - 2e-9 of the mass.
    lo = float((mu - 6.0 * sigma).min())
    hi = float((mu + 6.0 * sigma).max())
    edges = np.linspace(lo, hi, B + 1)
    sums = _row_sums(dist, edges)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Bracket — exact match must be lossless.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 5, 20])
def test_bracket_ladder_lossless_when_edges_match(prov, ids_ts, B):
    """Bracket dist priced on its own edges → row sums == 1.0 exactly."""
    N = 7
    ids, ts = ids_ts(N)
    edges = np.linspace(0.0, 10.0, B + 1)
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 1.0, size=(N, B))
    probs = raw / raw.sum(axis=1, keepdims=True)
    dist = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    sums = _row_sums(dist, edges)
    np.testing.assert_allclose(sums, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Quantile (clip) — this is where the bug lives.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 5, 20])
def test_quantile_clip_ladder_sums_to_one_when_edges_cover_quantile_range(
    prov, ids_ts, B,
):
    """Quantile-backed dist with clip tail policy, ladder edges covering
    [qvals[0], qvals[-1]] should sum to 1.0.

    Rationale: clip semantics say mass beyond outermost stored quantile
    is zero — so the stored [qvals[0], qvals[-1]] range carries the
    *entire* distribution. A ladder covering that range loses nothing.
    """
    N = 5
    ids, ts = ids_ts(N)
    taus = np.linspace(0.05, 0.95, 19)
    # Shifted standard normal quantiles per row, monotone non-decreasing.
    from scipy.stats import norm
    base = norm.ppf(taus)                       # (Q,)
    shifts = np.linspace(-0.5, 0.5, N)[:, None] # (N, 1)
    qvals = base[None, :] + shifts              # (N, Q)
    dist = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Ladder edges exactly span the stored quantile range.
    lo = float(qvals.min())
    hi = float(qvals.max())
    # Expand a tiny ε so the outermost edges include the boundary point.
    edges = np.linspace(lo - 1e-9, hi + 1e-9, B + 1)
    sums = _row_sums(dist, edges)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_quantile_clip_ladder_inside_quantile_range_documented(
    prov, ids_ts,
):
    """When ladder edges fall *inside* [qvals[0], qvals[-1]], the row sum
    equals exactly taus[-1] - taus[0] (the τ-mass contained in the stored
    range). This pins the current clip semantics: tail mass outside
    [qvals[0], qvals[-1]] is dropped.

    If we later change clip to "preserve point-mass at qvals[0]/qvals[-1]",
    this test should be updated, and the cover-the-range test above will
    still pass.
    """
    N = 3
    ids, ts = ids_ts(N)
    taus = np.linspace(0.05, 0.95, 19)
    from scipy.stats import norm
    qvals = np.broadcast_to(norm.ppf(taus), (N, taus.shape[0])).copy()
    dist = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Edges strictly inside [qvals[0], qvals[-1]] = [norm.ppf(0.05), norm.ppf(0.95)].
    edges = np.array([norm.ppf(0.10), 0.0, norm.ppf(0.90)])
    sums = _row_sums(dist, edges)
    # cdf(norm.ppf(0.90)) - cdf(norm.ppf(0.10)) = 0.90 - 0.10 = 0.80.
    np.testing.assert_allclose(sums, 0.80, atol=1e-6)


def test_quantile_clip_ladder_warns_when_ladder_doesnt_cover_qvals(
    prov, ids_ts,
):
    """B1 regression: when ladder edges[-1] < max(qvals) (or edges[0] >
    min(qvals)), mass is silently dropped. The library MUST either raise
    or emit a loud warning.

    Scenario mirrors the housing example: edges[-1]=5.0 but qvals[i][-1]
    plateaus at 5.04 for some rows → ~50% of mass missed for those rows.
    """
    N = 1
    ids, ts = ids_ts(N)
    taus = np.linspace(0.05, 0.95, 11)
    # Plateau at the top (mimics LightGBM quantile output).
    qvals = np.array([[2.5, 4.1, 4.7, 4.9, 4.95, 5.04, 5.04, 5.04, 5.04, 5.04, 5.04]])
    dist = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Ladder STOPS at 5.0, qvals[-1]=5.04 → coverage violation.
    edges = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    ladder = BracketLadder(edges=edges)
    with pytest.warns(UserWarning, match="ladder does not cover"):
        ladder.price(dist)


def test_quantile_clip_ladder_strict_raises_on_coverage_violation(
    prov, ids_ts,
):
    """strict=True raises ValueError instead of warning."""
    N = 1
    ids, ts = ids_ts(N)
    taus = np.linspace(0.05, 0.95, 11)
    qvals = np.array([[2.5, 4.1, 4.7, 4.9, 4.95, 5.04, 5.04, 5.04, 5.04, 5.04, 5.04]])
    dist = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    edges = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    ladder = BracketLadder(edges=edges, strict=True)
    with pytest.raises(ValueError, match="ladder does not cover"):
        ladder.price(dist)


def test_quantile_clip_ladder_edges_outside_quantile_range_captures_all_mass(
    prov, ids_ts,
):
    """The regression test for B1: ladder edges strictly *outside* the
    stored quantile range should still sum to 1.0 under clip semantics,
    because clip says the entire distribution lives in [qvals[0], qvals[-1]].

    With the bug present (clip mapping `cdf(x<qvals[0]) = 0` instead of
    `cdf(x ≤ qvals[0]) = taus[0]` as a point mass), the outermost bins
    miss the τ_0 + (1-τ_-1) tail mass and the row sum is < 1.0.
    """
    N = 1
    ids, ts = ids_ts(N)
    taus = np.linspace(0.05, 0.95, 19)
    from scipy.stats import norm
    qvals = norm.ppf(taus)[None, :]
    dist = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Edges that strictly contain [qvals[0], qvals[-1]] with room to spare.
    edges = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
    sums = _row_sums(dist, edges)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Mixture normal — full support, should be 1.0 with wide enough edges.
# ---------------------------------------------------------------------------


def test_mixture_normal_ladder_sums_to_one(prov, ids_ts):
    N = 4
    ids, ts = ids_ts(N)
    K = 2
    weights = np.full((N, K), 0.5)
    mus = np.tile(np.array([-2.0, 2.0]), (N, 1))
    sigmas = np.full((N, K), 0.5)
    dist = DistributionForecast.from_mixture_normal(
        weights=weights, mus=mus, sigmas=sigmas,
        ids=ids, timestamps=ts, provenance=prov,
    )
    edges = np.linspace(-10.0, 10.0, 21)
    sums = _row_sums(dist, edges)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)
