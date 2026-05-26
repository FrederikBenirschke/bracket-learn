"""Per-row bracket ladder (varying edges per entity) — adapter + cdf_at_grid.

Motivating venue: Kalshi temperature contracts list a different bracket
grid each day. ``DistributionForecast.cdf_at_grid`` (the underlying
primitive) and ``PerRowBracketLadder`` (the contract adapter built on it)
together let a single fitted distribution be priced against per-row
edge sets without N independent ``cdf`` calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as _stats

from bracketlearn.adapters import PerRowBracketLadder
from bracketlearn.forecast import DistributionForecast
from bracketlearn.tail import TailPolicy, TailRule


# ---------------------------------------------------------------------------
# cdf_at_grid — must equal a row-by-row cdf call (without the (N, N) cost).
# ---------------------------------------------------------------------------


def test_cdf_at_grid_normal_matches_per_row_cdf(prov, ids_ts, rng):
    n, M = 30, 4
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 2.0, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.normal(0, 1, size=(n, M))
    out = d.cdf_at_grid(y)
    assert out.shape == (n, M)
    # Reference: per-row scipy directly.
    mu, sigma = d.params["mu"], d.params["sigma"]
    ref = _stats.norm.cdf(y, loc=mu[:, None], scale=sigma[:, None])
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_cdf_at_grid_student_t(prov, ids_ts, rng):
    n, M = 20, 3
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_student_t(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 1.5, n),
        df=np.full(n, 5.0),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.normal(0, 1, size=(n, M))
    out = d.cdf_at_grid(y)
    ref = _stats.t.cdf(y, df=5.0, loc=d.params["mu"][:, None], scale=d.params["sigma"][:, None])
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_cdf_at_grid_mixture_normal(prov, ids_ts, rng):
    n, K, M = 15, 3, 5
    ids, ts = ids_ts(n)
    raw = rng.uniform(0.1, 1.0, size=(n, K))
    w = raw / raw.sum(axis=1, keepdims=True)
    mus = rng.normal(0, 1, size=(n, K))
    sigmas = rng.uniform(0.3, 1.5, size=(n, K))
    d = DistributionForecast.from_mixture_normal(
        weights=w, mus=mus, sigmas=sigmas,
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.normal(0, 1, size=(n, M))
    out = d.cdf_at_grid(y)
    # Reference: row-by-row mixture CDF.
    ref = np.empty((n, M))
    for i in range(n):
        ref[i] = (w[i] * _stats.norm.cdf(y[i][:, None], loc=mus[i], scale=sigmas[i])).sum(axis=1)
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_cdf_at_grid_bracket(prov, ids_ts, rng):
    n, B, M = 12, 6, 4
    ids, ts = ids_ts(n)
    edges = np.linspace(-3.0, 3.0, B + 1)
    raw = rng.uniform(0.1, 1.0, size=(n, B))
    probs = raw / raw.sum(axis=1, keepdims=True)
    d = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    # Mix of below/inside/above support to exercise all three branches.
    y = rng.uniform(-4.0, 4.0, size=(n, M))
    out = d.cdf_at_grid(y)
    # Reference: directly evaluate the bracket CDF row-by-row.
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(probs, axis=1)], axis=1)
    ref = np.empty((n, M))
    for i in range(n):
        for j in range(M):
            yv = y[i, j]
            if yv <= edges[0]:
                ref[i, j] = 0.0
            elif yv >= edges[-1]:
                ref[i, j] = 1.0
            else:
                k = int(np.searchsorted(edges, yv, side="right") - 1)
                k = max(0, min(k, B - 1))
                frac = (yv - edges[k]) / (edges[k + 1] - edges[k])
                ref[i, j] = cum[i, k] + frac * probs[i, k]
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_cdf_at_grid_quantile(prov, ids_ts, rng):
    n, Q, M = 10, 5, 3
    ids, ts = ids_ts(n)
    taus = np.linspace(0.1, 0.9, Q)
    qvals = np.sort(rng.normal(0, 1, size=(n, Q)), axis=1)
    d = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.uniform(-2.0, 2.0, size=(n, M))
    out = d.cdf_at_grid(y)
    ref = np.empty((n, M))
    for i in range(n):
        ref[i] = np.interp(y[i], qvals[i], taus, left=0.0, right=1.0)
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_cdf_at_grid_preserves_nan(prov, ids_ts, rng):
    """Ragged callers pad with NaN — those slots must round-trip as NaN."""
    n, M = 6, 4
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.normal(0, 1, size=(n, M))
    y[0, 3] = np.nan
    y[2, 2:] = np.nan
    out = d.cdf_at_grid(y)
    assert np.isnan(out[0, 3])
    assert np.all(np.isnan(out[2, 2:]))
    # Non-NaN entries unaffected.
    finite = ~np.isnan(y)
    np.testing.assert_allclose(out[finite], _stats.norm.cdf(y[finite]), atol=1e-12)


def test_cdf_at_grid_rejects_wrong_shape(prov, ids_ts):
    n = 4
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    with pytest.raises(ValueError, match="2-D"):
        d.cdf_at_grid(np.zeros(n))               # 1-D
    with pytest.raises(ValueError, match="rows"):
        d.cdf_at_grid(np.zeros((n + 1, 3)))      # row count mismatch


# ---------------------------------------------------------------------------
# PerRowBracketLadder — adapter contract.
# ---------------------------------------------------------------------------


def test_per_row_ladder_known_answer_normal(prov, ids_ts):
    """Hand-computed: N=2 standard normal, two distinct ladders."""
    n = 2
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Wide ladders so coverage_tol doesn't trip.
    edges = [
        np.array([-10.0, 0.0, 10.0]),       # row 0: P(X<0), P(X≥0)
        np.array([-10.0, -1.0, 1.0, 10.0]), # row 1: P(X<-1), P(-1≤X<1), P(X≥1)
    ]
    cf = PerRowBracketLadder(edges_per_row=edges).price(d)
    # row 0 has 2 buckets, row 1 has 3 → 5 contracts total.
    assert cf.fair_price.shape == (5,)
    np.testing.assert_array_equal(cf.entity_ids, np.array([0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(cf.contract_ids, np.array([0, 1, 0, 1, 2]))
    # Expected probs.
    p_row0 = [_stats.norm.cdf(0.0), 1.0 - _stats.norm.cdf(0.0)]
    p_row1 = [
        _stats.norm.cdf(-1.0),
        _stats.norm.cdf(1.0) - _stats.norm.cdf(-1.0),
        1.0 - _stats.norm.cdf(1.0),
    ]
    np.testing.assert_allclose(
        cf.fair_price, np.array(p_row0 + p_row1), atol=1e-10
    )


def test_per_row_ladder_tail_buckets_sum_to_one(prov, ids_ts, rng):
    n = 5
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 1.5, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Narrow ladders that DON'T cover the tails — but tail buckets are on,
    # so totals must still be exactly 1.0 per entity.
    edges = [np.array([-0.5, 0.0, 0.5]) for _ in range(n)]
    cf = PerRowBracketLadder(
        edges_per_row=edges,
        include_tail_buckets=True,
    ).price(d)
    # 4 contracts per entity: below, 2 interior, above.
    assert cf.fair_price.shape == (4 * n,)
    for i in range(n):
        block = cf.fair_price[cf.entity_ids == ids[i]]
        np.testing.assert_allclose(block.sum(), 1.0, atol=1e-10)


def test_per_row_ladder_coverage_check_warns(prov, ids_ts):
    n = 3
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Narrow ladders, no tail buckets → mass leaks out → warn.
    edges = [np.array([-0.1, 0.0, 0.1]) for _ in range(n)]
    with pytest.warns(UserWarning, match="does not cover"):
        PerRowBracketLadder(edges_per_row=edges).price(d)


def test_per_row_ladder_coverage_check_raises_strict(prov, ids_ts):
    n = 3
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    edges = [np.array([-0.1, 0.0, 0.1]) for _ in range(n)]
    with pytest.raises(ValueError, match="does not cover"):
        PerRowBracketLadder(edges_per_row=edges, strict=True).price(d)


def test_per_row_ladder_rejects_length_mismatch(prov, ids_ts):
    n = 4
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    edges = [np.array([-1.0, 0.0, 1.0])] * (n - 1)   # short
    with pytest.raises(ValueError, match="length"):
        PerRowBracketLadder(edges_per_row=edges).price(d)


def test_per_row_ladder_rejects_non_monotone_edges(prov, ids_ts):
    n = 2
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    edges = [
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, -1.0]),   # decreasing
    ]
    with pytest.raises(ValueError, match="monotone"):
        PerRowBracketLadder(edges_per_row=edges).price(d)


def test_per_row_ladder_matches_shared_ladder_when_edges_equal(prov, ids_ts, rng):
    """Sanity: a per-row ladder with all rows sharing the same edges must
    produce the same prices as the original BracketLadder."""
    from bracketlearn.adapters import BracketLadder

    n = 8
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 1.5, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    edges = np.linspace(-5.0, 5.0, 7)
    shared = BracketLadder(edges=edges).price(d)
    per_row = PerRowBracketLadder(
        edges_per_row=[edges.copy() for _ in range(n)]
    ).price(d)
    # Both flatten N rows × B buckets in the same (entity-major) order.
    np.testing.assert_allclose(per_row.fair_price, shared.fair_price, atol=1e-12)
    np.testing.assert_array_equal(per_row.entity_ids, shared.entity_ids)
