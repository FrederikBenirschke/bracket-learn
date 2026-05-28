"""v0.3.0 — per-row BracketForecast storage + integrate() lift.

BracketForecast.edges is now (N, B+1) per row, with NaN padding for
ragged rows. integrate(edges_per_row) on DistributionForecast lifts any
subclass to a BracketForecast on a per-row grid via cdf_at_grid + diff.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as _stats

from bracketlearn.forecast import (
    BracketForecast,
    DistributionForecast,
    NormalForecast,
    QuantileForecast,
    StudentTForecast,
    TailPolicy,
    TailRule,
)

# ---------------------------------------------------------------------------
# Storage: edges is always 2-D (N, B+1).
# ---------------------------------------------------------------------------


def test_from_brackets_1d_edges_broadcast_to_2d(prov, ids_ts, rng):
    n, B = 5, 4
    ids, ts = ids_ts(n)
    edges = np.linspace(0.0, 1.0, B + 1)
    probs = rng.dirichlet(np.ones(B), size=n)
    d = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    assert isinstance(d, BracketForecast)
    assert d.edges.shape == (n, B + 1)
    # Every row equals the input ladder.
    for i in range(n):
        np.testing.assert_array_equal(d.edges[i], edges)


def test_per_row_edges_round_trip(prov, ids_ts, rng):
    n, B = 4, 3
    ids, ts = ids_ts(n)
    # Per-row ladder: row i is shifted by i.
    edges = np.array([np.linspace(i, i + 1, B + 1) for i in range(n)])
    probs = rng.dirichlet(np.ones(B), size=n)
    d = BracketForecast.from_arrays(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    np.testing.assert_array_equal(d.edges, edges)
    np.testing.assert_array_equal(d.probs, probs)


def test_ragged_rows_with_nan_padding(prov, ids_ts):
    n = 3
    ids, ts = ids_ts(n)
    # row 0: 4 bins, row 1: 2 bins, row 2: 3 bins.
    edges = np.full((n, 5), np.nan)
    probs = np.full((n, 4), np.nan)
    edges[0, :5] = np.linspace(0, 1, 5)
    probs[0, :4] = [0.25, 0.25, 0.25, 0.25]
    edges[1, :3] = np.linspace(10, 20, 3)
    probs[1, :2] = [0.3, 0.7]
    edges[2, :4] = np.linspace(-5, 5, 4)
    probs[2, :3] = [0.1, 0.6, 0.3]
    d = BracketForecast.from_arrays(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    # mean per row uses only finite mids.
    m = d.mean()
    # Row 0 mean = sum of mids * probs = 0.125+0.375*0.25+0.625*0.25+0.875*0.25 = 0.5
    assert np.isclose(m[0], 0.5)
    # Row 1: mid bin = (10 + 15)/2*0.3 + (15 + 20)/2*0.7 = 12.5*0.3 + 17.5*0.7 = 16.0
    assert np.isclose(m[1], 16.0)
    # realized_bin handles ragged rows correctly.
    bins = d.realized_bin(np.array([0.6, 12.0, 1.0]))
    # row0 edges [0, 0.25, 0.5, 0.75, 1.0]: 0.6 → bin 2.
    # row1 edges [10, 15, 20]: 12 → bin 0.
    # row2 edges [-5, -5/3, 5/3, 5]: 1.0 → bin 1 (in [-1.67, 1.67]).
    assert bins.tolist() == [2, 0, 1]


def test_shared_edges_returns_1d_when_uniform(prov, ids_ts, rng):
    n, B = 6, 4
    ids, ts = ids_ts(n)
    edges = np.linspace(0, 1, B + 1)
    probs = rng.dirichlet(np.ones(B), size=n)
    d = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    out = d.shared_edges()
    assert out.shape == (B + 1,)
    np.testing.assert_array_equal(out, edges)


def test_shared_edges_raises_on_per_row(prov, ids_ts, rng):
    n, B = 4, 3
    ids, ts = ids_ts(n)
    edges = np.array([np.linspace(i, i + 1, B + 1) for i in range(n)])
    probs = rng.dirichlet(np.ones(B), size=n)
    d = BracketForecast.from_arrays(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    with pytest.raises(ValueError, match="distinct edge vectors"):
        d.shared_edges()


def test_shared_edges_raises_on_ragged(prov, ids_ts):
    n = 2
    ids, ts = ids_ts(n)
    edges = np.full((n, 4), np.nan)
    probs = np.full((n, 3), np.nan)
    edges[0, :4] = [0.0, 1.0, 2.0, 3.0]
    probs[0, :3] = [0.3, 0.3, 0.4]
    edges[1, :3] = [0.0, 1.0, 2.0]
    probs[1, :2] = [0.5, 0.5]
    d = BracketForecast.from_arrays(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    with pytest.raises(ValueError, match="NaN-padded"):
        d.shared_edges()


# ---------------------------------------------------------------------------
# integrate() lift — per subclass.
# ---------------------------------------------------------------------------


def test_integrate_normal_to_bracket(prov, ids_ts, rng):
    n, B = 8, 5
    ids, ts = ids_ts(n)
    mu = rng.normal(0, 1, n)
    sigma = rng.uniform(0.5, 1.5, n)
    d = NormalForecast.from_arrays(
        mu=mu, sigma=sigma, ids=ids, timestamps=ts, provenance=prov,
    )
    edges = np.linspace(-5, 5, B + 1)
    out = d.integrate(edges)
    assert isinstance(out, BracketForecast)
    assert out.edges.shape == (n, B + 1)
    assert out.probs.shape == (n, B)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-9)
    # Recover per-row mass on each bin from scipy directly.
    cdf_lo = _stats.norm.cdf(edges[:-1], loc=mu[:, None], scale=sigma[:, None])
    cdf_hi = _stats.norm.cdf(edges[1:], loc=mu[:, None], scale=sigma[:, None])
    ref = cdf_hi - cdf_lo
    ref = ref / ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(out.probs, ref, atol=1e-9)


def test_integrate_normal_per_row_edges(prov, ids_ts, rng):
    n, B = 5, 4
    ids, ts = ids_ts(n)
    mu = rng.normal(0, 1, n)
    sigma = rng.uniform(0.5, 1.5, n)
    d = NormalForecast.from_arrays(
        mu=mu, sigma=sigma, ids=ids, timestamps=ts, provenance=prov,
    )
    # Different edges per row.
    edges_per_row = [
        np.linspace(mu[i] - 3 * sigma[i], mu[i] + 3 * sigma[i], B + 1)
        for i in range(n)
    ]
    out = d.integrate(edges_per_row)
    assert out.edges.shape == (n, B + 1)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-9)


def test_integrate_student_t(prov, ids_ts, rng):
    n, B = 6, 4
    ids, ts = ids_ts(n)
    mu = rng.normal(0, 1, n)
    sigma = rng.uniform(0.5, 1.5, n)
    df = np.full(n, 5.0)
    d = StudentTForecast.from_arrays(
        mu=mu, sigma=sigma, df=df, ids=ids, timestamps=ts, provenance=prov,
    )
    edges = np.linspace(-10, 10, B + 1)
    out = d.integrate(edges)
    assert isinstance(out, BracketForecast)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-9)


def test_integrate_quantile(prov, ids_ts, rng):
    n, Q, B = 4, 9, 4
    ids, ts = ids_ts(n)
    taus = np.linspace(0.05, 0.95, Q)
    # Monotone qvals per row.
    qvals = np.sort(rng.normal(0, 1, (n, Q)), axis=1)
    d = QuantileForecast.from_arrays(
        taus=taus, qvals=qvals, tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    edges = np.linspace(qvals.min() - 0.5, qvals.max() + 0.5, B + 1)
    out = d.integrate(edges)
    assert isinstance(out, BracketForecast)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-9)


def test_integrate_bracket_to_bracket_identity(prov, ids_ts, rng):
    """Bracket → same-edges bracket should round-trip to identical probs
    (no resampling)."""
    n, B = 5, 4
    ids, ts = ids_ts(n)
    edges = np.linspace(0, 1, B + 1)
    probs = rng.dirichlet(np.ones(B), size=n)
    d = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    out = d.integrate(edges)
    np.testing.assert_allclose(out.probs, probs, atol=1e-9)


def test_integrate_rejects_grid_outside_support(prov, ids_ts):
    """A bracket grid entirely below/above the distribution → zero mass
    everywhere → loud error per Rule #0.5."""
    n = 3
    ids, ts = ids_ts(n)
    d = NormalForecast.from_arrays(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    # Place edges far below the distribution support.
    edges = np.linspace(-100, -99, 4)
    with pytest.raises(ValueError, match="zero total mass"):
        d.integrate(edges)


# ---------------------------------------------------------------------------
# Per-row accessor sanity: cdf_at on BracketForecast handles per-row edges.
# ---------------------------------------------------------------------------


def test_cdf_at_per_row_edges_matches_searchsorted(prov, ids_ts, rng):
    n, B = 4, 3
    ids, ts = ids_ts(n)
    edges = np.array([np.linspace(i * 10, (i + 1) * 10, B + 1) for i in range(n)])
    probs = rng.dirichlet(np.ones(B), size=n)
    d = BracketForecast.from_arrays(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    # Query y at left edge of each row → CDF = 0.
    y_left = edges[:, 0]
    out = d.cdf_at(y_left)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)
    # Query y at right edge → CDF = 1.
    y_right = edges[:, -1]
    out = d.cdf_at(y_right)
    np.testing.assert_allclose(out, 1.0, atol=1e-12)
