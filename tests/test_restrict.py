"""Tests for BracketMask — per-row restriction of a bracket forecast.

Pins the invariants from restrict.py's docstring:
- mass preservation per row (sum=1 over surviving brackets)
- zeros at masked-out positions
- all-True mask is the identity
- all-False mask raises with row index
- zero-mass-on-tradable raises with row index
- mask shape mismatch raises
- mask dtype non-bool raises (no silent coerce; Rule #0.5)
- non-bracket input raises (no silent discretisation)
- provenance gets BracketMask appended to conversion_chain
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.forecast import DistributionForecast
from bracketlearn.restrict import BracketMask


def _bracket_dist(probs: np.ndarray, prov, ids_ts) -> DistributionForecast:
    N, B = probs.shape
    ids, ts = ids_ts(N)
    edges = np.linspace(0.0, 1.0, B + 1)
    return DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )


def test_basic_renorm(prov, ids_ts):
    probs = np.array([
        [0.1, 0.4, 0.3, 0.2],
        [0.25, 0.25, 0.25, 0.25],
    ])
    mask = np.array([
        [True, False, True, True],
        [True, True, True, True],
    ])
    dist = _bracket_dist(probs, prov, ids_ts)
    out = BracketMask().transform(dist, mask)
    # row 0: kept = 0.1+0.3+0.2 = 0.6; renorm
    np.testing.assert_allclose(
        out.probs[0], [0.1 / 0.6, 0.0, 0.3 / 0.6, 0.2 / 0.6], atol=1e-12,
    )
    np.testing.assert_allclose(out.probs[1], probs[1], atol=1e-12)


def test_row_sums_to_one(prov, ids_ts, rng):
    N, B = 50, 6
    probs = rng.dirichlet(np.ones(B), size=N)
    mask = rng.random((N, B)) > 0.3
    # Force at least one True per row so we exercise the happy path only.
    for i in range(N):
        if not mask[i].any():
            mask[i, rng.integers(0, B)] = True
    dist = _bracket_dist(probs, prov, ids_ts)
    out = BracketMask().transform(dist, mask)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-12)


def test_zeros_on_masked_positions(prov, ids_ts, rng):
    N, B = 20, 5
    probs = rng.dirichlet(np.ones(B), size=N)
    mask = rng.random((N, B)) > 0.3
    for i in range(N):
        if not mask[i].any():
            mask[i, 0] = True
    dist = _bracket_dist(probs, prov, ids_ts)
    out = BracketMask().transform(dist, mask)
    assert np.all(out.probs[~mask] == 0.0)


def test_all_true_mask_is_identity(prov, ids_ts, rng):
    probs = rng.dirichlet(np.ones(4), size=10)
    mask = np.ones_like(probs, dtype=bool)
    dist = _bracket_dist(probs, prov, ids_ts)
    out = BracketMask().transform(dist, mask)
    np.testing.assert_allclose(out.probs, probs, atol=1e-12)


def test_all_false_row_raises(prov, ids_ts):
    probs = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ])
    mask = np.array([
        [True, True, False, False],
        [False, False, False, False],  # offending
    ])
    dist = _bracket_dist(probs, prov, ids_ts)
    with pytest.raises(ValueError, match=r"all-False mask.*\[1\]"):
        BracketMask().transform(dist, mask)


def test_zero_mass_on_tradable_raises(prov, ids_ts):
    # Row 0: all forecast mass sits in the masked-out bracket → zero
    # mass on tradable subset.
    probs = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    mask = np.array([
        [True, True, False, True],
        [True, True, True, True],
    ])
    dist = _bracket_dist(probs, prov, ids_ts)
    with pytest.raises(ValueError, match=r"zero forecast mass.*\[0\]"):
        BracketMask().transform(dist, mask)


def test_shape_mismatch_raises(prov, ids_ts):
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    mask = np.array([[True, True, False]])
    dist = _bracket_dist(probs, prov, ids_ts)
    with pytest.raises(ValueError, match="shape"):
        BracketMask().transform(dist, mask)


def test_non_bool_mask_raises(prov, ids_ts):
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    mask = np.array([[1, 1, 0, 1]])  # int, not bool
    dist = _bracket_dist(probs, prov, ids_ts)
    with pytest.raises(TypeError, match="bool"):
        BracketMask().transform(dist, mask)


def test_non_bracket_backing_raises(prov, ids_ts):
    ids, ts = ids_ts(3)
    dist = DistributionForecast.from_normal(
        mu=np.array([0.0, 0.0, 0.0]),
        sigma=np.array([1.0, 1.0, 1.0]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    mask = np.ones((3, 4), dtype=bool)
    with pytest.raises(TypeError, match="BracketForecast"):
        BracketMask().transform(dist, mask)


def test_provenance_chain_extended(prov, ids_ts):
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    mask = np.array([[True, False, True, True]])
    dist = _bracket_dist(probs, prov, ids_ts)
    out = BracketMask().transform(dist, mask)
    assert out.provenance.conversion_chain[-1] == "BracketMask"


def test_edges_preserved(prov, ids_ts):
    probs = np.array([[0.1, 0.4, 0.3, 0.2]])
    mask = np.array([[True, False, True, True]])
    dist = _bracket_dist(probs, prov, ids_ts)
    out = BracketMask().transform(dist, mask)
    np.testing.assert_array_equal(out.edges, dist.edges)
