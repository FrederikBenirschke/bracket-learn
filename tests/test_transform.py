"""GroupByZScore transformer: per-group scale learning + z-round-trip.

Self-contained (no parent-repo deps). The polars-vs-numpy parity check lives
in the parent weather test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn import GroupByZScore, NormalForecast
from bracketlearn.forecast._meta import ProvenanceMeta


def test_fit_learns_per_group_std_of_anomaly():
    # Two groups with ≥5 obs each → per-group scale = std(y − center, ddof=0).
    ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    center = np.array([10.0] * 5 + [60.0] * 5)
    y = center + np.array([1.0, -1.0, 2.0, -2.0, 0.0, 3.0, -3.0, 1.0, -1.0, 0.0])
    gz = GroupByZScore(min_group=5).fit(np.zeros((10, 1)), y, ids=ids, center=center)
    anom = y - center
    assert gz.scale_by_[0] == pytest.approx(np.std(anom[:5], ddof=0))
    assert gz.scale_by_[1] == pytest.approx(np.std(anom[5:], ddof=0))
    assert gz.scale_global_ == pytest.approx(np.std(anom, ddof=0))


def test_small_group_falls_back_to_global_scale():
    ids = np.array([0, 0, 0, 0, 0, 1])          # group 1 has 1 obs (< min_group)
    center = np.zeros(6)
    y = np.array([1.0, -1.0, 2.0, -2.0, 0.0, 99.0])
    gz = GroupByZScore(min_group=5).fit(np.zeros((6, 1)), y, ids=ids, center=center)
    assert gz.scale_by_[1] == gz.scale_global_     # explicit fallback, never 1.0


def test_transform_levels_spreads_passthrough():
    ids = np.array([0, 0, 0, 0, 0])
    center = np.full(5, 10.0)
    y = center + np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    gz = GroupByZScore(spread_cols=(1,), passthrough_cols=(2,)).fit(
        np.zeros((5, 1)), y, ids=ids, center=center,
    )
    s = gz.scale_by_[0]
    X = np.column_stack([
        np.full(5, 14.0),    # level → (14 − 10)/s
        np.full(5, 3.0),     # spread → 3/s
        np.full(5, 1.0),     # passthrough → 1
    ])
    Xz = gz.transform(X, ids=ids, center=center)
    np.testing.assert_allclose(Xz[:, 0], (14.0 - 10.0) / s)
    np.testing.assert_allclose(Xz[:, 1], 3.0 / s)
    np.testing.assert_allclose(Xz[:, 2], 1.0)


def test_level_cols_empty_is_target_only():
    # level_cols=() → every feature column passes through unchanged, but the
    # target + forecast are still z-scored (the target-only mode used by the
    # vendor-X bracket trainers whose feature roles aren't known by index).
    ids = np.array([0, 0, 0, 0, 0])
    center = np.full(5, 10.0)
    y = center + np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    gz = GroupByZScore(level_cols=()).fit(
        np.zeros((5, 3)), y, ids=ids, center=center,
    )
    s = gz.scale_by_[0]
    X = np.column_stack([np.full(5, 14.0), np.full(5, 3.0), np.full(5, 1.0)])
    Xz = gz.transform(X, ids=ids, center=center)
    np.testing.assert_allclose(Xz, X)               # all features untouched
    # target still standardized
    np.testing.assert_allclose(gz.transform_target(y), (y - center) / s)


def test_level_cols_explicit_subset():
    # Only the named index is a level; other non-spread cols pass through.
    ids = np.array([0, 0, 0, 0, 0])
    center = np.full(5, 10.0)
    y = center + np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    gz = GroupByZScore(level_cols=(0,)).fit(
        np.zeros((5, 2)), y, ids=ids, center=center,
    )
    s = gz.scale_by_[0]
    X = np.column_stack([np.full(5, 14.0), np.full(5, 7.0)])
    Xz = gz.transform(X, ids=ids, center=center)
    np.testing.assert_allclose(Xz[:, 0], (14.0 - 10.0) / s)   # explicit level
    np.testing.assert_allclose(Xz[:, 1], 7.0)                 # implicit passthrough


def test_target_and_dist_roundtrip_to_original_scale():
    ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    center = np.array([10.0] * 5 + [60.0] * 5)
    y = center + np.array([1.0, -1.0, 2.0, -2.0, 0.5, 3.0, -3.0, 1.0, -1.0, 0.2])
    gz = GroupByZScore().fit(np.zeros((10, 1)), y, ids=ids, center=center)
    gz.transform(np.zeros((10, 1)), ids=ids, center=center)   # stamp (c, s)
    s = gz._scale
    # target → z, then undo by hand == original
    yz = gz.transform_target(y)
    np.testing.assert_allclose(yz * s + center, y, atol=1e-9)
    # inverse_dist: a z-space NormalForecast → °F via affine(center, scale)
    muz, sigz = np.full(10, 0.3), np.full(10, 1.1)
    dz = NormalForecast.from_arrays(
        mu=muz, sigma=sigz, ids=np.arange(10), timestamps=np.zeros(10),
        provenance=ProvenanceMeta.placeholder("t"),
    )
    out = gz.inverse_dist(dz)
    np.testing.assert_allclose(out.mu, muz * s + center, atol=1e-9)
    np.testing.assert_allclose(out.sigma, sigz * s, atol=1e-9)


def test_degenerate_global_scale_raises():
    ids = np.zeros(4, dtype=int)
    center = np.zeros(4)
    y = np.full(4, 5.0)                    # zero variance → scale 0 → raise
    with pytest.raises(ValueError, match="finite-positive"):
        GroupByZScore().fit(np.zeros((4, 1)), y, ids=ids, center=center)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        GroupByZScore().transform(np.zeros((2, 1)), ids=np.zeros(2))
