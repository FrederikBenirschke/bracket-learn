"""Tests for normalize_bracket_probs — the 'valid distribution' primitive.

Pins invariants:
- 1-D input → 1-D output summing to 1
- 2-D input → per-row sum-1
- zero/negative total mass raises with source name + row indices
- negative entries raise (no silent clip)
- wrong-dim input raises
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn import normalize_bracket_probs


def test_1d_basic():
    out = normalize_bracket_probs(
        np.array([0.1, 0.4, 0.3, 0.2]), source="test",
    )
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(out, [0.1, 0.4, 0.3, 0.2], atol=1e-12)


def test_1d_overround():
    # YES prices with overround: 0.30 + 0.45 + 0.30 + 0.10 = 1.15
    raw = np.array([0.30, 0.45, 0.30, 0.10])
    out = normalize_bracket_probs(raw, source="test")
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(out, raw / raw.sum(), atol=1e-12)


def test_2d_per_row():
    raw = np.array([
        [0.30, 0.45, 0.30, 0.10],  # overround
        [0.10, 0.10, 0.10, 0.05],  # underround
    ])
    out = normalize_bracket_probs(raw, source="test")
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_zero_total_raises_1d():
    with pytest.raises(ValueError, match=r"my_source.*≤ 0"):
        normalize_bracket_probs(np.zeros(4), source="my_source")


def test_zero_total_raises_2d_reports_row_index():
    raw = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.0, 0.0, 0.0, 0.0],   # offending
        [0.10, 0.40, 0.30, 0.20],
    ])
    with pytest.raises(ValueError, match=r"my_source.*\[1\]"):
        normalize_bracket_probs(raw, source="my_source")


def test_negative_entries_raise():
    raw = np.array([0.1, -0.05, 0.4, 0.3])
    with pytest.raises(ValueError, match="negative"):
        normalize_bracket_probs(raw, source="test")


def test_wrong_dim_raises():
    with pytest.raises(ValueError, match=r"1-D or 2-D"):
        normalize_bracket_probs(np.zeros((2, 3, 4)), source="test")


def test_idempotent_on_valid():
    valid = np.array([0.1, 0.4, 0.3, 0.2])
    out1 = normalize_bracket_probs(valid, source="test")
    out2 = normalize_bracket_probs(out1, source="test")
    np.testing.assert_allclose(out1, out2, atol=1e-12)
