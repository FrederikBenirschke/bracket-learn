"""The VALUE scorers must use each row's own ladder, like the accuracy ones.

``brier_bracket``/``log_loss_bracket`` were fixed to accept per-row ``edges``
and to refuse a single shared vector when the ladder was priced per-row.
``_qmr_from_bracket``, which feeds ``edge_alignment_bracket`` and
``value_report_bracket``, was fixed in the same commit and got no test. It
re-injected silently: reverting it to the 1-D-only form left the whole suite
green while EA moved from +0.006138 to -0.013380 on a rotating ladder.

That is the more damaging of the two paths. Brier is a magnitude, so a wrong
ladder gives a wrong number that still reads as a bad score. EA is signed, and
its sign IS the verdict: positive means the forecast's deviations from the
reference point the right way, negative means they point the wrong way.

How the error presents depends on the reference ladder. On the fixture below
it is a ~114x magnitude error with the sign preserved; the audit that found
this measured a case where the sign inverted outright. Only the magnitude is
asserted here, because that is what this fixture actually demonstrates, and a
test should pin what it shows rather than the worst case it has heard of.

These tests exist because the sibling path was the one that mattered, and the
tests written alongside the fix covered only the path that was already safe.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.adapters import BracketLadder
from bracketlearn.score import (
    edge_alignment_bracket,
    value_report_bracket,
)
from bracketlearn.trainers import EMOS


def _fitted(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(60, 10, n), rng.uniform(1, 3, n)])
    y = X[:, 0] + rng.normal(0, 2, n)
    em = EMOS(fit_method="crps_nelder_mead").fit(X, y)
    d = em.predict_dist(X, ids=np.arange(n), timestamps=np.arange(n, dtype=float))
    return X, y, d


def _rotating(X):
    """Same bracket COUNT, different VALUES per row, a real Kalshi ladder."""
    return [np.array([-np.inf, m - 2, m, m + 2, np.inf]) for m in X[:, 0]]


def _reference(contracts, seed=1):
    """A reference ladder that is not the model, normalised per row."""
    rng = np.random.default_rng(seed)
    m = np.asarray(contracts.fair_price, dtype=float).copy()
    m = np.clip(m + rng.normal(0, 0.05, m.shape), 1e-6, 1 - 1e-6)
    return m


def _manual_ea(contracts, reference, edges, y):
    """EA = mean over every (row, bracket) contract of (q - m)(r - m)."""
    n, b = len(edges), len(edges[0]) - 1
    q = np.asarray(contracts.fair_price, float).reshape(n, b)
    m = np.asarray(reference, float).reshape(n, b)
    acc = []
    for i in range(n):
        r = np.zeros(b)
        r[np.clip(np.searchsorted(edges[i], y[i], "right") - 1, 0, b - 1)] = 1.0
        acc.append((q[i] - m[i]) * (r - m[i]))
    return float(np.mean(np.concatenate(acc)))


def test_edge_alignment_matches_a_hand_computation_on_a_rotating_ladder():
    """The per-row path is not merely self-consistent, it is arithmetically right."""
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    m = _reference(c)
    assert edge_alignment_bracket(c, m, edges, y) == pytest.approx(
        _manual_ea(c, m, edges, y), rel=1e-12)


def test_one_rows_edges_for_a_rotating_ladder_is_refused():
    """Passing row 0's vector must raise, not return a plausible number.

    This is the guard the accuracy scorers got. Without it the caller receives
    a number computed against a grid 39 of the 40 rows never traded on.
    """
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    m = _reference(c)
    with pytest.raises(ValueError, match="edges"):
        edge_alignment_bracket(c, m, edges[0], y)


def test_the_wrong_ladder_really_would_have_returned_a_different_answer():
    """Pin the consequence, so the guard above cannot be dismissed as pedantry.

    Scoring against row 0's ladder is what the pre-v0.8 code did internally.
    Reproduced here by pricing a SECOND ladder in which every row genuinely
    carries row 0's edges: same shapes, same call, a different answer.

    On this fixture both answers are positive and the error is one of
    magnitude, EA +0.0013 against +0.1496, a factor of ~114. The sign can
    also flip, and does on other reference ladders, but that depends on the
    reference rather than on the mechanism, so it is not asserted here. The
    magnitude alone is disqualifying: EA is quoted in cents per contract and
    a 100x error is not a number anyone can act on.
    """
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    m = _reference(c)
    right = edge_alignment_bracket(c, m, edges, y)

    shared = [edges[0].copy() for _ in edges]
    c_shared = BracketLadder(edges_per_row=shared).price(d)
    wrong = edge_alignment_bracket(c_shared, m, shared, y)

    assert right != pytest.approx(wrong, rel=1e-6), (
        "per-row and shared ladders returned the same number, so this test "
        "no longer demonstrates anything")
    assert abs(wrong) > 10 * abs(right), (
        "the shared-ladder answer should be wrong by orders of magnitude; "
        f"got right={right:+.6f} wrong={wrong:+.6f}")


def test_value_report_uses_per_row_edges_too():
    """value_report_bracket shares _qmr_from_bracket, so it inherits the fix."""
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    m = _reference(c)
    rep = value_report_bracket(c, m, edges, y)
    assert rep["EA"] == pytest.approx(
        edge_alignment_bracket(c, m, edges, y), rel=1e-12)
    with pytest.raises(ValueError, match="edges"):
        value_report_bracket(c, m, edges[0], y)


def test_ragged_bracket_counts_are_supported_by_the_value_path():
    """Rows with different bracket COUNTS, the loud half of the original bug."""
    X, y, d = _fitted(n=12)
    edges = [
        np.array([-np.inf, m - 2, m, m + 2, np.inf]) if i % 2 else
        np.array([-np.inf, m - 2, m + 2, np.inf])
        for i, m in enumerate(X[:, 0])
    ]
    c = BracketLadder(edges_per_row=edges).price(d)
    m_ref = _reference(c)
    v = edge_alignment_bracket(c, m_ref, edges, y)
    assert np.isfinite(v)
    assert v == pytest.approx(_manual_ea_ragged(c, m_ref, edges, y), rel=1e-12)


def _manual_ea_ragged(contracts, reference, edges, y):
    q = np.asarray(contracts.fair_price, float)
    m = np.asarray(reference, float)
    acc, off = [], 0
    for i, e in enumerate(edges):
        b = len(e) - 1
        qi, mi = q[off:off + b], m[off:off + b]
        r = np.zeros(b)
        r[np.clip(np.searchsorted(e, y[i], "right") - 1, 0, b - 1)] = 1.0
        acc.append((qi - mi) * (r - mi))
        off += b
    return float(np.mean(np.concatenate(acc)))
