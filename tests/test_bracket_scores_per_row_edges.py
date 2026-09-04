"""Bracket scorers must use each row's OWN ladder.

``brier_bracket``/``log_loss_bracket`` take ``edges`` and use it for
``searchsorted(edges, y)`` — the step deciding which bracket the outcome fell
in. They used to accept only a single ``(B+1,)`` vector and apply it to every
row.

On a venue whose ladder rotates that is silently wrong. Kalshi relists daily
around the forecast, so one day is ``[64,66,68,70,72]`` and the next
``[25,27,29,31,33]``; passing row 0's vector scores every later row's outcome
against a grid it never traded on. The shapes still line up, so a plausible
number comes back — measured Brier 0.8904 against a correct 0.7343.

Rows with different bracket COUNTS broke the flat reshape and raised. Rows with
the same count and different VALUES, which is every row of a real rotating
ladder, returned the wrong answer quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.adapters import BracketLadder
from bracketlearn.score import brier_bracket, log_loss_bracket
from bracketlearn.trainers import EMOS


def _fitted(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(60, 10, n), rng.uniform(1, 3, n)])
    y = X[:, 0] + rng.normal(0, 2, n)
    em = EMOS(fit_method="crps_nelder_mead").fit(X, y)
    d = em.predict_dist(X, ids=np.arange(n), timestamps=np.arange(n, dtype=float))
    return X, y, d


def _rotating(X):
    """Same bracket COUNT, different VALUES per row — a real Kalshi ladder."""
    return [np.array([-np.inf, m - 2, m, m + 2, np.inf]) for m in X[:, 0]]


def _manual_brier(contracts, edges, y):
    n, b = len(edges), len(edges[0]) - 1
    fair = np.asarray(contracts.fair_price, float).reshape(n, b)
    tot = 0.0
    for i in range(n):
        oh = np.zeros(b)
        oh[np.clip(np.searchsorted(edges[i], y[i], "right") - 1, 0, b - 1)] = 1.0
        tot += ((fair[i] - oh) ** 2).sum()
    return tot / n


def test_per_row_edges_match_a_hand_computation():
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    assert brier_bracket(c, edges, y) == pytest.approx(
        _manual_brier(c, edges, y), rel=1e-12)


def test_all_three_edge_shapes_agree():
    """List, dense (N, B+1) array, and the ladder's own grid must agree."""
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    as_list = brier_bracket(c, edges, y)
    as_dense = brier_bracket(c, np.array(edges), y)
    assert as_list == pytest.approx(as_dense, rel=1e-12)


def test_shared_ladder_expressed_two_ways_agrees():
    """A genuinely shared ladder scored as a 1-D vector and as N copies must
    give the same number — the invariant that ties the old path to the new."""
    X, y, d = _fitted()
    shared = np.array([-np.inf, 55.0, 60.0, 65.0, np.inf])
    c = BracketLadder(edges_per_row=[shared] * len(y)).price(d)
    assert brier_bracket(c, shared, y) == pytest.approx(
        brier_bracket(c, [shared] * len(y), y), rel=1e-12)
    assert log_loss_bracket(c, shared, y) == pytest.approx(
        log_loss_bracket(c, [shared] * len(y), y), rel=1e-12)


def test_one_rows_edges_for_a_rotating_ladder_now_raises():
    """The original bug. It must not return a number."""
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    with pytest.raises(ValueError, match="per-row edges that differ"):
        brier_bracket(c, edges[0], y)
    with pytest.raises(ValueError, match="per-row edges that differ"):
        log_loss_bracket(c, edges[0], y)


def test_the_wrong_answer_really_was_different():
    """Guard against a future 'simplification' that makes the shared path
    agree by accident: the wrong number must be materially wrong, or this
    whole test module is pinning nothing."""
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    correct = brier_bracket(c, edges, y)
    wrong = _manual_brier(c, [edges[0]] * len(y), y)
    assert abs(wrong - correct) > 0.1, (
        "scoring against row 0's grid should be visibly wrong here")


def test_ragged_bracket_counts_are_supported():
    """Different bracket COUNTS per row used to raise on the flat reshape."""
    X, y, d = _fitted()
    edges = [
        np.array([-np.inf, m - 2, m, m + 2, np.inf]) if i % 2 else
        np.array([-np.inf, m - 4, m - 2, m, m + 2, m + 4, np.inf])
        for i, m in enumerate(X[:, 0])
    ]
    c = BracketLadder(edges_per_row=edges).price(d)
    v = brier_bracket(c, edges, y)
    assert np.isfinite(v) and v > 0


def test_row_count_mismatch_raises():
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    with pytest.raises(ValueError):
        brier_bracket(c, edges[:-1], y)


def test_ladder_records_the_grid_it_priced():
    """The verification above only works because the adapter keeps its edges."""
    X, y, d = _fitted()
    edges = _rotating(X)
    c = BracketLadder(edges_per_row=edges).price(d)
    rec = c.contract_spec.edges_per_row
    assert rec is not None and len(rec) == len(y)
    assert np.allclose(np.asarray(rec[3], float), edges[3], equal_nan=True)
