"""``PipelineResult.score`` must slice ``edges`` alongside the OOF ``y``.

A stage's out-of-fold coverage is a subset of the rows: under a walk-forward
the first fold's train rows are never predicted, and a stage added partway
through a chain covers fewer rows still. ``y`` is indexed by ``dist.ids``, and
a per-row ladder has to be indexed the same way or the two disagree about
which row is which.

Passing the full-length ladder against a sliced ``y`` raised "edges describe N
rows; the forecast has M", which made the per-row shape unusable through this
API entirely. ``_slice_edges`` was added in the same commit as the value-path
fix and, like it, shipped with no test: reverting it left the whole suite
green.

The shared 1-D vector is the case that must NOT be indexed. It describes every
row by construction, so slicing it would silently truncate a (B+1,) ladder to
len(idx) edges and change how many brackets the caller appears to have.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.pipeline import _slice_edges


def test_shared_vector_passes_through_untouched():
    """A 1-D ladder describes every row; indexing it would corrupt it."""
    edges = np.array([-np.inf, 60.0, 62.0, 64.0, np.inf])
    idx = np.array([3, 1, 7])
    out = _slice_edges(edges, idx)
    assert out is edges or np.array_equal(out, edges)
    assert np.asarray(out).shape == (5,), (
        "the shared vector was indexed, so the caller now has 3 edges where "
        "the ladder has 5")


def test_shared_vector_given_as_a_list_is_also_left_alone():
    """A 1-D ladder handed in as a plain list takes the same path."""
    edges = [-np.inf, 60.0, 62.0, 64.0, np.inf]
    out = _slice_edges(edges, np.array([2, 0]))
    assert np.asarray(out).shape == (5,)
    assert np.array_equal(np.asarray(out), np.asarray(edges, dtype=float))


def test_dense_grid_is_indexed_by_row():
    edges = np.arange(40.0).reshape(8, 5)
    idx = np.array([5, 0, 3])
    out = _slice_edges(edges, idx)
    assert out.shape == (3, 5)
    assert np.array_equal(out, edges[idx])


def test_ragged_sequence_is_indexed_by_row_and_keeps_its_widths():
    edges = [
        np.array([-np.inf, 10.0, 12.0, np.inf]),
        np.array([-np.inf, 20.0, 22.0, 24.0, np.inf]),
        np.array([-np.inf, 30.0, 32.0, np.inf]),
        np.array([-np.inf, 40.0, 42.0, 44.0, 46.0, np.inf]),
    ]
    out = _slice_edges(edges, np.array([3, 0]))
    assert len(out) == 2
    assert np.array_equal(out[0], edges[3])
    assert np.array_equal(out[1], edges[0])


def test_none_passes_through():
    assert _slice_edges(None, np.array([0, 1])) is None


def test_order_follows_idx_not_the_original_order():
    """idx is dist.ids, which is not required to be sorted."""
    edges = np.arange(25.0).reshape(5, 5)
    idx = np.array([4, 2, 0])
    out = _slice_edges(edges, idx)
    assert np.array_equal(out[0], edges[4])
    assert np.array_equal(out[1], edges[2])
    assert np.array_equal(out[2], edges[0])


def test_score_on_a_partial_oof_stage_with_a_per_row_ladder():
    """End to end: the failure this helper exists to prevent.

    A stage whose OOF coverage is a strict subset of the rows, scored with a
    per-row ladder covering ALL rows. Without the slice the ladder and the
    forecast disagree on row count and the call raises.
    """
    from bracketlearn.compose import WalkForward
    from bracketlearn.trainers import EMOS

    rng = np.random.default_rng(0)
    n = 60
    X = np.column_stack([rng.normal(60, 10, n), rng.uniform(1, 3, n)])
    y = X[:, 0] + rng.normal(0, 2, n)
    ids = np.arange(n)
    ts = np.arange(n, dtype=float)

    res = WalkForward(cv="expanding-window", n_folds=3).fit_predict(
        EMOS(fit_method="crps_nelder_mead"), X, y, ids=ids, timestamps=ts)

    covered = next(iter(res.forecasts.values())).ids.shape[0]
    assert covered < n, (
        "this fixture no longer has partial OOF coverage, so it cannot "
        "demonstrate the slicing requirement")

    edges = [np.array([-np.inf, m - 2, m, m + 2, np.inf]) for m in X[:, 0]]
    scores = res.score(y, metrics=["brier_bracket"], edges=edges)
    for name, row in scores.items():
        assert np.isfinite(row["brier_bracket"]), name
        assert row["n_oof"] == covered


def test_score_rejects_a_ladder_that_is_short_for_the_full_row_set():
    """The slice indexes by row id, so a ladder must cover every row.

    A per-row ladder shorter than the data is a caller error and must raise
    rather than silently score whichever rows happen to line up.
    """
    from bracketlearn.compose import WalkForward
    from bracketlearn.trainers import EMOS

    rng = np.random.default_rng(0)
    n = 40
    X = np.column_stack([rng.normal(60, 10, n), rng.uniform(1, 3, n)])
    y = X[:, 0] + rng.normal(0, 2, n)
    res = WalkForward(cv="expanding-window", n_folds=3).fit_predict(
        EMOS(fit_method="crps_nelder_mead"), X, y,
        ids=np.arange(n), timestamps=np.arange(n, dtype=float))

    short = [np.array([-np.inf, m - 2, m, m + 2, np.inf]) for m in X[:5, 0]]
    with pytest.raises((ValueError, IndexError)):
        res.score(y, metrics=["brier_bracket"], edges=short)
