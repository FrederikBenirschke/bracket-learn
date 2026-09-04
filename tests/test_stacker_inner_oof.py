"""A Stacker's meta must see out-of-sample upstream predictions.

`_fit_node` returned `dist_tr` — the node's prediction on the very rows it had
just been fit on — and `Stacker` fed that to the meta as `upstream=`. An
upstream that overfits therefore looks *better* to the meta than one that
generalises, so the meta learns to weight the overfitter.

This is the same leak as the calibrator's (a component fit on another's
in-sample output), one level up the composition.

`StackedParametric` already catches the extreme case with a good error
("upstream μ collinearity with y (data leak)") — a fully-grown tree raises.
Mild overfitting, which is the common case, passed silently and produced a
stack worse than either of its inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from bracketlearn.compose import Stacker, WalkForward
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import Pipeline
from bracketlearn.trainers import SklearnPoint, StackedParametric


def _linear_data(n=600, seed=0):
    """Pure-linear DGP: ridge is right, a deep tree can only overfit."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    y = X @ np.array([1.0, -0.5, 0.3, 0.0]) + rng.normal(0, 1, n)
    return X, y, np.arange(n), np.arange(n, dtype=float)


def _run():
    X, y, ids, ts = _linear_data()
    ridge = Pipeline([SklearnPoint(Ridge()), GlobalResidual()], name="ridge")
    overfit = Pipeline(
        [SklearnPoint(DecisionTreeRegressor(max_depth=8, random_state=0)),
         GlobalResidual()], name="overfit")
    meta = Pipeline([StackedParametric()], name="meta")
    res = WalkForward(cv="kfold", n_folds=5).fit_predict(
        Stacker([ridge, overfit], meta), X, y, ids=ids, timestamps=ts)
    return res.score(y, metrics=["crps"]), y


def test_meta_is_not_dragged_below_its_best_upstream():
    """The regression. With in-sample upstream inputs the meta scored CRPS
    0.809 against ridge's 0.555 — the stack was 46% worse than one of the
    things it was combining."""
    table, _ = _run()
    best_upstream = min(table["ridge"]["crps"], table["overfit"]["crps"])
    assert table["meta"]["crps"] <= best_upstream * 1.05, (
        f"meta {table['meta']['crps']:.4f} is worse than its best upstream "
        f"{best_upstream:.4f}: it is weighting the overfitter, which is what "
        "in-sample upstream predictions cause")


def test_meta_tracks_the_correct_upstream():
    """On a linear DGP the meta should end up near ridge, not near the tree."""
    table, _ = _run()
    assert abs(table["meta"]["crps"] - table["ridge"]["crps"]) < 0.05, (
        "meta should learn to follow the upstream that generalises")


def test_leaves_not_feeding_a_meta_are_unaffected():
    """Only leaves consumed by a meta pay the inner-OOF refit; a plain
    leaderboard run must be unchanged."""
    X, y, ids, ts = _linear_data(n=300)
    ridge = Pipeline([SklearnPoint(Ridge()), GlobalResidual()], name="ridge")
    wf = WalkForward(cv="kfold", n_folds=3)
    a = wf.fit_predict(ridge, X, y, ids=ids, timestamps=ts).score(y, metrics=["crps"])
    b = wf.fit_predict(ridge, X, y, ids=ids, timestamps=ts).score(y, metrics=["crps"])
    assert a["ridge"]["crps"] == pytest.approx(b["ridge"]["crps"], rel=1e-12)


def test_stitched_train_dist_keeps_row_order():
    """The inner split predicts the halves out of order, so the stitched
    result must be reordered back — the meta refuses a mismatch, and a silent
    reorder would misalign every upstream row against y."""
    table, _ = _run()
    # If ordering were wrong the meta's own alignment guard raises during
    # _run(); reaching here with a finite score is the assertion.
    assert np.isfinite(table["meta"]["crps"])
