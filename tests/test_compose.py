"""WalkForward + Stacker (the object-graph composition surface).

Covers shared-upstream dedup, nested stackers, multi-root leaderboards,
duplicate-name rejection, the refit/predict path, and Stacker construction
guards.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from bracketlearn import Pipeline, Stacker, WalkForward
from bracketlearn.lift import GlobalResidual
from bracketlearn.trainers import EMOS, SklearnPoint, StackedParametric, BMAStacking


def _synthetic(n: int = 200, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    truth = 10.0 + 0.05 * days + rng.normal(0, 1.0, n)
    X = truth[:, None] + rng.normal(0, 0.5, (n, k))
    ids = np.arange(n)
    ts = days.astype(float)
    return X, truth, ids, ts


def test_shared_upstream_computed_once_and_nested_stacker_runs():
    # emos reused by two stackers, then a stack-of-stacks. Exercises object
    # dedup + recursion; result must address every named node.
    X, y, ids, ts = _synthetic(n=160)
    emos = Pipeline([EMOS()], name="emos")
    ridge = Pipeline([SklearnPoint(LinearRegression()), GlobalResidual()], name="ridge")
    bma = Stacker([emos, ridge], BMAStacking(), name="bma")
    stk = Stacker([emos, ridge], StackedParametric(), name="stk")
    sup = Stacker([bma, stk], BMAStacking(), name="super")
    res = WalkForward(n_folds=3).fit_predict(sup, X, y, ids=ids, timestamps=ts)
    for nm in ("emos", "ridge", "bma", "stk", "super"):
        assert nm in res.forecasts, nm
        assert res[nm].ids.shape[0] > 0


def test_walkforward_predict_on_unseen_rows():
    """After refit_on_full, predict() emits a dist for every node on truly
    unseen rows, and the meta consumes its upstreams' fresh predictions."""
    X, y, ids, ts = _synthetic(n=220)
    Xtr, ytr, idtr, tstr = X[:180], y[:180], ids[:180], ts[:180]
    Xte, idte, tste = X[180:], ids[180:], ts[180:]

    ridge = Pipeline([SklearnPoint(LinearRegression()), GlobalResidual()], name="ridge")
    emos = Pipeline([EMOS()], name="emos")
    stack = Stacker([ridge, emos], StackedParametric(), name="stack")
    wf = WalkForward(n_folds=3, refit_on_full=True)
    wf.fit_predict(stack, Xtr, ytr, ids=idtr, timestamps=tstr)
    pred = wf.predict(Xte, ids=idte, timestamps=tste)

    assert set(pred) == {"ridge", "emos", "stack"}
    n_te = Xte.shape[0]
    for nm in ("ridge", "emos", "stack"):
        assert pred[nm].params["mu"].shape == (n_te,)
        assert np.all(pred[nm].params["sigma"] > 0)
    # The stack's μ is a (calibration-free) affine blend of its upstreams'
    # μ — it must sit within their row-wise envelope plus a small margin.
    lo = np.minimum(pred["ridge"].params["mu"], pred["emos"].params["mu"])
    hi = np.maximum(pred["ridge"].params["mu"], pred["emos"].params["mu"])
    span = hi - lo + 1.0
    assert np.all(pred["stack"].params["mu"] >= lo - 2 * span)
    assert np.all(pred["stack"].params["mu"] <= hi + 2 * span)


def test_predict_requires_refit():
    X, y, ids, ts = _synthetic(n=120)
    wf = WalkForward(n_folds=3)  # refit_on_full=False
    wf.fit_predict(Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts)
    with pytest.raises(RuntimeError, match="refit_on_full=True"):
        wf.predict(X, ids=ids, timestamps=ts)


def test_walkforward_multi_root_outputs_and_dedup():
    # A list of independent models = multiple leaderboard outputs; a stacker
    # sharing one of them must NOT duplicate it (object identity dedup).
    X, y, ids, ts = _synthetic(n=160)
    emos = Pipeline([EMOS()], name="emos")
    ridge = Pipeline([SklearnPoint(LinearRegression()), GlobalResidual()], name="ridge")
    stack = Stacker([emos, ridge], StackedParametric(), name="stack")
    res = WalkForward(n_folds=3).fit_predict([emos, ridge, stack], X, y, ids=ids, timestamps=ts)
    assert set(res.forecasts) == {"emos", "ridge", "stack"}


def test_walkforward_rejects_duplicate_node_names():
    X, y, ids, ts = _synthetic(n=120)
    a = Pipeline([EMOS()], name="dup")
    b = Pipeline([EMOS()], name="dup")
    with pytest.raises(ValueError, match="duplicate node name"):
        WalkForward(n_folds=3).fit_predict([a, b], X, y, ids=ids, timestamps=ts)


def test_stacker_requires_upstreams_and_meta():
    with pytest.raises(ValueError, match="at least one upstream"):
        Stacker([], StackedParametric())
    with pytest.raises(ValueError, match="meta-combiner"):
        Stacker([Pipeline([EMOS()])], None)
