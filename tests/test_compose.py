"""WalkForward + Stacker (new clean surface) parity vs ForecastPipeline.

The object-graph surface must reproduce the legacy name-keyed surface
bit-for-bit: same folds, same inner lifter split, same in-sample deps fed to
the meta, same OOF dists. Any drift is a bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression

from bracketlearn import Pipeline, Stacker, WalkForward, ForecastPipeline
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import LiftedForecaster
from bracketlearn.trainers import EMOS, SklearnPoint, StackedParametric, BMAStacking


def _synthetic(n: int = 200, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    truth = 10.0 + 0.05 * days + rng.normal(0, 1.0, n)
    X = truth[:, None] + rng.normal(0, 0.5, (n, k))
    ids = np.arange(n)
    ts = days.astype(float)
    return X, truth, ids, ts


def _aligned(dist):
    """Return (mu, sigma) sorted by ids so two surfaces compare row-for-row."""
    o = np.argsort(dist.ids)
    return dist.ids[o], dist.params["mu"][o], dist.params["sigma"][o]


def test_walkforward_bare_pipeline_matches_forecastpipeline():
    X, y, ids, ts = _synthetic()
    old = ForecastPipeline(
        steps=[("emos", EMOS())], n_folds=4, refit_on_full=False,
    ).fit_predict(X, y, ids=ids, timestamps=ts)
    new = WalkForward(n_folds=4).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    ia, ma, sa = _aligned(old["emos"])
    ib, mb, sb = _aligned(new["emos"])
    assert_allclose(ia, ib)
    assert_allclose(ma, mb, rtol=1e-9, atol=1e-9)
    assert_allclose(sa, sb, rtol=1e-9, atol=1e-9)


def test_walkforward_stacker_matches_forecastpipeline():
    X, y, ids, ts = _synthetic()
    # Legacy name-keyed surface.
    old = ForecastPipeline(
        steps=[
            ("ridge", LiftedForecaster(
                base=SklearnPoint(LinearRegression()),
                lifter=GlobalResidual(),
                name="ridge",
            )),
            ("emos", EMOS()),
            ("stack", StackedParametric(deps=("ridge", "emos"))),
        ],
        n_folds=3, refit_on_full=False,
    ).fit_predict(X, y, ids=ids, timestamps=ts)

    # New object-graph surface — same model, no names-as-wiring.
    ridge = Pipeline([SklearnPoint(LinearRegression()), GlobalResidual()], name="ridge")
    emos = Pipeline([EMOS()], name="emos")
    stack = Stacker([ridge, emos], StackedParametric(), name="stack")
    new = WalkForward(n_folds=3).fit_predict(stack, X, y, ids=ids, timestamps=ts)

    for nm in ("ridge", "emos", "stack"):
        ia, ma, sa = _aligned(old[nm])
        ib, mb, sb = _aligned(new[nm])
        assert_allclose(ia, ib, err_msg=f"{nm}: ids")
        assert_allclose(ma, mb, rtol=1e-9, atol=1e-9, err_msg=f"{nm}: mu")
        assert_allclose(sa, sb, rtol=1e-9, atol=1e-9, err_msg=f"{nm}: sigma")


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


def test_stacker_requires_upstreams_and_meta():
    with pytest.raises(ValueError, match="at least one upstream"):
        Stacker([], StackedParametric())
    with pytest.raises(ValueError, match="meta-combiner"):
        Stacker([Pipeline([EMOS()])], None)
