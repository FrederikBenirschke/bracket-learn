"""Trainer smoke tests.

One test per trainer: fits on tiny synthetic data, checks that
predict_dist returns a DistributionForecast of the expected backing/shape.
Trainers requiring heavy optional deps (NGBoost, LightGBM, quantile-forest,
torch) are skipped if the dep is not installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from bracketlearn.forecast import Backing, ParametricFamily
from bracketlearn.trainers import (
    EMOS,
    MixtureNormals,
    OnlineAggregator,
    SklearnPoint,
)


def _synthetic(n: int = 100, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    truth = 10.0 + rng.normal(0, 1.0, n)
    X = truth[:, None] + rng.normal(0, 0.5, (n, k))
    return X, truth, np.arange(n), days.astype(float)


def _skip_if_missing(modname: str) -> None:
    try:
        importlib.import_module(modname)
    except ImportError:
        pytest.skip(f"{modname} not installed")


# ---------------------------------------------------------------------------
# Native parametric.
# ---------------------------------------------------------------------------


def test_sklearn_point_predicts_pointforecast():
    from sklearn.linear_model import Ridge
    X, y, ids, ts = _synthetic()
    sp = SklearnPoint(Ridge()).fit(X, y)
    pf = sp.predict(X, ids=ids, timestamps=ts)
    assert pf.mu.shape == (X.shape[0],)


def test_emos_emits_parametric_normal():
    X, y, ids, ts = _synthetic()
    e = EMOS().fit(X, y)
    d = e.predict_dist(X, ids=ids, timestamps=ts)
    assert d.backing == Backing.PARAMETRIC
    assert d.family == ParametricFamily.NORMAL
    assert d.params["mu"].shape == (X.shape[0],)
    assert np.all(d.params["sigma"] > 0)


def test_mixture_normals_emits_mixture_normal():
    X, y, ids, ts = _synthetic(k=4)
    m = MixtureNormals().fit(X, y)
    d = m.predict_dist(X, ids=ids, timestamps=ts)
    assert d.backing == Backing.PARAMETRIC
    assert d.family == ParametricFamily.MIXTURE_NORMAL
    assert d.params["weights"].shape == (X.shape[0], X.shape[1])
    np.testing.assert_allclose(d.params["weights"].sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Online + RNN.
# ---------------------------------------------------------------------------


def test_online_aggregator_snapshots_weights():
    X, y, ids, ts = _synthetic(n=200, k=5)
    agg = OnlineAggregator(min_experts=2).fit(X, y)
    assert agg.final_w_.shape == (X.shape[1],)
    np.testing.assert_allclose(agg.final_w_.sum(), 1.0)
    pf = agg.predict(X, ids=ids, timestamps=ts)
    assert pf.mu.shape == (X.shape[0],)


def test_online_aggregator_raises_when_no_awake_rows():
    """If every row is NaN-filled, AdaHedge has no observations to update on."""
    X = np.full((50, 3), np.nan)
    y = np.zeros(50)
    with pytest.raises(RuntimeError, match="awake"):
        OnlineAggregator(min_experts=2).fit(X, y)


def test_rnn_hourly_predicts_residual():
    _skip_if_missing("torch")
    from bracketlearn.trainers import RNNHourly
    rng = np.random.default_rng(0)
    N, T, C = 60, 24, 4
    X = rng.normal(20, 5, (N, T, C)).astype(np.float32)
    y = X[:, :, 0].max(axis=1) - 0.3 * X[:, :, 2].mean(axis=1)
    rnn = RNNHourly(epochs=10, hidden=8, embed=2).fit(X, y)
    pf = rnn.predict(X, ids=np.arange(N), timestamps=np.arange(N, dtype=float))
    assert pf.mu.shape == (N,)


def test_rnn_hourly_rejects_2d_x():
    _skip_if_missing("torch")
    from bracketlearn.trainers import RNNHourly
    X = np.random.randn(50, 6).astype(np.float32)
    y = np.zeros(50)
    with pytest.raises(ValueError, match="3-D"):
        RNNHourly(epochs=1).fit(X, y)


# ---------------------------------------------------------------------------
# Optional-dep trainers (skip if missing).
# ---------------------------------------------------------------------------


def test_ngboost_normal_emits_parametric_normal():
    _skip_if_missing("ngboost")
    from bracketlearn.trainers import NGBoostNormal
    X, y, ids, ts = _synthetic(n=80)
    ng = NGBoostNormal(n_estimators=30, learning_rate=0.05, random_seed=0).fit(X, y)
    d = ng.predict_dist(X, ids=ids, timestamps=ts)
    assert d.backing == Backing.PARAMETRIC
    assert d.family == ParametricFamily.NORMAL


def test_quantile_reg_emits_quantile_backing():
    _skip_if_missing("lightgbm")
    from bracketlearn.trainers import QuantileReg
    X, y, ids, ts = _synthetic(n=80)
    qr = QuantileReg(n_estimators=20, random_seed=0).fit(X, y)
    d = qr.predict_dist(X, ids=ids, timestamps=ts)
    assert d.backing == Backing.QUANTILE
    assert d.qvals.shape[0] == X.shape[0]


def test_quantile_forest_emits_quantile_backing():
    _skip_if_missing("quantile_forest")
    from bracketlearn.trainers import QuantileForest
    X, y, ids, ts = _synthetic(n=80)
    qf = QuantileForest(n_estimators=20, random_seed=0).fit(X, y)
    d = qf.predict_dist(X, ids=ids, timestamps=ts)
    assert d.backing == Backing.QUANTILE


def test_cumulative_binary_emits_bracket():
    _skip_if_missing("lightgbm")
    from bracketlearn.trainers import CumulativeBinary
    X, y, ids, ts = _synthetic(n=80)
    cb = CumulativeBinary(cutpoints=np.array([8.0, 10.0, 12.0]),
                          outer_edges=(0.0, 20.0),
                          n_estimators=20).fit(X, y)
    d = cb.predict_dist(X, ids=ids, timestamps=ts)
    assert d.backing == Backing.BRACKET
    np.testing.assert_allclose(d.probs.sum(axis=1), 1.0)
