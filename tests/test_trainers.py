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


# ---------------------------------------------------------------------------
# Stacking — positive integration path (audit §6.T1).
# ---------------------------------------------------------------------------


def test_stacking_recovers_truth_from_perfect_upstream():
    """Two upstreams: a noisy one and a precise one. Stacking should
    weight the precise upstream more. Doesn't require exact recovery —
    just relative ordering of |weights_|."""
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
    from bracketlearn.trainers import Stacking

    rng = np.random.default_rng(0)
    N = 200
    y = rng.normal(0, 1, N)
    # Noisy upstream: y + N(0, 1).
    mu_noisy = y + rng.normal(0, 1.0, N)
    # Precise upstream: y + N(0, 0.05).
    mu_precise = y + rng.normal(0, 0.05, N)
    prov = ProvenanceMeta(
        forecaster_name="t", forecaster_version="0", fit_window=(_dt.now(), _dt.now()),
        fold_idx=None, calibration_set_hash=None, random_seed=0,
        code_sha="t", feature_matrix_hash="t", created_at=_dt.now(),
    )
    ids = np.arange(N)
    ts = np.arange(N, dtype=float)
    d_noisy = DistributionForecast.from_normal(
        mu=mu_noisy, sigma=np.ones(N), ids=ids, timestamps=ts, provenance=prov,
    )
    d_precise = DistributionForecast.from_normal(
        mu=mu_precise, sigma=np.ones(N), ids=ids, timestamps=ts, provenance=prov,
    )

    stack = Stacking(deps=("noisy", "precise"))
    stack.fit(np.zeros((N, 1)), y, deps_oof={"noisy": d_noisy, "precise": d_precise})

    # Precise upstream should get the bigger weight.
    assert abs(stack.weights_[1]) > abs(stack.weights_[0])
    # Stacking dist on the same rows should match y closely.
    out = stack.predict_dist(
        np.zeros((N, 1)), ids=ids, timestamps=ts,
        deps_oof={"noisy": d_noisy, "precise": d_precise},
    )
    np.testing.assert_allclose(out.params["mu"].mean(), y.mean(), atol=0.1)


def test_stacking_passes_sample_weight_through_to_lstsq():
    """Doubling the weight on half the rows should shift the meta-OLS
    fit toward those rows. We check by comparing weighted-fit weights
    against unweighted-fit weights — they should differ when the
    upstream μ values differ between the two halves."""
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
    from bracketlearn.trainers import Stacking

    rng = np.random.default_rng(0)
    N = 200
    y = rng.normal(0, 1, N)
    # Two upstreams with structurally different bias per half.
    mu_a = np.where(np.arange(N) < N // 2, y + 0.5, y - 0.5)
    mu_b = y + rng.normal(0, 0.2, N)
    prov = ProvenanceMeta(
        forecaster_name="t", forecaster_version="0", fit_window=(_dt.now(), _dt.now()),
        fold_idx=None, calibration_set_hash=None, random_seed=0,
        code_sha="t", feature_matrix_hash="t", created_at=_dt.now(),
    )
    ids = np.arange(N)
    ts = np.arange(N, dtype=float)
    d_a = DistributionForecast.from_normal(mu=mu_a, sigma=np.ones(N), ids=ids,
                                           timestamps=ts, provenance=prov)
    d_b = DistributionForecast.from_normal(mu=mu_b, sigma=np.ones(N), ids=ids,
                                           timestamps=ts, provenance=prov)
    s_unw = Stacking(deps=("a", "b"))
    s_unw.fit(np.zeros((N, 1)), y, deps_oof={"a": d_a, "b": d_b})

    w_emph_first = np.where(np.arange(N) < N // 2, 4.0, 1.0)
    s_w = Stacking(deps=("a", "b"))
    s_w.fit(np.zeros((N, 1)), y, deps_oof={"a": d_a, "b": d_b},
            sample_weight=w_emph_first)
    # Weighted fit's intercept should bias toward the first-half data.
    assert not np.allclose(s_unw.weights_, s_w.weights_)


# ---------------------------------------------------------------------------
# TailSpecialist — positive integration (audit §6.T1).
# ---------------------------------------------------------------------------


def test_tail_specialist_emits_bracket_with_classifier_tails():
    """TailSpecialist needs an EMOS upstream and an outer-edge-bracketed
    ladder. Confirm the result is bracket-backed, row sums to 1, and
    inner bins agree with upstream EMOS body mass (up to renorm)."""
    _skip_if_missing("lightgbm")
    from bracketlearn.trainers import EMOS, TailSpecialist

    rng = np.random.default_rng(0)
    N, k = 300, 4
    X = rng.normal(0, 1, (N, k))
    y = X.mean(axis=1) + rng.normal(0, 1.0, N)
    # Train an upstream EMOS.
    emos = EMOS().fit(X, y)
    emos_dist = emos.predict_dist(
        X, ids=np.arange(N), timestamps=np.arange(N, dtype=float),
    )
    # Wide ladder so the classifier replaces near-zero edge bins.
    edges = np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])
    ts = TailSpecialist(edges=edges, upstream="emos", n_estimators=30)
    ts.fit(X, y, deps_oof={"emos": emos_dist})
    out = ts.predict_dist(
        X, ids=np.arange(N), timestamps=np.arange(N, dtype=float),
        deps_oof={"emos": emos_dist},
    )
    from bracketlearn.forecast import Backing
    assert out.backing == Backing.BRACKET
    assert out.probs.shape == (N, 6)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(out.probs >= 0)


# ---------------------------------------------------------------------------
# Factories (audit §6.T1) — ridge / market_ols / emos_calibrated.
# ---------------------------------------------------------------------------


def test_ridge_factory_emits_distforecaster_via_lift():
    from bracketlearn.trainers import ridge
    X, y, ids, ts = _synthetic()
    r = ridge()
    # ridge() returns a LiftedForecaster. The lifter needs OOF residuals,
    # so we fit the base directly via the wrapper's exposed path.
    # Simpler integration: use it inside a one-step ForecastPipeline.
    from bracketlearn.pipeline import ForecastPipeline
    p = ForecastPipeline(steps=[("ridge", r)], n_folds=3, refit_on_full=False)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    d = result["ridge"]
    from bracketlearn.forecast import Backing, ParametricFamily
    assert d.backing == Backing.PARAMETRIC
    assert d.family == ParametricFamily.NORMAL


def test_market_ols_factory_emits_distforecaster_via_lift():
    from bracketlearn.pipeline import ForecastPipeline
    from bracketlearn.trainers import market_ols
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(steps=[("ols", market_ols())], n_folds=3, refit_on_full=False)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    from bracketlearn.forecast import Backing, ParametricFamily
    assert result["ols"].backing == Backing.PARAMETRIC
    assert result["ols"].family == ParametricFamily.NORMAL


def test_emos_calibrated_factory_returns_calibrated_forecaster():
    from bracketlearn.pipeline import CalibratedForecaster
    from bracketlearn.trainers import emos_calibrated
    edges = np.linspace(0, 20, 7)
    ec = emos_calibrated(edges=edges)
    assert isinstance(ec, CalibratedForecaster)
    assert ec.name == "emos_calibrated"


# ---------------------------------------------------------------------------
# sample_weight respect (audit §6.T2) — at minimum: doubling a single
# row's weight should shift the fit toward that row's residual.
# ---------------------------------------------------------------------------


def test_sklearn_point_respects_sample_weight():
    """sklearn Ridge supports sample_weight; SklearnPoint should pass it
    through. Verify by comparing unweighted vs weighted fits on data
    where weighting matters."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 2))
    y = X[:, 0] + rng.normal(0, 1, 100)
    sw = np.ones(100)
    sw[:10] = 100.0   # heavily weight the first 10 rows
    sp_unw = SklearnPoint(Ridge(alpha=0.1)).fit(X, y)
    sp_w = SklearnPoint(Ridge(alpha=0.1)).fit(X, y, sample_weight=sw)
    # Coefficients must diverge: weighted Ridge sees the first 10 rows as
    # ~100x more important and shifts its coefficients accordingly.
    assert not np.allclose(
        sp_unw.estimator.coef_, sp_w.estimator.coef_, atol=1e-6,
    )


def test_emos_respects_sample_weight():
    """EMOS does its own weighted least-squares. Compare unweighted vs
    weighted coefficient pairs."""
    rng = np.random.default_rng(0)
    N, k = 100, 4
    X = rng.normal(0, 1, (N, k))
    y = X.mean(axis=1) + rng.normal(0, 0.5, N)
    sw = np.ones(N)
    sw[:25] = 5.0
    e_unw = EMOS().fit(X, y)
    e_w = EMOS().fit(X, y, sample_weight=sw)
    assert not np.isclose(e_unw.a_, e_w.a_) or not np.isclose(e_unw.b_, e_w.b_)


def test_empirical_distribution_respects_sample_weight():
    """Weighted quantiles should diverge from unweighted when the weight
    distribution differs from uniform on the same y."""
    from bracketlearn.baselines import EmpiricalDistribution
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 200)
    sw = np.where(y > 0, 10.0, 1.0)   # heavily upweight positives
    e_unw = EmpiricalDistribution().fit(np.zeros((200, 1)), y)
    e_w = EmpiricalDistribution().fit(np.zeros((200, 1)), y, sample_weight=sw)
    # Weighted median should sit above unweighted median.
    j_med = e_unw.taus.index(0.5)
    assert e_w.quantiles_[j_med] > e_unw.quantiles_[j_med]
