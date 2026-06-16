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

from bracketlearn.forecast import (
    BracketForecast,
    MixtureNormalForecast,
    NormalForecast,
    QuantileForecast,
    StudentTForecast,
)
from bracketlearn.trainers import (
    EMOS,
    BayesianRidge,
    BMAStacking,
    HeteroscedasticNormal,
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
    assert isinstance(d, NormalForecast)
    assert d.params["mu"].shape == (X.shape[0],)
    assert np.all(d.params["sigma"] > 0)


def test_heteroscedastic_normal_emits_parametric_normal():
    X, y, ids, ts = _synthetic()
    h = HeteroscedasticNormal().fit(X, y)
    d = h.predict_dist(X, ids=ids, timestamps=ts)
    assert isinstance(d, NormalForecast)
    assert d.params["mu"].shape == (X.shape[0],)
    assert np.all(d.params["sigma"] > 0)


def test_heteroscedastic_normal_recovers_feature_driven_scale():
    """Column 0 drives the mean, column 1 drives the (log) scale.

    The fitted σ̂ must track the true feature-driven scale — the property
    EMOS (scale ← ens_std only) and homoscedastic regressions cannot model.
    """
    rng = np.random.default_rng(0)
    n = 3000
    x_mean = rng.normal(0, 1, n)
    x_scale = rng.uniform(0, 1, n)
    true_mu = 2.0 + 3.0 * x_mean
    true_sigma = np.exp(0.0 + 1.2 * x_scale)
    y = true_mu + true_sigma * rng.standard_normal(n)
    X = np.column_stack([x_mean, x_scale])

    h = HeteroscedasticNormal().fit(X, y)
    d = h.predict_dist(X, ids=np.arange(n), timestamps=np.zeros(n))
    mu, sigma = d.params["mu"], d.params["sigma"]
    assert np.corrcoef(mu, true_mu)[0, 1] > 0.99
    assert np.corrcoef(sigma, true_sigma)[0, 1] > 0.95


def test_heteroscedastic_normal_head_selection():
    """mu_idx / sigma_idx route columns to the right moment; out-of-range raises."""
    X, y, ids, ts = _synthetic(k=3)
    h = HeteroscedasticNormal(mu_idx=(0, 1), sigma_idx=(2,)).fit(X, y)
    d = h.predict_dist(X, ids=ids, timestamps=ts)
    # μ head has intercept + 2 features; σ head intercept + 1.
    assert h.beta_mu_.shape == (3,)
    assert h.beta_sigma_.shape == (2,)
    assert np.all(d.params["sigma"] > 0)
    with pytest.raises(ValueError, match="out of range"):
        HeteroscedasticNormal(mu_idx=(0, 9)).fit(X, y)


def test_heteroscedastic_normal_rejects_nonfinite():
    X, y, ids, ts = _synthetic()
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        HeteroscedasticNormal().fit(X_bad, y)


def test_bayesian_ridge_emits_parametric_student_t():
    rng = np.random.default_rng(0)
    N, d = 150, 3
    X = rng.standard_normal((N, d))
    y = X @ np.array([0.5, -1.0, 2.0]) + rng.standard_normal(N) * 0.5
    ids = np.arange(N)
    ts = np.arange(N, dtype=float)
    br = BayesianRidge().fit(X, y)
    out = br.predict_dist(X, ids=ids, timestamps=ts)
    assert isinstance(out, StudentTForecast)
    assert out.params["mu"].shape == (N,)
    assert np.all(out.params["sigma"] > 0)
    assert np.all(out.params["df"] > 2.0)


def test_bayesian_ridge_recovers_coefficients_and_sigma():
    """Tight prior_precision shrinks toward zero; flat prior recovers OLS."""
    rng = np.random.default_rng(1)
    N, d = 500, 3
    X = rng.standard_normal((N, d))
    beta_true = np.array([0.5, -1.0, 2.0])
    sigma_true = 0.5
    y = X @ beta_true + rng.standard_normal(N) * sigma_true
    br = BayesianRidge(prior_precision=1e-3).fit(X, y)
    # Slopes recovered after destandardisation: m_n is in standardised X-space
    # so just check the standardised-space predictions match OLS closely on train.
    pred = br.predict_dist(X, ids=np.arange(N), timestamps=np.zeros(N))
    np.testing.assert_allclose(pred.mu.mean(), y.mean(), atol=0.05)
    # Posterior sigma should bracket the noise scale.
    assert 0.3 < pred.sigma.mean() < 0.8


def test_bayesian_ridge_sigma_inflates_away_from_training_data():
    """Predictive σ uses (1 + x*ᵀ V_n x*); rows far from train must be wider.
    Use small N so V_n stays loose enough for the inflation to be visible."""
    rng = np.random.default_rng(2)
    N, d = 30, 2
    X = rng.standard_normal((N, d))
    y = X @ np.array([1.0, -1.0]) + rng.standard_normal(N) * 0.3
    br = BayesianRidge().fit(X, y)
    X_near = rng.standard_normal((20, d))
    X_far = rng.standard_normal((20, d)) * 20.0
    ids = np.arange(20)
    ts = np.zeros(20)
    d_near = br.predict_dist(X_near, ids=ids, timestamps=ts)
    d_far = br.predict_dist(X_far, ids=ids, timestamps=ts)
    assert d_far.sigma.mean() > d_near.sigma.mean() * 1.5


def test_bayesian_ridge_raises_on_zero_variance_column():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 3))
    X[:, 1] = 7.0
    y = rng.standard_normal(50)
    with pytest.raises(ValueError, match="zero-variance column"):
        BayesianRidge().fit(X, y)


def test_bayesian_ridge_raises_on_collinear_columns_without_prior():
    """prior_precision=0 leaves Xᵀ X singular on collinear designs → loud raise."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 3))
    X[:, 2] = X[:, 0]
    y = rng.standard_normal(50)
    with pytest.raises(ValueError, match="prior_precision .* must be strictly positive"):
        BayesianRidge(prior_precision=0.0).fit(X, y)


def test_bayesian_ridge_predict_before_fit_raises():
    br = BayesianRidge()
    with pytest.raises(RuntimeError, match="predict_dist called before fit"):
        br.predict_dist(np.zeros((3, 2)), ids=np.arange(3), timestamps=np.zeros(3))


def test_mixture_normals_emits_mixture_normal():
    X, y, ids, ts = _synthetic(k=4)
    m = MixtureNormals().fit(X, y)
    d = m.predict_dist(X, ids=ids, timestamps=ts)
    assert isinstance(d, MixtureNormalForecast)
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


def test_online_aggregator_grouped_specialises_per_group():
    """Per-group AdaHedge: different best expert per group → different weights."""
    rng = np.random.default_rng(0)
    N_per = 200
    # Group A: expert 0 is best (low noise). Group B: expert 2 is best.
    y_a = rng.normal(0, 1, N_per)
    y_b = rng.normal(0, 1, N_per)
    X_a = np.column_stack([
        y_a + rng.normal(0, 0.1, N_per),
        y_a + rng.normal(0, 2.0, N_per),
        y_a + rng.normal(0, 2.0, N_per),
    ])
    X_b = np.column_stack([
        y_b + rng.normal(0, 2.0, N_per),
        y_b + rng.normal(0, 2.0, N_per),
        y_b + rng.normal(0, 0.1, N_per),
    ])
    X = np.vstack([X_a, X_b])
    y = np.concatenate([y_a, y_b])
    groups = np.array(["A"] * N_per + ["B"] * N_per)
    agg = OnlineAggregator(min_experts=2).fit(X, y, groups=groups)
    assert agg.final_w_by_group_ is not None
    assert set(agg.final_w_by_group_) == {"A", "B"}
    w_a = agg.final_w_by_group_["A"]
    w_b = agg.final_w_by_group_["B"]
    # Each group should concentrate weight on its own best expert.
    assert w_a[0] > w_a[2], f"Group A expected expert 0 > expert 2, got {w_a}"
    assert w_b[2] > w_b[0], f"Group B expected expert 2 > expert 0, got {w_b}"


def test_online_aggregator_grouped_predict_round_trips():
    rng = np.random.default_rng(1)
    N, K = 60, 3
    y = rng.normal(0, 1, N)
    X = y[:, None] + rng.normal(0, 0.5, (N, K))
    groups = np.array(["A" if i % 2 == 0 else "B" for i in range(N)])
    agg = OnlineAggregator(min_experts=2).fit(X, y, groups=groups)
    pf = agg.predict(
        X, ids=np.arange(N), timestamps=np.arange(N, dtype=float),
        groups=groups,
    )
    assert pf.mu.shape == (N,)
    assert not np.isnan(pf.mu).any()


def test_online_aggregator_grouped_predict_rejects_unseen_group():
    rng = np.random.default_rng(2)
    N, K = 40, 3
    y = rng.normal(0, 1, N)
    X = y[:, None] + rng.normal(0, 0.5, (N, K))
    groups_train = np.array(["A"] * N)
    agg = OnlineAggregator(min_experts=2).fit(X, y, groups=groups_train)
    groups_predict = np.array(["B"] * N)
    with pytest.raises(RuntimeError, match="absent from fit-time"):
        agg.predict(
            X, ids=np.arange(N), timestamps=np.arange(N, dtype=float),
            groups=groups_predict,
        )


def test_online_aggregator_grouped_predict_requires_groups_when_grouped_fit():
    rng = np.random.default_rng(3)
    N, K = 40, 3
    y = rng.normal(0, 1, N)
    X = y[:, None] + rng.normal(0, 0.5, (N, K))
    groups_train = np.array(["A"] * (N // 2) + ["B"] * (N - N // 2))
    agg = OnlineAggregator(min_experts=2).fit(X, y, groups=groups_train)
    with pytest.raises(ValueError, match="per-group AdaHedge"):
        agg.predict(X, ids=np.arange(N), timestamps=np.arange(N, dtype=float))


def test_online_aggregator_global_fit_still_works():
    """Default fit (no groups) preserves original single-pool behavior."""
    rng = np.random.default_rng(4)
    N, K = 80, 4
    y = rng.normal(0, 1, N)
    X = y[:, None] + rng.normal(0, 0.5, (N, K))
    agg = OnlineAggregator(min_experts=2).fit(X, y)
    assert agg.final_w_ is not None
    assert agg.final_w_by_group_ is None
    np.testing.assert_allclose(agg.final_w_.sum(), 1.0)
    pf = agg.predict(X, ids=np.arange(N), timestamps=np.arange(N, dtype=float))
    assert pf.mu.shape == (N,)


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
    assert isinstance(d, NormalForecast)


def test_quantile_reg_emits_quantile_backing():
    _skip_if_missing("lightgbm")
    from bracketlearn.trainers import QuantileReg
    X, y, ids, ts = _synthetic(n=80)
    qr = QuantileReg(n_estimators=20, random_seed=0).fit(X, y)
    d = qr.predict_dist(X, ids=ids, timestamps=ts)
    assert isinstance(d, QuantileForecast)
    assert d.qvals.shape[0] == X.shape[0]


def test_quantile_forest_emits_quantile_backing():
    _skip_if_missing("quantile_forest")
    from bracketlearn.trainers import QuantileForest
    X, y, ids, ts = _synthetic(n=80)
    qf = QuantileForest(n_estimators=20, random_seed=0).fit(X, y)
    d = qf.predict_dist(X, ids=ids, timestamps=ts)
    assert isinstance(d, QuantileForecast)


def test_cumulative_binary_emits_bracket():
    _skip_if_missing("lightgbm")
    from bracketlearn.trainers import CumulativeBinary
    X, y, ids, ts = _synthetic(n=80)
    # v0.3: per-row cutpoints + outer_edges via id-keyed dicts. Here the
    # cutpoints happen to be shared across rows but the API requires the
    # dict — exercises the broadcast path.
    shared_cuts = np.array([8.0, 10.0, 12.0])
    cutpoints_by_id = {int(k): shared_cuts for k in ids}
    outer_edges_by_id = {int(k): (0.0, 20.0) for k in ids}
    grids = dict(cutpoints_by_id=cutpoints_by_id, outer_edges_by_id=outer_edges_by_id)
    cb = CumulativeBinary(n_estimators=20).fit(X, y, ids=ids, **grids)
    d = cb.predict_dist(X, ids=ids, timestamps=ts, **grids)
    assert isinstance(d, BracketForecast)
    np.testing.assert_allclose(d.probs.sum(axis=1), 1.0)


def test_cumulative_binary_per_row_varying_cutpoints():
    """Different rows can have different cutpoint counts and grids.
    Output BracketForecast has NaN-padded ragged columns."""
    _skip_if_missing("lightgbm")
    from bracketlearn.trainers import CumulativeBinary
    X, y, ids, ts = _synthetic(n=60)
    # Half the rows get 3 cuts, the other half get 5.
    cuts_a = np.array([8.0, 10.0, 12.0])
    cuts_b = np.array([6.0, 8.0, 10.0, 12.0, 14.0])
    cutpoints_by_id = {int(k): (cuts_a if k % 2 == 0 else cuts_b) for k in ids}
    outer_edges_by_id = {int(k): (0.0, 20.0) for k in ids}
    grids = dict(cutpoints_by_id=cutpoints_by_id, outer_edges_by_id=outer_edges_by_id)
    cb = CumulativeBinary(n_estimators=20).fit(X, y, ids=ids, **grids)
    d = cb.predict_dist(X, ids=ids, timestamps=ts, **grids)
    assert isinstance(d, BracketForecast)
    # Per-row valid bin count B_i = K_i + 1: half the rows have 4 bins
    # (NaN padding in trailing columns), other half have 6.
    valid_per_row = (~np.isnan(d.probs)).sum(axis=1)
    even_rows = ids % 2 == 0
    np.testing.assert_array_equal(valid_per_row[even_rows], 4)   # 3 cuts → 4 bins
    np.testing.assert_array_equal(valid_per_row[~even_rows], 6)  # 5 cuts → 6 bins
    # Per-row sum-to-1 (nansum across the row's valid prefix).
    row_sum = np.nansum(d.probs, axis=1)
    np.testing.assert_allclose(row_sum, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# StackedParametric — positive integration path (audit §6.T1).
# ---------------------------------------------------------------------------


def test_stacking_recovers_truth_from_perfect_upstream():
    """Two upstreams: a noisy one and a precise one. StackedParametric should
    weight the precise upstream more. Doesn't require exact recovery —
    just relative ordering of |weights_|."""
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
    from bracketlearn.trainers import StackedParametric

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

    stack = StackedParametric()
    stack.fit(np.zeros((N, 1)), y, upstream=[d_noisy, d_precise])

    # Precise upstream should get the bigger weight.
    assert abs(stack.weights_[1]) > abs(stack.weights_[0])
    # StackedParametric dist on the same rows should match y closely.
    out = stack.predict_dist(
        np.zeros((N, 1)), ids=ids, timestamps=ts,
        upstream=[d_noisy, d_precise],
    )
    np.testing.assert_allclose(out.params["mu"].mean(), y.mean(), atol=0.1)


def test_stacking_passes_sample_weight_through_to_lstsq():
    """Doubling the weight on half the rows should shift the meta-OLS
    fit toward those rows. We check by comparing weighted-fit weights
    against unweighted-fit weights — they should differ when the
    upstream μ values differ between the two halves."""
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
    from bracketlearn.trainers import StackedParametric

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
    s_unw = StackedParametric()
    s_unw.fit(np.zeros((N, 1)), y, upstream=[d_a, d_b])

    w_emph_first = np.where(np.arange(N) < N // 2, 4.0, 1.0)
    s_w = StackedParametric()
    s_w.fit(np.zeros((N, 1)), y, upstream=[d_a, d_b],
            sample_weight=w_emph_first)
    # Weighted fit's intercept should bias toward the first-half data.
    assert not np.allclose(s_unw.weights_, s_w.weights_)


def test_stacking_convex_weights_sum_to_one_and_nonneg():
    """weight_constraint='convex' must produce non-negative weights summing
    to 1, while still recovering preference for the precise upstream."""
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
    from bracketlearn.trainers import StackedParametric

    rng = np.random.default_rng(0)
    N = 300
    y = rng.normal(0, 1, N)
    mu_noisy = y + rng.normal(0, 1.0, N)
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
    stack = StackedParametric(weight_constraint="convex")
    stack.fit(
        np.zeros((N, 1)), y,
        upstream=[d_noisy, d_precise],
    )
    assert np.all(stack.weights_ >= -1e-9)
    np.testing.assert_allclose(stack.weights_.sum(), 1.0, atol=1e-6)
    # Precise upstream still preferred.
    assert stack.weights_[1] > stack.weights_[0]


def test_stacking_geometric_sigma_tracks_upstream_dispersion():
    """sigma_method='geometric_mean_upstream': σ̂(x) should vary with
    upstream σⱼ(x). Build upstreams whose σ varies row-wise and check
    that the stacked σ̂ moves with them (not constant)."""
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
    from bracketlearn.trainers import StackedParametric

    rng = np.random.default_rng(1)
    N = 400
    # Heteroscedastic ground truth: σ(x) = 0.5 + 0.5 * row-fraction.
    row_frac = np.arange(N) / N
    true_sigma = 0.5 + 0.5 * row_frac
    y = rng.normal(0, true_sigma)
    mu_a = y + rng.normal(0, true_sigma)
    mu_b = y + rng.normal(0, true_sigma)
    # Upstream σ tracks the truth (perfect dispersion info).
    sigma_up = true_sigma.copy()
    prov = ProvenanceMeta(
        forecaster_name="t", forecaster_version="0", fit_window=(_dt.now(), _dt.now()),
        fold_idx=None, calibration_set_hash=None, random_seed=0,
        code_sha="t", feature_matrix_hash="t", created_at=_dt.now(),
    )
    ids = np.arange(N)
    ts = np.arange(N, dtype=float)
    d_a = DistributionForecast.from_normal(
        mu=mu_a, sigma=sigma_up, ids=ids, timestamps=ts, provenance=prov,
    )
    d_b = DistributionForecast.from_normal(
        mu=mu_b, sigma=sigma_up, ids=ids, timestamps=ts, provenance=prov,
    )
    stack = StackedParametric(
        sigma_method="geometric_mean_upstream",
    )
    stack.fit(np.zeros((N, 1)), y, upstream=[d_a, d_b])
    out = stack.predict_dist(
        np.zeros((N, 1)), ids=ids, timestamps=ts,
        upstream=[d_a, d_b],
    )
    sigma_hat = out.params["sigma"]
    # σ̂ should not be flat (constant fallback would fail this).
    assert sigma_hat.std() > 0.05
    # σ̂ should correlate positively with true σ on this scale.
    assert np.corrcoef(sigma_hat, true_sigma)[0, 1] > 0.5


def test_stacking_student_t_emits_t_backed_forecast_with_matching_variance():
    """dist_family='student_t': output is t-backed with df=student_t_df,
    and forecast variance equals σ̂² (the conversion scale = σ̂·√((ν−2)/ν)
    so variance = scale²·ν/(ν−2) = σ̂² holds row-wise)."""
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
    from bracketlearn.trainers import StackedParametric

    rng = np.random.default_rng(2)
    N = 300
    y = rng.normal(0, 1, N)
    mu_a = y + rng.normal(0, 0.5, N)
    prov = ProvenanceMeta(
        forecaster_name="t", forecaster_version="0", fit_window=(_dt.now(), _dt.now()),
        fold_idx=None, calibration_set_hash=None, random_seed=0,
        code_sha="t", feature_matrix_hash="t", created_at=_dt.now(),
    )
    ids = np.arange(N)
    ts = np.arange(N, dtype=float)
    d_a = DistributionForecast.from_normal(
        mu=mu_a, sigma=np.ones(N), ids=ids, timestamps=ts, provenance=prov,
    )
    stack = StackedParametric(dist_family="student_t", student_t_df=5.0)
    stack.fit(np.zeros((N, 1)), y, upstream=[d_a])
    out = stack.predict_dist(
        np.zeros((N, 1)), ids=ids, timestamps=ts, upstream=[d_a],
    )
    # t-backed with df=5 on every row.
    assert isinstance(out, StudentTForecast)
    assert "df" in out.params
    np.testing.assert_allclose(out.params["df"], 5.0)
    # Variance == σ̂² (constant-σ branch, so σ̂ = stack.sigma_).
    scale = out.params["sigma"]
    var = scale ** 2 * 5.0 / (5.0 - 2.0)
    np.testing.assert_allclose(var, stack.sigma_ ** 2, rtol=1e-10)


def test_stacking_invalid_options_raise_loudly():
    """__post_init__ guards against unknown enum values and df ≤ 2."""
    import pytest

    from bracketlearn.trainers import StackedParametric

    with pytest.raises(ValueError, match="weight_constraint"):
        StackedParametric(weight_constraint="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sigma_method"):
        StackedParametric(sigma_method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dist_family"):
        StackedParametric(dist_family="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="student_t_df"):
        StackedParametric(dist_family="student_t", student_t_df=2.0)


# ---------------------------------------------------------------------------
# BMAStacking — Bayesian model averaging meta-learner.
# ---------------------------------------------------------------------------


def _mk_normal_upstream(mu: np.ndarray, sigma: np.ndarray):
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta

    N = mu.shape[0]
    prov = ProvenanceMeta(
        forecaster_name="t", forecaster_version="0",
        fit_window=(_dt.now(), _dt.now()),
        fold_idx=None, calibration_set_hash=None, random_seed=0,
        code_sha="t", feature_matrix_hash="t", created_at=_dt.now(),
    )
    return DistributionForecast.from_normal(
        mu=mu, sigma=sigma,
        ids=np.arange(N), timestamps=np.arange(N, dtype=float),
        provenance=prov,
    )


def test_bma_stacking_emits_mixture_normal_with_row_sum_weights():
    rng = np.random.default_rng(0)
    N = 200
    y = rng.normal(0, 1, N)
    d_a = _mk_normal_upstream(y + rng.normal(0, 0.3, N), np.full(N, 0.3))
    d_b = _mk_normal_upstream(y + rng.normal(0, 1.0, N), np.full(N, 1.0))
    bma = BMAStacking().fit(
        np.zeros((N, 1)), y, upstream=[d_a, d_b],
    )
    out = bma.predict_dist(
        np.zeros((N, 1)),
        ids=np.arange(N), timestamps=np.arange(N, dtype=float),
        upstream=[d_a, d_b],
    )
    assert isinstance(out, MixtureNormalForecast)
    assert out.weights.shape == (N, 2)
    np.testing.assert_allclose(out.weights.sum(axis=1), 1.0, atol=1e-6)
    assert bma.weights_[0] > bma.weights_[1]
    np.testing.assert_allclose(bma.alpha_n_.sum(), 2 * 1.0 + N, rtol=1e-6)


def test_bma_stacking_sigma_inflates_when_upstreams_disagree():
    """Mixture marginal variance grows where upstream μ's disagree —
    StackedParametric with default sigma_method='constant' cannot do this."""
    rng = np.random.default_rng(1)
    N = 300
    y = rng.normal(0, 1, N)
    d_agree = _mk_normal_upstream(y, np.full(N, 0.3))
    d_disagree_a = _mk_normal_upstream(y + 2.0, np.full(N, 0.3))
    d_disagree_b = _mk_normal_upstream(y - 2.0, np.full(N, 0.3))
    bma_agree = BMAStacking().fit(
        np.zeros((N, 1)), y, upstream=[d_agree, d_agree],
    )
    bma_dis = BMAStacking().fit(
        np.zeros((N, 1)), y, upstream=[d_disagree_a, d_disagree_b],
    )
    out_agree = bma_agree.predict_dist(
        np.zeros((N, 1)), ids=np.arange(N),
        timestamps=np.arange(N, dtype=float),
        upstream=[d_agree, d_agree],
    )
    out_dis = bma_dis.predict_dist(
        np.zeros((N, 1)), ids=np.arange(N),
        timestamps=np.arange(N, dtype=float),
        upstream=[d_disagree_a, d_disagree_b],
    )
    assert out_dis.variance().mean() > out_agree.variance().mean() * 3.0


def test_bma_stacking_rejects_misaligned_upstream_ids():
    from datetime import datetime as _dt

    from bracketlearn.forecast import DistributionForecast, ProvenanceMeta

    N = 50
    prov = ProvenanceMeta(
        forecaster_name="t", forecaster_version="0",
        fit_window=(_dt.now(), _dt.now()),
        fold_idx=None, calibration_set_hash=None, random_seed=0,
        code_sha="t", feature_matrix_hash="t", created_at=_dt.now(),
    )
    mu = np.zeros(N)
    sigma = np.ones(N)
    d_a = DistributionForecast.from_normal(
        mu=mu, sigma=sigma, ids=np.arange(N),
        timestamps=np.arange(N, dtype=float), provenance=prov,
    )
    d_b = DistributionForecast.from_normal(
        mu=mu, sigma=sigma, ids=np.arange(N) + 1000,
        timestamps=np.arange(N, dtype=float), provenance=prov,
    )
    with pytest.raises(ValueError, match="ids does not match"):
        BMAStacking().fit(
            np.zeros((N, 1)), np.zeros(N), upstream=[d_a, d_b],
        )


def test_bma_stacking_rejects_invalid_alpha_prior():
    with pytest.raises(ValueError, match="alpha_prior"):
        BMAStacking(alpha_prior=0.0)
    with pytest.raises(ValueError, match="alpha_prior"):
        BMAStacking(alpha_prior=-1.0)


# ---------------------------------------------------------------------------
# HierarchicalNormal — cross-site partial-pooling regression.
# ---------------------------------------------------------------------------


def _make_multisite(*, K, S, n_per_site, beta_0, tau, sigma, seed):
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    for s in range(S):
        beta_s = beta_0 + rng.standard_normal(K) * tau
        X = rng.standard_normal((n_per_site, K))
        y = X @ beta_s + rng.standard_normal(n_per_site) * sigma
        Xs.append(X); ys.append(y); gs.extend([s] * n_per_site)
    return np.vstack(Xs), np.concatenate(ys), np.array(gs)


def test_hierarchical_normal_recovers_variance_components():
    """EB estimates of σ², τ² should land within ~30% of truth for moderate N."""
    from bracketlearn.trainers import HierarchicalNormal
    K = 3
    X, y, g = _make_multisite(
        K=K, S=5, n_per_site=120,
        beta_0=np.array([0.5, -1.0, 2.0]),
        tau=0.4, sigma=0.5, seed=0,
    )
    hn = HierarchicalNormal().fit(X, y, groups=g)
    assert 0.15 < hn.sigma2_ < 0.4   # truth = 0.25
    assert 0.05 < hn.tau2_ < 0.4     # truth = 0.16


def test_hierarchical_normal_emits_normal_predictive():
    from bracketlearn.trainers import HierarchicalNormal
    X, y, g = _make_multisite(
        K=2, S=3, n_per_site=80,
        beta_0=np.array([1.0, -1.0]), tau=0.2, sigma=0.5, seed=1,
    )
    hn = HierarchicalNormal().fit(X, y, groups=g)
    out = hn.predict_dist(
        X[:10], ids=np.arange(10), timestamps=np.zeros(10), groups=g[:10],
    )
    assert isinstance(out, NormalForecast)
    assert out.mu.shape == (10,)
    assert np.all(out.sigma > 0)


def test_hierarchical_normal_rejects_unseen_site_by_default():
    from bracketlearn.trainers import HierarchicalNormal
    X, y, g = _make_multisite(
        K=2, S=3, n_per_site=50,
        beta_0=np.array([1.0, -1.0]), tau=0.2, sigma=0.5, seed=2,
    )
    hn = HierarchicalNormal().fit(X, y, groups=g)
    with pytest.raises(ValueError, match="unseen at fit"):
        hn.predict_dist(
            X[:5], ids=np.arange(5), timestamps=np.zeros(5),
            groups=np.array([999] * 5),
        )


def test_hierarchical_normal_unseen_site_sigma_inflates():
    """With allow_unseen_sites=True, predictive σ on a new site must
    exceed σ on a known site (extra τ² + posterior on β₀)."""
    from bracketlearn.trainers import HierarchicalNormal
    X, y, g = _make_multisite(
        K=2, S=3, n_per_site=50,
        beta_0=np.array([1.0, -1.0]), tau=0.5, sigma=0.5, seed=3,
    )
    hn = HierarchicalNormal(allow_unseen_sites=True).fit(X, y, groups=g)
    X_new = np.random.default_rng(9).standard_normal((20, 2))
    out_seen = hn.predict_dist(
        X_new, ids=np.arange(20), timestamps=np.zeros(20),
        groups=np.array([0] * 20),
    )
    out_unseen = hn.predict_dist(
        X_new, ids=np.arange(20), timestamps=np.zeros(20),
        groups=np.array([999] * 20),
    )
    assert out_unseen.sigma.mean() > out_seen.sigma.mean() * 1.2


def test_hierarchical_normal_beats_per_site_on_thin_sites():
    """Imbalanced sites: per-site Ridge overfits the thin one,
    HierarchicalNormal pools toward β₀ and wins."""
    from sklearn.linear_model import Ridge

    from bracketlearn.trainers import HierarchicalNormal

    rng = np.random.default_rng(4)
    K = 4
    beta_0 = np.array([0.5, -1.0, 0.0, 2.0])
    tau = 0.5
    sigma = 0.5
    Xs, ys, gs = [], [], []
    test_X, test_y, test_g = [], [], []
    for s, n_tr in enumerate([10, 200, 200, 200]):
        beta_s = beta_0 + rng.standard_normal(K) * tau
        X_all = rng.standard_normal((n_tr + 40, K))
        y_all = X_all @ beta_s + rng.standard_normal(n_tr + 40) * sigma
        Xs.append(X_all[:n_tr]); ys.append(y_all[:n_tr]); gs.extend([s] * n_tr)
        test_X.append(X_all[n_tr:]); test_y.append(y_all[n_tr:]); test_g.extend([s] * 40)
    X_tr = np.vstack(Xs); y_tr = np.concatenate(ys); g_tr = np.array(gs)
    X_te = np.vstack(test_X); y_te = np.concatenate(test_y); g_te = np.array(test_g)

    hn = HierarchicalNormal().fit(X_tr, y_tr, groups=g_tr)
    hn_pred = hn.predict_dist(
        X_te, ids=np.arange(len(y_te)), timestamps=np.zeros(len(y_te)),
        groups=g_te,
    )
    hn_rmse = float(np.sqrt(((hn_pred.mu - y_te) ** 2).mean()))

    # Per-site Ridge.
    rmse_ps = 0.0
    for s in np.unique(g_tr):
        mask_tr = g_tr == s; mask_te = g_te == s
        m = Ridge(alpha=1.0).fit(X_tr[mask_tr], y_tr[mask_tr])
        rmse_ps += ((m.predict(X_te[mask_te]) - y_te[mask_te]) ** 2).sum()
    ps_rmse = float(np.sqrt(rmse_ps / len(y_te)))

    assert hn_rmse < ps_rmse, (
        f"HierarchicalNormal RMSE {hn_rmse:.3f} should beat "
        f"per-site Ridge {ps_rmse:.3f} on imbalanced sites"
    )


def test_hierarchical_normal_requires_groups_at_fit():
    from bracketlearn.trainers import HierarchicalNormal
    X = np.random.default_rng(0).standard_normal((20, 2))
    y = np.zeros(20)
    with pytest.raises(ValueError, match="groups .* is required"):
        HierarchicalNormal().fit(X, y)


def test_hierarchical_normal_requires_multiple_sites():
    from bracketlearn.trainers import HierarchicalNormal
    X = np.random.default_rng(0).standard_normal((20, 2))
    y = np.zeros(20)
    with pytest.raises(ValueError, match="≥2 sites"):
        HierarchicalNormal().fit(X, y, groups=np.zeros(20))


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
    ids_arr = np.arange(N)
    brackets_by_id = {int(i): edges for i in ids_arr}
    ts = TailSpecialist(
        brackets_by_id=brackets_by_id, n_estimators=30,
    )
    ts.fit(X, y, ids=ids_arr, upstream=[emos_dist])
    # Wide outer bins → EMOS body mass concentrates inside, so the classifier's
    # tail probabilities legitimately disagree with EMOS at the edges. That
    # disagreement is what we want; assert the warning fires and silence it.
    with pytest.warns(UserWarning, match="TailSpecialist"):
        out = ts.predict_dist(
            X, ids=ids_arr, timestamps=np.arange(N, dtype=float),
            upstream=[emos_dist],
        )
    assert isinstance(out, BracketForecast)
    assert out.probs.shape == (N, 6)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(out.probs >= 0)


# ---------------------------------------------------------------------------
# Factories (audit §6.T1) — ridge / emos_calibrated.
# ---------------------------------------------------------------------------


def test_ridge_factory_emits_distforecaster_via_lift():
    from bracketlearn.compose import WalkForward
    from bracketlearn.pipeline import Pipeline
    from bracketlearn.trainers import ridge
    X, y, ids, ts = _synthetic()
    r = ridge()
    # ridge() returns a self-contained Pipeline([SklearnPoint(Ridge), Lift]).
    assert isinstance(r, Pipeline)
    assert r.name == "ridge"
    result = WalkForward(n_folds=3, refit_on_full=False).fit_predict(
        r, X, y, ids=ids, timestamps=ts,
    )
    assert isinstance(result["ridge"], NormalForecast)


def test_emos_calibrated_factory_returns_pipeline():
    from bracketlearn.pipeline import Pipeline
    from bracketlearn.trainers import emos_calibrated
    edges = np.linspace(0, 20, 7)
    ec = emos_calibrated(edges=edges)
    assert isinstance(ec, Pipeline)
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


# ---------------------------------------------------------------------------
# BracketExpander — per-row <-> per-(row, bracket) reshape + dist assembly.
#
# Replaces the BracketClassifier / BracketRegressor unit tests deleted in
# v0.5.0. The classes are gone — callers now compose BracketExpander with
# any sklearn estimator. These tests exercise the expander directly plus
# one end-to-end "expander + LogisticRegression" scenario to prove the
# composition path.
# ---------------------------------------------------------------------------


def test_bracket_expander_fit_transform_default_target_is_hit_indicator():
    """fit_transform(X, y, ids) produces (X_exp, y_exp) where y_exp[k]==1
    iff y[i] fell in bracket b of row i."""
    from bracketlearn import BracketExpander

    edges = np.array([0.0, 1.0, 2.0, 3.0])  # 3 bins
    bbi = {0: edges, 1: edges}
    X = np.array([[10.0], [20.0]])
    y = np.array([0.5, 2.5])                # row 0 -> bin 0, row 1 -> bin 2
    ids = np.array([0, 1])
    exp = BracketExpander(brackets_by_id=bbi)
    X_exp, y_exp = exp.fit_transform(X, y, ids=ids)
    assert X_exp.shape == (6, 3)            # 2 rows * 3 bins, (1 feat + lo + hi)
    assert y_exp.shape == (6,)
    np.testing.assert_array_equal(y_exp, [1, 0, 0, 0, 0, 1])


def test_bracket_expander_transform_y_none_returns_only_x():
    """transform(X, ids) -> (X_exp, None) for predict-side."""
    from bracketlearn import BracketExpander

    edges = np.array([0.0, 1.0, 2.0])
    exp = BracketExpander(brackets_by_id={0: edges})
    X_exp, y_exp = exp.transform(np.array([[5.0]]), ids=np.array([0]))
    assert X_exp.shape == (2, 3)
    assert y_exp is None


def test_bracket_expander_supports_ragged_brackets():
    """Per-row B may differ; M = sum(B_i)."""
    from bracketlearn import BracketExpander

    bbi = {
        0: np.array([0.0, 1.0, 2.0]),                  # 2 bins
        1: np.array([0.0, 0.5, 1.0, 1.5, 2.0]),         # 4 bins
    }
    X = np.zeros((2, 1))
    y = np.array([0.5, 1.25])
    exp = BracketExpander(brackets_by_id=bbi)
    X_exp, y_exp = exp.fit_transform(X, y, ids=np.array([0, 1]))
    assert X_exp.shape == (6, 3)
    np.testing.assert_array_equal(y_exp, [1, 0, 0, 0, 1, 0])


def test_bracket_expander_assemble_dist_round_trips():
    """assemble_dist(scores) clips + renormalises per row and packs into a
    BracketForecast whose probs sum to 1 row-wise."""
    from bracketlearn import BracketExpander
    from bracketlearn.forecast import BracketForecast

    edges = np.array([0.0, 1.0, 2.0, 3.0])
    bbi = {0: edges, 1: edges}
    exp = BracketExpander(brackets_by_id=bbi)
    exp.transform(np.zeros((2, 1)), ids=np.array([0, 1]))
    scores = np.array([0.3, 0.4, 0.3, 0.1, 0.5, 0.4])
    dist = exp.assemble_dist(scores, ids=np.array([0, 1]),
                             timestamps=np.array([0.0, 1.0]))
    assert isinstance(dist, BracketForecast)
    assert dist.probs.shape == (2, 3)
    np.testing.assert_allclose(dist.probs.sum(axis=1), 1.0, atol=1e-12)


def test_bracket_expander_assemble_dist_length_mismatch_raises():
    """If predictions length != offsets_[-1] (transform/predict drift), raise."""
    from bracketlearn import BracketExpander

    edges = np.array([0.0, 1.0, 2.0])
    exp = BracketExpander(brackets_by_id={0: edges})
    exp.transform(np.zeros((1, 1)), ids=np.array([0]))
    with pytest.raises(ValueError, match="predictions length"):
        exp.assemble_dist(np.zeros(5), ids=np.array([0]),
                          timestamps=np.array([0.0]))


def test_bracket_expander_assemble_dist_before_transform_raises():
    from bracketlearn import BracketExpander

    edges = np.array([0.0, 1.0, 2.0])
    exp = BracketExpander(brackets_by_id={0: edges})
    with pytest.raises(RuntimeError, match="before transform"):
        exp.assemble_dist(np.zeros(2), ids=np.array([0]),
                          timestamps=np.array([0.0]))


def test_bracket_expander_y_length_mismatch_raises():
    """fit_transform with y shape != ids shape raises."""
    from bracketlearn import BracketExpander

    edges = np.array([0.0, 1.0, 2.0])
    exp = BracketExpander(brackets_by_id={0: edges, 1: edges})
    with pytest.raises(ValueError, match="y has length"):
        exp.fit_transform(np.zeros((2, 1)), y=np.array([0.5]), ids=np.array([0, 1]))


def test_bracket_expander_missing_id_raises():
    """Predict-time id not in brackets_by_id raises (matches old
    BracketClassifier behaviour)."""
    from bracketlearn import BracketExpander

    edges = np.array([0.0, 1.0, 2.0])
    exp = BracketExpander(brackets_by_id={0: edges})
    with pytest.raises(KeyError):
        exp.transform(np.zeros((1, 1)), ids=np.array([99]))


def test_bracket_expander_end_to_end_with_logistic_regression():
    """Full bracket-hit-classifier pipeline composed by hand. Output should
    be a BracketForecast with per-row probs summing to 1 and (on simple
    linear data) the realised bin sees the largest mass on average."""
    from sklearn.linear_model import LogisticRegression

    from bracketlearn import BracketExpander
    from bracketlearn.forecast import BracketForecast

    rng = np.random.default_rng(0)
    N, F = 200, 3
    X = rng.standard_normal((N, F))
    y = X[:, 0] + 0.3 * rng.standard_normal(N)
    ids = np.arange(N)
    edges = np.linspace(-3.0, 3.0, 7)        # 6 shared bins
    bbi = {int(k): edges for k in ids}

    exp = BracketExpander(brackets_by_id=bbi)
    X_exp, y_exp = exp.fit_transform(X, y, ids=ids)
    clf = LogisticRegression(max_iter=500).fit(X_exp, y_exp)

    X_pred_exp, _ = exp.transform(X, ids=ids)
    scores = clf.predict_proba(X_pred_exp)[:, 1]
    dist = exp.assemble_dist(
        scores, ids=ids, timestamps=np.arange(N, dtype=float),
    )
    assert isinstance(dist, BracketForecast)
    assert dist.probs.shape == (N, 6)
    np.testing.assert_allclose(dist.probs.sum(axis=1), 1.0, atol=1e-9)
    # On well-fit linear data, the bin containing y should outscore a
    # uniform baseline (which would be 1/6 ≈ 0.167). Spot-check rows where
    # the realised bin is interior (drops ±∞ tails).
    realised_bin = np.searchsorted(edges, y, side="right") - 1
    interior = (realised_bin >= 0) & (realised_bin < 6)
    mass_at_truth = dist.probs[np.arange(N)[interior], realised_bin[interior]]
    assert float(mass_at_truth.mean()) > 1.0 / 6.0



# ---------------------------------------------------------------------------
# BracketStacking — multiclass head over concatenated bracket-prob deps.
# ---------------------------------------------------------------------------


def _mk_bracket_upstream(
    probs: np.ndarray, edges: np.ndarray, *, source: str = "test",
):
    from bracketlearn.forecast import BracketForecast, ProvenanceMeta

    N = probs.shape[0]
    prov = ProvenanceMeta.placeholder(source)
    return BracketForecast.from_arrays(
        edges=edges,
        probs=probs,
        ids=np.arange(N),
        timestamps=np.arange(N, dtype=float),
        provenance=prov,
    )


def test_bracket_stacking_emits_bracket_forecast_with_correct_K():
    _skip_if_missing("lightgbm")
    import lightgbm as lgb

    from bracketlearn.trainers import BracketStacking

    rng = np.random.default_rng(0)
    N, K = 300, 4
    edges = np.linspace(0.0, 10.0, K + 1)
    # Two upstreams with noisy probs over K bins.
    pa = rng.dirichlet(np.full(K, 2.0), size=N)
    pb = rng.dirichlet(np.full(K, 2.0), size=N)
    d_a = _mk_bracket_upstream(pa, edges, source="a")
    d_b = _mk_bracket_upstream(pb, edges, source="b")
    # Truth concentrated in one of the bins per row.
    y = rng.uniform(edges[0], edges[-1], N)
    stack = BracketStacking(
        estimator=lgb.LGBMClassifier(
            n_estimators=20, num_leaves=4, min_child_samples=10,
            objective="multiclass", verbose=-1,
        ),
    )
    stack.fit(np.zeros((N, 1)), y,
              ids=np.arange(N),
              upstream=[d_a, d_b])
    out = stack.predict_dist(
        np.zeros((N, 1)),
        ids=np.arange(N),
        timestamps=np.arange(N, dtype=float),
        upstream=[d_a, d_b],
    )
    from bracketlearn.forecast import BracketForecast
    assert isinstance(out, BracketForecast)
    assert out.probs.shape == (N, K)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-6)
    assert stack.K_ == K


def test_bracket_stacking_rejects_K_mismatch():
    _skip_if_missing("lightgbm")
    import lightgbm as lgb

    from bracketlearn.trainers import BracketStacking

    rng = np.random.default_rng(1)
    N = 100
    edges_a = np.linspace(0, 10, 5)  # K=4
    edges_b = np.linspace(0, 10, 4)  # K=3
    pa = rng.dirichlet(np.full(4, 2.0), size=N)
    pb = rng.dirichlet(np.full(3, 2.0), size=N)
    d_a = _mk_bracket_upstream(pa, edges_a)
    d_b = _mk_bracket_upstream(pb, edges_b)
    stack = BracketStacking(
        estimator=lgb.LGBMClassifier(verbose=-1),
    )
    with pytest.raises(ValueError, match="must share bracket count"):
        stack.fit(np.zeros((N, 1)), np.zeros(N),
                  ids=np.arange(N),
                  upstream=[d_a, d_b])


def test_bracket_stacking_rejects_misaligned_ids():
    _skip_if_missing("lightgbm")
    import lightgbm as lgb

    from bracketlearn.forecast import BracketForecast, ProvenanceMeta
    from bracketlearn.trainers import BracketStacking

    rng = np.random.default_rng(2)
    N, K = 60, 3
    edges = np.linspace(0, 10, K + 1)
    pa = rng.dirichlet(np.full(K, 2.0), size=N)
    pb = rng.dirichlet(np.full(K, 2.0), size=N)
    prov = ProvenanceMeta.placeholder("t")
    d_a = BracketForecast.from_arrays(
        edges=edges, probs=pa,
        ids=np.arange(N), timestamps=np.arange(N, dtype=float),
        provenance=prov,
    )
    d_b = BracketForecast.from_arrays(
        edges=edges, probs=pb,
        ids=np.arange(N) + 1000, timestamps=np.arange(N, dtype=float),
        provenance=prov,
    )
    stack = BracketStacking(
        estimator=lgb.LGBMClassifier(verbose=-1),
    )
    with pytest.raises(ValueError, match="ids does not match"):
        stack.fit(np.zeros((N, 1)), np.zeros(N),
                  ids=np.arange(N),
                  upstream=[d_a, d_b])


def test_bracket_stacking_rejects_non_bracket_upstream():
    _skip_if_missing("lightgbm")
    import lightgbm as lgb

    from bracketlearn.trainers import BracketStacking

    rng = np.random.default_rng(3)
    N, K = 60, 3
    edges = np.linspace(0, 10, K + 1)
    pa = rng.dirichlet(np.full(K, 2.0), size=N)
    d_a = _mk_bracket_upstream(pa, edges)
    d_normal = _mk_normal_upstream(np.zeros(N), np.ones(N))
    stack = BracketStacking(
        estimator=lgb.LGBMClassifier(verbose=-1),
    )
    with pytest.raises(NotImplementedError, match="bracket-backed"):
        stack.fit(np.zeros((N, 1)), np.zeros(N),
                  ids=np.arange(N),
                  upstream=[d_a, d_normal])


def test_bracket_stacking_predict_before_fit_raises():
    from sklearn.dummy import DummyClassifier

    from bracketlearn.trainers import BracketStacking

    stack = BracketStacking(
        estimator=DummyClassifier(strategy="uniform"),
    )
    with pytest.raises(RuntimeError, match="before fit"):
        stack.predict_dist(
            np.zeros((1, 1)), ids=np.array([0]),
            timestamps=np.array([0.0]),
            upstream=[None],
        )


def test_bracket_stacking_requires_upstream_at_fit():
    # Empty deps is legal at construction now (the positional ``upstream=``
    # contract supplies forecasts at fit). The "no upstream" guard moved to
    # fit-time: calling fit with neither upstream nor deps_oof raises.
    import numpy as np
    from sklearn.dummy import DummyClassifier

    from bracketlearn.trainers import BracketStacking

    est = BracketStacking(estimator=DummyClassifier())  # OK: no raise
    with pytest.raises(ValueError, match="upstream"):
        est.fit(np.zeros((4, 1)), np.zeros(4), ids=np.arange(4))
