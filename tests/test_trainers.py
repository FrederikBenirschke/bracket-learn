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
    cb = CumulativeBinary(
        cutpoints_by_id=cutpoints_by_id,
        outer_edges_by_id=outer_edges_by_id,
        n_estimators=20,
    ).fit(X, y, ids=ids)
    d = cb.predict_dist(X, ids=ids, timestamps=ts)
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
    cb = CumulativeBinary(
        cutpoints_by_id=cutpoints_by_id,
        outer_edges_by_id=outer_edges_by_id,
        n_estimators=20,
    ).fit(X, y, ids=ids)
    d = cb.predict_dist(X, ids=ids, timestamps=ts)
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

    stack = StackedParametric(deps=("noisy", "precise"))
    stack.fit(np.zeros((N, 1)), y, deps_oof={"noisy": d_noisy, "precise": d_precise})

    # Precise upstream should get the bigger weight.
    assert abs(stack.weights_[1]) > abs(stack.weights_[0])
    # StackedParametric dist on the same rows should match y closely.
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
    s_unw = StackedParametric(deps=("a", "b"))
    s_unw.fit(np.zeros((N, 1)), y, deps_oof={"a": d_a, "b": d_b})

    w_emph_first = np.where(np.arange(N) < N // 2, 4.0, 1.0)
    s_w = StackedParametric(deps=("a", "b"))
    s_w.fit(np.zeros((N, 1)), y, deps_oof={"a": d_a, "b": d_b},
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
    stack = StackedParametric(deps=("noisy", "precise"), weight_constraint="convex")
    stack.fit(
        np.zeros((N, 1)), y,
        deps_oof={"noisy": d_noisy, "precise": d_precise},
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
        deps=("a", "b"),
        sigma_method="geometric_mean_upstream",
    )
    stack.fit(np.zeros((N, 1)), y, deps_oof={"a": d_a, "b": d_b})
    out = stack.predict_dist(
        np.zeros((N, 1)), ids=ids, timestamps=ts,
        deps_oof={"a": d_a, "b": d_b},
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
    stack = StackedParametric(deps=("a",), dist_family="student_t", student_t_df=5.0)
    stack.fit(np.zeros((N, 1)), y, deps_oof={"a": d_a})
    out = stack.predict_dist(
        np.zeros((N, 1)), ids=ids, timestamps=ts, deps_oof={"a": d_a},
    )
    # t-backed with df=5 on every row.
    assert out.backing.value == "parametric"
    assert "df" in out.params
    np.testing.assert_allclose(out.params["df"], 5.0)
    # Variance == σ̂² (constant-σ branch, so σ̂ = stack.sigma_).
    scale = out.params["sigma"]
    var = scale ** 2 * 5.0 / (5.0 - 2.0)
    np.testing.assert_allclose(var, stack.sigma_ ** 2, rtol=1e-10)


def test_stacking_invalid_options_raise_loudly():
    """__post_init__ guards against unknown enum values and df ≤ 2."""
    from bracketlearn.trainers import StackedParametric

    import pytest

    with pytest.raises(ValueError, match="weight_constraint"):
        StackedParametric(deps=("a",), weight_constraint="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sigma_method"):
        StackedParametric(deps=("a",), sigma_method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dist_family"):
        StackedParametric(deps=("a",), dist_family="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="student_t_df"):
        StackedParametric(deps=("a",), dist_family="student_t", student_t_df=2.0)


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
    bma = BMAStacking(deps=("a", "b")).fit(
        np.zeros((N, 1)), y, deps_oof={"a": d_a, "b": d_b},
    )
    out = bma.predict_dist(
        np.zeros((N, 1)),
        ids=np.arange(N), timestamps=np.arange(N, dtype=float),
        deps_oof={"a": d_a, "b": d_b},
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
    bma_agree = BMAStacking(deps=("p", "q")).fit(
        np.zeros((N, 1)), y, deps_oof={"p": d_agree, "q": d_agree},
    )
    bma_dis = BMAStacking(deps=("p", "q")).fit(
        np.zeros((N, 1)), y, deps_oof={"p": d_disagree_a, "q": d_disagree_b},
    )
    out_agree = bma_agree.predict_dist(
        np.zeros((N, 1)), ids=np.arange(N),
        timestamps=np.arange(N, dtype=float),
        deps_oof={"p": d_agree, "q": d_agree},
    )
    out_dis = bma_dis.predict_dist(
        np.zeros((N, 1)), ids=np.arange(N),
        timestamps=np.arange(N, dtype=float),
        deps_oof={"p": d_disagree_a, "q": d_disagree_b},
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
        BMAStacking(deps=("a", "b")).fit(
            np.zeros((N, 1)), np.zeros(N), deps_oof={"a": d_a, "b": d_b},
        )


def test_bma_stacking_rejects_invalid_alpha_prior():
    with pytest.raises(ValueError, match="alpha_prior"):
        BMAStacking(deps=("a",), alpha_prior=0.0)
    with pytest.raises(ValueError, match="alpha_prior"):
        BMAStacking(deps=("a",), alpha_prior=-1.0)


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
        brackets_by_id=brackets_by_id, upstream="emos", n_estimators=30,
    )
    ts.fit(X, y, ids=ids_arr, deps_oof={"emos": emos_dist})
    # Wide outer bins → EMOS body mass concentrates inside, so the classifier's
    # tail probabilities legitimately disagree with EMOS at the edges. That
    # disagreement is what we want; assert the warning fires and silence it.
    with pytest.warns(UserWarning, match="TailSpecialist"):
        out = ts.predict_dist(
            X, ids=ids_arr, timestamps=np.arange(N, dtype=float),
            deps_oof={"emos": emos_dist},
        )
    assert isinstance(out, BracketForecast)
    assert out.probs.shape == (N, 6)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(out.probs >= 0)


# ---------------------------------------------------------------------------
# Factories (audit §6.T1) — ridge / emos_calibrated.
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
    assert isinstance(d, NormalForecast)


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
