"""Tests for ppf extensions + DistAsFeatures + LinearPoolDist."""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.forecast import DistributionForecast, TailPolicy, TailRule
from bracketlearn.trainers import (
    EMOS,
    CDFBoostBracket,
    DistAsFeatures,
    LinearPoolDist,
    MixtureNormals,
    SklearnPoint,
)


def _skip_if_no_lightgbm():
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm not installed")


# ---------------------------------------------------------------------------
# ppf extensions
# ---------------------------------------------------------------------------


class TestPpfQuantile:
    def test_recovers_stored_taus(self, prov, ids_ts):
        ids, ts = ids_ts(3)
        taus = np.array([0.1, 0.5, 0.9])
        qvals = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [-1.0, 0.0, 1.0]])
        d = DistributionForecast.from_quantiles(
            taus=taus, qvals=qvals,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=ids, timestamps=ts, provenance=prov,
        )
        # ppf at the stored taus must reproduce qvals exactly.
        out = d.ppf(taus)
        np.testing.assert_allclose(out, qvals)

    def test_clip_outside_range(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        d = DistributionForecast.from_quantiles(
            taus=np.array([0.2, 0.8]),
            qvals=np.array([[5.0, 15.0]]),
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=ids, timestamps=ts, provenance=prov,
        )
        out = d.ppf(np.array([0.0, 0.5, 1.0]))
        # 0.5 is midpoint of [0.2, 0.8] → linear interp at 0.5 → 10.0.
        np.testing.assert_allclose(out[0], [5.0, 10.0, 15.0])

    def test_monotone_in_tau(self, prov, ids_ts):
        ids, ts = ids_ts(2)
        d = DistributionForecast.from_quantiles(
            taus=np.array([0.1, 0.5, 0.9]),
            qvals=np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 5.0]]),
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=ids, timestamps=ts, provenance=prov,
        )
        out = d.ppf(np.linspace(0.05, 0.95, 20))
        assert np.all(np.diff(out, axis=1) >= -1e-12)


class TestPpfBracket:
    def test_uniform_brackets_recovers_edges(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        # Three equal bins on [0, 3] → each prob 1/3. ppf at 1/3 → 1.0.
        d = DistributionForecast.from_brackets(
            edges=np.array([0.0, 1.0, 2.0, 3.0]),
            probs=np.array([[1 / 3, 1 / 3, 1 / 3]]),
            ids=ids, timestamps=ts, provenance=prov,
        )
        out = d.ppf(np.array([0.0, 1 / 3, 2 / 3, 1.0]))
        np.testing.assert_allclose(out[0], [0.0, 1.0, 2.0, 3.0], atol=1e-10)

    def test_inverts_cdf(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        rng = np.random.default_rng(0)
        probs_row = rng.dirichlet(np.ones(5))
        d = DistributionForecast.from_brackets(
            edges=np.linspace(0, 10, 6),
            probs=probs_row[None, :],
            ids=ids, timestamps=ts, provenance=prov,
        )
        taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q = d.ppf(taus)
        # F(ppf(τ)) should round-trip back to τ. cdf on N=1 returns (1, M).
        recovered = d.cdf(q[0])
        np.testing.assert_allclose(recovered[0], taus, atol=1e-10)


class TestPpfMixtureNormal:
    def test_collapses_to_normal_when_single_component_weight(self, prov, ids_ts):
        ids, ts = ids_ts(2)
        # Two components but second has zero weight → behaves as N(μ_1, σ_1).
        # Mixture validator forbids weight=0? Check: it requires nonneg, not >0.
        d = DistributionForecast.from_mixture_normal(
            weights=np.array([[1.0, 0.0], [0.5, 0.5]]),
            mus=np.array([[0.0, 100.0], [-1.0, 1.0]]),
            sigmas=np.array([[1.0, 1.0], [0.5, 0.5]]),
            ids=ids, timestamps=ts, provenance=prov,
        )
        med = d.ppf(0.5)
        # Row 0: pure N(0, 1) → median = 0.
        # Row 1: symmetric 50/50 mixture of N(-1, .5) and N(1, .5) → median = 0.
        np.testing.assert_allclose(med, [0.0, 0.0], atol=1e-6)

    def test_round_trips_cdf(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        d = DistributionForecast.from_mixture_normal(
            weights=np.array([[0.3, 0.7]]),
            mus=np.array([[-2.0, 3.0]]),
            sigmas=np.array([[1.0, 0.8]]),
            ids=ids, timestamps=ts, provenance=prov,
        )
        taus = np.array([0.1, 0.5, 0.9])
        q = d.ppf(taus)
        recovered = d.cdf(q[0])
        np.testing.assert_allclose(recovered[0], taus, atol=1e-4)


# ---------------------------------------------------------------------------
# DistAsFeatures
# ---------------------------------------------------------------------------


def _normal_dist(mu_arr, sigma_arr, prov, ids, ts):
    return DistributionForecast.from_normal(
        mu=mu_arr, sigma=sigma_arr, ids=ids, timestamps=ts, provenance=prov,
    )


class TestDistAsFeatures:
    def test_featurise_shape(self, prov, ids_ts):
        ids, ts = ids_ts(10)
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 10)
        d1 = _normal_dist(rng.normal(0, 1, 10), np.full(10, 1.0), prov, ids, ts)
        d2 = _normal_dist(rng.normal(0, 1, 10), np.full(10, 2.0), prov, ids, ts)
        from sklearn.linear_model import LinearRegression
        node = DistAsFeatures(
            deps=("a", "b"),
            downstream=SklearnPoint(LinearRegression()),
            feature_taus=(0.1, 0.5, 0.9),
            tail_cutpoints=(-1.0, 1.0),
        )
        deps = {"a": d1, "b": d2}
        node.fit(np.zeros((10, 1)), y, deps_oof=deps)
        # 2 deps × (3 taus + mean + var + 2 cutpoints) = 14
        assert node._n_features_ == 2 * (3 + 1 + 1 + 2)

    def test_predict_round_trips(self, prov, ids_ts):
        ids, ts = ids_ts(20)
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 20)
        d1 = _normal_dist(rng.normal(0, 1, 20), np.full(20, 1.0), prov, ids, ts)
        d2 = _normal_dist(rng.normal(0, 1, 20), np.full(20, 1.0), prov, ids, ts)
        from sklearn.linear_model import LinearRegression
        node = DistAsFeatures(
            deps=("a", "b"),
            downstream=SklearnPoint(LinearRegression()),
        )
        deps = {"a": d1, "b": d2}
        node.fit(np.zeros((20, 1)), y, deps_oof=deps)
        pf = node.predict(np.zeros((20, 1)), ids=ids, timestamps=ts, deps_oof=deps)
        assert pf.mu.shape == (20,)

    def test_missing_dep_raises(self, prov, ids_ts):
        ids, ts = ids_ts(5)
        from sklearn.linear_model import LinearRegression
        node = DistAsFeatures(
            deps=("a", "b"),
            downstream=SklearnPoint(LinearRegression()),
        )
        with pytest.raises(ValueError, match="deps_oof"):
            node.fit(np.zeros((5, 1)), np.zeros(5), deps_oof={"a": None})


# ---------------------------------------------------------------------------
# LinearPoolDist
# ---------------------------------------------------------------------------


class TestLinearPoolDist:
    def test_prefers_better_component(self, prov, ids_ts):
        # Component A is calibrated; B is far off. Weight should concentrate on A.
        ids, ts = ids_ts(100)
        rng = np.random.default_rng(2)
        y = rng.normal(0, 1, 100)
        # A: N(y, 1) — perfect predictor with noise.
        d_a = _normal_dist(y + rng.normal(0, 0.1, 100), np.full(100, 1.0),
                           prov, ids, ts)
        # B: N(10, 1) — far off.
        d_b = _normal_dist(np.full(100, 10.0), np.full(100, 1.0), prov, ids, ts)
        pool = LinearPoolDist(deps=("a", "b"), n_samples=50)
        pool.fit(np.zeros((100, 1)), y, deps_oof={"a": d_a, "b": d_b})
        assert pool.weights_[0] > 0.9
        assert pool.weights_[1] < 0.1

    def test_predict_returns_quantile_dist(self, prov, ids_ts):
        ids, ts = ids_ts(30)
        rng = np.random.default_rng(3)
        y = rng.normal(0, 1, 30)
        d_a = _normal_dist(rng.normal(0, 1, 30), np.full(30, 1.0), prov, ids, ts)
        d_b = _normal_dist(rng.normal(0, 1, 30), np.full(30, 1.5), prov, ids, ts)
        pool = LinearPoolDist(deps=("a", "b"), n_samples=50)
        pool.fit(np.zeros((30, 1)), y, deps_oof={"a": d_a, "b": d_b})
        out = pool.predict_dist(
            np.zeros((30, 1)), ids=ids, timestamps=ts,
            deps_oof={"a": d_a, "b": d_b},
        )
        assert out.qvals.shape == (30, 99)
        # Monotone non-decreasing along τ.
        assert np.all(np.diff(out.qvals, axis=1) >= -1e-9)

    def test_works_with_mixture_upstream(self, prov, ids_ts):
        # Exercises the mixture_normal ppf path.
        ids, ts = ids_ts(40)
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (40, 3))
        y = X.mean(axis=1) + rng.normal(0, 0.5, 40)
        m = MixtureNormals().fit(X, y)
        d_mix = m.predict_dist(X, ids=ids, timestamps=ts)
        e = EMOS().fit(X, y)
        d_emos = e.predict_dist(X, ids=ids, timestamps=ts)
        pool = LinearPoolDist(deps=("mix", "emos"), n_samples=40)
        pool.fit(X, y, deps_oof={"mix": d_mix, "emos": d_emos})
        # Weights sum to 1.
        np.testing.assert_allclose(pool.weights_.sum(), 1.0)

    def test_rejects_single_dep(self):
        with pytest.raises(ValueError, match="≥2"):
            LinearPoolDist(deps=("only_one",))


# ---------------------------------------------------------------------------
# CDFBoostBracket
# ---------------------------------------------------------------------------


class TestCDFBoostBracket:
    def test_emits_bracket_dist_with_renormed_probs(self, prov, ids_ts):
        _skip_if_no_lightgbm()
        ids, ts = ids_ts(120)
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, (120, 3))
        y = X.mean(axis=1) + rng.normal(0, 0.5, 120)
        d_a = _normal_dist(X.mean(axis=1), np.full(120, 1.0), prov, ids, ts)
        d_b = _normal_dist(X.mean(axis=1) + 0.5, np.full(120, 1.5), prov, ids, ts)
        edges = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        brackets_by_id = {int(k): edges for k in ids}
        node = CDFBoostBracket(
            deps=("a", "b"), brackets_by_id=brackets_by_id,
            n_estimators=30, num_leaves=7, min_child_samples=5,
        )
        node.fit(X, y, ids=ids, deps_oof={"a": d_a, "b": d_b})
        out = node.predict_dist(
            X, ids=ids, timestamps=ts, deps_oof={"a": d_a, "b": d_b},
        )
        assert out.probs.shape == (120, 4)
        np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(out.probs >= 0)

    def test_include_raw_X_grows_feature_count(self, prov, ids_ts):
        _skip_if_no_lightgbm()
        ids, ts = ids_ts(60)
        rng = np.random.default_rng(8)
        X = rng.normal(0, 1, (60, 2))
        y = X[:, 0] + rng.normal(0, 0.3, 60)
        d_a = _normal_dist(X[:, 0], np.full(60, 1.0), prov, ids, ts)
        d_b = _normal_dist(X[:, 1], np.full(60, 1.0), prov, ids, ts)
        edges = np.array([-2.0, 0.0, 2.0])      # 2 bins
        brackets_by_id = {int(k): edges for k in ids}
        node = CDFBoostBracket(
            deps=("a", "b"), brackets_by_id=brackets_by_id,
            n_estimators=20, min_child_samples=5,
            include_raw_X=True,
        )
        node.fit(X, y, ids=ids, deps_oof={"a": d_a, "b": d_b})
        # Feature width = X.shape[1] + K * (B+1) = 2 + 2*3 = 8.
        # Check via the first trained head (skip the "const" sentinel case).
        for kind, model in node.clfs_:
            if kind == "model":
                assert model.n_features_in_ == 8
                break

    def test_rejects_bad_edges(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            CDFBoostBracket(deps=("a",), brackets_by_id={0: np.array([0.0, 2.0, 1.0])})
        with pytest.raises(ValueError, match=r"≥2 bins"):
            CDFBoostBracket(deps=("a",), brackets_by_id={0: np.array([0.0, 1.0])})

    def test_rejects_ragged_B(self):
        with pytest.raises(ValueError, match="uniform bin count"):
            CDFBoostBracket(
                deps=("a",),
                brackets_by_id={
                    0: np.array([0.0, 1.0, 2.0]),
                    1: np.array([0.0, 1.0, 2.0, 3.0]),
                },
            )

    def test_missing_dep_raises_at_fit(self, prov, ids_ts):
        _skip_if_no_lightgbm()
        node = CDFBoostBracket(
            deps=("a", "b"),
            brackets_by_id={int(i): np.array([0.0, 1.0, 2.0, 3.0]) for i in range(5)},
            n_estimators=10,
        )
        with pytest.raises(ValueError, match="deps_oof"):
            node.fit(np.zeros((5, 1)), np.zeros(5), ids=np.arange(5), deps_oof={"a": None})
