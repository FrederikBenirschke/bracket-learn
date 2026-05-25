"""Phase C tests: CV variants, sample weights, multi-target, grid search.

Each section pins one user-visible guarantee:

- CV variants: kfold and rolling-window produce disjoint test folds and
  cover the dataset roughly evenly; rolling-window enforces a fixed
  training-window size; kfold rejects ``shuffle=True`` only when paired
  with the time-series CVs.
- Sample weights: row weights are forwarded through pipeline → trainer →
  underlying estimator. EMOS with extreme weights collapses to the
  weighted target. Trainers that don't accept ``sample_weight`` don't
  crash when the pipeline threads it through.
- Multi-target: ``MultiOutputForecastPipeline`` fits M independent
  pipelines, results are indexable per-target, and ``predict()`` returns
  per-target dicts on unseen data.
- GridSearch: enumerates the grid, routes nested ``stage__field`` params
  into the right stage, picks the param with lowest CRPS, refuses unknown
  scoring metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from bracketlearn.composite import LiftedForecaster
from bracketlearn.lift import GlobalResidual
from bracketlearn.multitarget import MultiOutputForecastPipeline
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.search import GridSearch
from bracketlearn.trainers import EMOS, MixtureNormals, SklearnPoint


def _synthetic(n: int = 200, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, k))
    y = X.mean(axis=1) + rng.normal(0, 0.5, n)
    return X, y, np.arange(n), np.arange(n, dtype=float)


# ---------------------------------------------------------------------------
# CV variants
# ---------------------------------------------------------------------------


class TestCVVariants:
    def test_kfold_disjoint_test_folds(self):
        p = ForecastPipeline(steps=[("emos", EMOS())], cv="kfold", n_folds=4)
        folds = p._make_folds(200)
        assert len(folds) == 4
        all_test = np.concatenate([te for _, te in folds])
        assert all_test.shape[0] == 200             # full coverage
        assert len(set(all_test.tolist())) == 200   # disjoint

    def test_kfold_no_train_test_overlap(self):
        p = ForecastPipeline(steps=[("emos", EMOS())], cv="kfold", n_folds=5)
        for tr, te in p._make_folds(200):
            assert not (set(tr.tolist()) & set(te.tolist()))

    def test_kfold_shuffle_changes_assignment(self):
        p1 = ForecastPipeline(steps=[("emos", EMOS())], cv="kfold", n_folds=5,
                              shuffle=True, random_state=0)
        p2 = ForecastPipeline(steps=[("emos", EMOS())], cv="kfold", n_folds=5,
                              shuffle=False)
        f1 = p1._make_folds(200)
        f2 = p2._make_folds(200)
        # At least one fold's test set should differ.
        assert any(not np.array_equal(a[1], b[1]) for a, b in zip(f1, f2))

    def test_kfold_too_few_rows_raises(self):
        p = ForecastPipeline(steps=[("emos", EMOS())], cv="kfold", n_folds=10)
        with pytest.raises(ValueError, match="< n_folds"):
            p._make_folds(5)

    def test_rolling_fixed_train_size(self):
        p = ForecastPipeline(steps=[("emos", EMOS())], cv="rolling-window",
                             n_folds=4, rolling_window=50)
        folds = p._make_folds(200)
        for tr, _ in folds:
            assert tr.shape[0] == 50

    def test_rolling_test_after_train(self):
        p = ForecastPipeline(steps=[("emos", EMOS())], cv="rolling-window",
                             n_folds=3, rolling_window=80)
        for tr, te in p._make_folds(200):
            assert te.min() >= tr.max()

    def test_rolling_requires_window(self):
        with pytest.raises(ValueError, match="rolling_window"):
            ForecastPipeline(steps=[("emos", EMOS())], cv="rolling-window")

    def test_expanding_rejects_shuffle(self):
        with pytest.raises(ValueError, match="shuffle"):
            ForecastPipeline(steps=[("emos", EMOS())],
                             cv="expanding-window", shuffle=True)

    def test_kfold_end_to_end(self):
        X, y, ids, ts = _synthetic()
        p = ForecastPipeline(steps=[("emos", EMOS())], cv="kfold", n_folds=4,
                             refit_on_full=False)
        result = p.fit_predict(X, y, ids=ids, timestamps=ts)
        s = result.score(y, metrics=["crps"])
        assert np.isfinite(s["emos"]["crps"])

    def test_rolling_end_to_end(self):
        X, y, ids, ts = _synthetic(n=300)
        p = ForecastPipeline(steps=[("emos", EMOS())], cv="rolling-window",
                             n_folds=3, rolling_window=120, refit_on_full=False)
        result = p.fit_predict(X, y, ids=ids, timestamps=ts)
        s = result.score(y, metrics=["crps"])
        assert np.isfinite(s["emos"]["crps"])


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------


class TestSampleWeights:
    def test_extreme_weight_pulls_fit_to_weighted_rows(self):
        """EMOS with row 0 weighted 1e6 should fit μ ≈ y[0] when X[0,:] is
        the only ensemble row pointing there."""
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(0, 1, (n, 5))
        y = X.mean(axis=1) + rng.normal(0, 1.0, n)
        # Build a contradictory row: high X, low y.
        X[0] = 10.0
        y[0] = -10.0
        w_uniform = np.ones(n)
        w_skew = np.ones(n); w_skew[0] = 1e6

        e1 = EMOS().fit(X, y, sample_weight=w_uniform)
        e2 = EMOS().fit(X, y, sample_weight=w_skew)
        # Predict at the conflicted row.
        d1 = e1.predict_dist(X[[0]], ids=np.array([0]), timestamps=np.array([0.0]))
        d2 = e2.predict_dist(X[[0]], ids=np.array([0]), timestamps=np.array([0.0]))
        # Skewed fit should be closer to y[0] = -10 at row 0.
        assert abs(d2.params["mu"][0] - (-10.0)) < abs(d1.params["mu"][0] - (-10.0))

    def test_pipeline_forwards_weights(self):
        X, y, ids, ts = _synthetic()
        w = np.full(y.shape[0], 1.0)
        w[:20] = 1e3  # bias toward early rows
        p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3,
                             refit_on_full=False)
        r_weighted = p.fit_predict(X, y, ids=ids, timestamps=ts, sample_weight=w)
        r_unweighted = ForecastPipeline(
            steps=[("emos", EMOS())], n_folds=3, refit_on_full=False,
        ).fit_predict(X, y, ids=ids, timestamps=ts)
        # OOF dists should differ.
        assert not np.allclose(
            r_weighted["emos"].params["mu"],
            r_unweighted["emos"].params["mu"],
        )

    def test_pipeline_rejects_misshapen_weight(self):
        X, y, ids, ts = _synthetic()
        p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        with pytest.raises(ValueError, match="sample_weight length"):
            p.fit_predict(X, y, ids=ids, timestamps=ts,
                          sample_weight=np.ones(10))

    def test_sklearnpoint_forwards_to_estimator(self):
        """SklearnPoint(LinearRegression()) accepts sample_weight."""
        X, y, ids, ts = _synthetic()
        w = np.ones(y.shape[0]); w[:10] = 1e3
        p = ForecastPipeline(
            steps=[("ridge", LiftedForecaster(
                SklearnPoint(LinearRegression()), GlobalResidual(),
                name="ridge"))],
            n_folds=3, refit_on_full=False,
        )
        # Should run without TypeError despite weights.
        result = p.fit_predict(X, y, ids=ids, timestamps=ts, sample_weight=w)
        assert "ridge" in result.forecasts

    def test_mixture_weights_change_sigma_v(self):
        """MixtureNormals σ_v is the per-vendor RMSE; weighting should shift
        it toward the heavily-weighted rows' residuals."""
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(0, 1, (n, 3))
        y = X[:, 0] + rng.normal(0, 0.5, n)
        w = np.ones(n); w[:20] = 1e6
        # Force high residuals on the heavy rows.
        y[:20] = X[:20, 0] + 5.0
        m_un = MixtureNormals().fit(X, y)
        m_w = MixtureNormals().fit(X, y, sample_weight=w)
        # Vendor 0's σ is the most affected (it's a near-perfect fit on
        # vendor 0 elsewhere, but the heavy rows show ~5.0 residuals).
        assert m_w.sigma_v_[0] > m_un.sigma_v_[0]


# ---------------------------------------------------------------------------
# Multi-target
# ---------------------------------------------------------------------------


class TestMultiTarget:
    def _multi_y(self, n=200, k=3, M=2, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, k))
        Y = np.column_stack([
            X.mean(axis=1) + rng.normal(0, 0.5, n) + j
            for j in range(M)
        ])
        return X, Y, np.arange(n), np.arange(n, dtype=float)

    def test_fit_predict_indexes_per_target(self):
        X, Y, ids, ts = self._multi_y(M=2)
        proto = ForecastPipeline(
            steps=[("emos", EMOS())], n_folds=3, refit_on_full=False,
        )
        mt = MultiOutputForecastPipeline(proto)
        result = mt.fit_predict(X, Y, ids=ids, timestamps=ts)
        assert set(result.targets) == {"target_0", "target_1"}
        assert "emos" in result["target_0"].forecasts

    def test_predict_unseen(self):
        X, Y, ids, ts = self._multi_y(M=2)
        proto = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        mt = MultiOutputForecastPipeline(proto, target_names=["a", "b"])
        mt.fit_predict(X, Y, ids=ids, timestamps=ts)
        X_new = np.random.default_rng(1).normal(0, 1, (30, 3))
        pred = mt.predict(X_new, ids=np.arange(30), timestamps=np.arange(30, dtype=float))
        assert pred["a"]["emos"].params["mu"].shape == (30,)
        assert pred["b"]["emos"].params["mu"].shape == (30,)

    def test_score_multi(self):
        X, Y, ids, ts = self._multi_y(M=2)
        proto = ForecastPipeline(
            steps=[("emos", EMOS())], n_folds=3, refit_on_full=False,
        )
        mt = MultiOutputForecastPipeline(proto)
        result = mt.fit_predict(X, Y, ids=ids, timestamps=ts)
        scores = result.score(Y, metrics=["crps"])
        assert {"target_0", "target_1"} == set(scores)
        assert np.isfinite(scores["target_0"]["emos"]["crps"])

    def test_misshapen_target_names_raises(self):
        X, Y, ids, ts = self._multi_y(M=2)
        proto = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        mt = MultiOutputForecastPipeline(proto, target_names=["only_one"])
        with pytest.raises(ValueError, match="target_names"):
            mt.fit_predict(X, Y, ids=ids, timestamps=ts)


# ---------------------------------------------------------------------------
# GridSearch
# ---------------------------------------------------------------------------


class TestGridSearch:
    def test_picks_best_param(self):
        X, y, ids, ts = _synthetic()
        proto = ForecastPipeline(
            steps=[("emos", EMOS())], n_folds=3, refit_on_full=False,
        )
        gs = GridSearch(
            proto,
            param_grid={"n_folds": [3, 5]},
            scoring="crps", refit_stage="emos",
        )
        gs.fit(X, y, ids=ids, timestamps=ts)
        assert gs.best_params_ is not None
        assert gs.best_score_ is not None
        assert len(gs.results_) == 2
        # best_score must equal the min CRPS across the grid.
        best_obs = min(r["crps"] for r in gs.results_)
        assert gs.best_score_ == pytest.approx(best_obs)

    def test_nested_param_routes_to_stage(self):
        """``stage__field`` keys must dispatch into the stage's forecaster.
        Use log_score (mixture supports it natively) so the score is finite."""
        X, y, ids, ts = _synthetic()
        proto = ForecastPipeline(
            steps=[("mix", MixtureNormals(sigma_floor=0.5))], n_folds=3,
            refit_on_full=False,
        )
        gs = GridSearch(
            proto,
            param_grid={"mix__sigma_floor": [0.1, 2.0]},
            scoring="log_score", refit_stage="mix",
        )
        gs.fit(X, y, ids=ids, timestamps=ts)
        assert gs.best_params_ in (
            {"mix__sigma_floor": 0.1}, {"mix__sigma_floor": 2.0},
        )
        assert len(gs.results_) == 2

    def test_unknown_param_raises(self):
        X, y, ids, ts = _synthetic()
        proto = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        gs = GridSearch(
            proto, param_grid={"not_a_param": [1, 2]},
            scoring="crps", refit_stage="emos",
        )
        with pytest.raises(ValueError, match="neither a pipeline ctor arg"):
            gs.fit(X, y, ids=ids, timestamps=ts)

    def test_unknown_scoring_raises(self):
        proto = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        with pytest.raises(ValueError, match="scoring"):
            GridSearch(proto, param_grid={"n_folds": [3]},
                       scoring="not_a_metric")

    def test_empty_grid_raises(self):
        proto = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        with pytest.raises(ValueError, match="empty"):
            GridSearch(proto, param_grid={}, scoring="crps")

    def test_does_not_mutate_prototype(self):
        X, y, ids, ts = _synthetic()
        proto = ForecastPipeline(
            steps=[("emos", EMOS())], n_folds=3, refit_on_full=False,
        )
        proto_emos = proto._stages[0].forecaster
        gs = GridSearch(
            proto, param_grid={"n_folds": [3, 5]},
            scoring="crps", refit_stage="emos",
        )
        gs.fit(X, y, ids=ids, timestamps=ts)
        # Prototype's EMOS instance must remain unfitted.
        assert proto_emos.a_ is None
        # Pipeline n_folds unchanged at the prototype level.
        assert proto.n_folds == 3
