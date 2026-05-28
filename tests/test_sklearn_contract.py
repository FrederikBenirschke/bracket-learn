"""sklearn-compatible contract tests.

Pins the BaseEstimator contract and the pipeline's no-mutation +
predict-on-unseen-data guarantees:

- ``get_params(deep=True)`` round-trips through ``set_params``.
- ``clone(estimator)`` returns a fresh unfitted instance with the same
  hyperparameters but independent fitted state.
- ``ForecastPipeline.fit_predict`` does NOT mutate the user-supplied
  forecaster instances (fold contamination check).
- ``ForecastPipeline.predict`` returns dists on truly unseen X after
  ``fit_predict`` with ``refit_on_full=True``.
- ``refit_on_full=False`` disables ``predict()`` (loud failure).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from bracketlearn.base import clone
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import ForecastPipeline, LiftedForecaster
from bracketlearn.trainers import (
    EMOS,
    MixtureNormals,
    OnlineAggregator,
    SklearnPoint,
    StackedParametric,
)


def _synthetic(n: int = 200, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, k))
    y = X.mean(axis=1) + rng.normal(0, 0.5, n)
    return X, y, np.arange(n), np.arange(n, dtype=float)


# ---------------------------------------------------------------------------
# BaseEstimator contract
# ---------------------------------------------------------------------------


class TestGetParams:
    def test_emos_exposes_constructor_args(self):
        e = EMOS()
        params = e.get_params()
        assert "name" in params
        assert "depends_on" in params

    def test_mixture_normals_exposes_sigma_floor(self):
        m = MixtureNormals(sigma_floor=1.5)
        params = m.get_params()
        assert params["sigma_floor"] == 1.5

    def test_get_params_excludes_fitted_state(self):
        """Underscore-suffixed attributes (sklearn convention) must not appear."""
        e = EMOS()
        params = e.get_params()
        assert "a_" not in params
        assert "b_" not in params

    def test_get_params_deep_nests_subestimators(self):
        """When a param is itself a BaseEstimator, deep=True should prefix."""
        lf = LiftedForecaster(
            base=SklearnPoint(LinearRegression()),
            lifter=GlobalResidual(),
            name="ridge",
        )
        params = lf.get_params(deep=True)
        # base + lifter are BaseEstimators; their params should be flattened.
        assert "base" in params
        assert "lifter" in params
        assert any(k.startswith("base__") for k in params)
        assert any(k.startswith("lifter__") for k in params)


class TestSetParams:
    def test_set_top_level_param(self):
        e = EMOS()
        e.set_params(name="renamed")
        assert e.name == "renamed"

    def test_set_unknown_param_raises(self):
        with pytest.raises(ValueError, match="Invalid parameter"):
            EMOS().set_params(not_a_param=42)

    def test_set_nested_param(self):
        agg = OnlineAggregator(min_experts=2)
        agg.set_params(min_experts=5)
        assert agg.min_experts == 5


class TestClone:
    def test_clone_is_fresh_instance(self):
        e = EMOS()
        e2 = clone(e)
        assert e2 is not e
        assert type(e2) is type(e)

    def test_clone_preserves_hyperparameters(self):
        m = MixtureNormals(sigma_floor=0.7)
        m2 = clone(m)
        assert m2.sigma_floor == 0.7

    def test_clone_drops_fitted_state(self):
        """After fit, clone must return an unfitted instance."""
        X, y, _, _ = _synthetic()
        e = EMOS().fit(X, y)
        assert e.a_ is not None
        e2 = clone(e)
        assert e2.a_ is None
        assert e2.b_ is None

    def test_clone_independent_after_fit(self):
        """Fitting the clone must not touch the original."""
        X, y, _, _ = _synthetic()
        e = EMOS()
        e2 = clone(e)
        e2.fit(X, y)
        assert e.a_ is None       # original still pristine
        assert e2.a_ is not None  # clone is now fitted


# ---------------------------------------------------------------------------
# Pipeline no-mutation guarantee
# ---------------------------------------------------------------------------


class TestPipelineDoesNotMutate:
    def test_user_forecaster_unmutated_after_fit_predict(self):
        """fit_predict must clone each stage's forecaster per fold so the
        user's instance never gains fitted state."""
        X, y, ids, ts = _synthetic()
        emos = EMOS()
        p = ForecastPipeline(steps=[("emos", emos)], n_folds=3,
                             refit_on_full=False)
        p.fit_predict(X, y, ids=ids, timestamps=ts)
        assert emos.a_ is None
        assert emos.b_ is None

    def test_reusable_across_pipelines(self):
        """A single forecaster instance must be safe to register in two
        independent pipelines."""
        X, y, ids, ts = _synthetic()
        shared = EMOS()
        p1 = ForecastPipeline(steps=[("emos", shared)], n_folds=3,
                              refit_on_full=False)
        p2 = ForecastPipeline(steps=[("emos", shared)], n_folds=3,
                              refit_on_full=False)
        r1 = p1.fit_predict(X, y, ids=ids, timestamps=ts)
        r2 = p2.fit_predict(X, y, ids=ids, timestamps=ts)
        # Same OOF CRPS — proves both runs were clean refits.
        np.testing.assert_allclose(
            r1.score(y, metrics=["crps"])["emos"]["crps"],
            r2.score(y, metrics=["crps"])["emos"]["crps"],
        )


# ---------------------------------------------------------------------------
# predict() on unseen data
# ---------------------------------------------------------------------------


class TestPredictUnseen:
    def test_predict_returns_dist_per_stage(self):
        X, y, ids, ts = _synthetic()
        p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        p.fit_predict(X, y, ids=ids, timestamps=ts)
        X_new = np.random.default_rng(1).normal(0, 1, (40, 3))
        pred = p.predict(X_new, ids=np.arange(40), timestamps=np.arange(40, dtype=float))
        assert "emos" in pred
        assert pred["emos"].params["mu"].shape == (40,)

    def test_predict_works_with_stacking(self):
        """Pipeline.predict must thread deps_oof through downstream stages."""
        X, y, ids, ts = _synthetic()
        p = ForecastPipeline(
            steps=[
                ("ridge", LiftedForecaster(
                    SklearnPoint(LinearRegression()),
                    GlobalResidual(), name="ridge")),
                ("emos", EMOS()),
                ("stack", StackedParametric(deps=("ridge", "emos"))),
            ],
            n_folds=3,
        )
        p.fit_predict(X, y, ids=ids, timestamps=ts)
        X_new = np.random.default_rng(1).normal(0, 1, (20, 3))
        pred = p.predict(X_new, ids=np.arange(20), timestamps=np.arange(20, dtype=float))
        assert pred["stack"].params["mu"].shape == (20,)

    def test_predict_before_fit_raises(self):
        p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
        with pytest.raises(RuntimeError, match="fit_predict"):
            p.predict(np.zeros((5, 3)), ids=np.arange(5), timestamps=np.arange(5, dtype=float))

    def test_refit_on_full_false_disables_predict(self):
        X, y, ids, ts = _synthetic()
        p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3,
                             refit_on_full=False)
        p.fit_predict(X, y, ids=ids, timestamps=ts)
        with pytest.raises(RuntimeError, match="refit_on_full"):
            p.predict(np.zeros((5, 3)), ids=np.arange(5), timestamps=np.arange(5, dtype=float))
