"""sklearn-contract upgrade tests (audit item 4).

Pins the user-visible sklearn-style improvements:

- Every estimator importable from the top level (``from bracketlearn import EMOS``).
- BaseEstimator inherits from ``sklearn.base.BaseEstimator``.
- Plain ``(X, y)`` / ``(X,)`` calls work — ids/timestamps auto-filled.
- ``__sklearn_is_fitted__`` flips True after fit.
- ``n_features_in_`` set on fit when X has 2D shape.
- ``feature_names_in_`` set on fit when X is a pandas DataFrame.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_top_level_estimator_imports():
    """Every public estimator is importable from the top-level package."""
    import bracketlearn as bl

    expected = {
        # baselines
        "EmpiricalDistribution", "Persistence",
        # trainers
        "EMOS", "StackedParametric", "SklearnPoint", "MixtureNormals",
        "NGBoostNormal", "QuantileReg", "QuantileForest",
        "CumulativeBinary", "TailSpecialist", "OnlineAggregator",
        "RNNHourly", "CDFBoostBracket", "DistAsFeatures", "LinearPoolDist",
        # lifters / calibrators
        "GlobalResidual", "StudentTResidual", "GARCHResidual",
        "Isotonic", "ConformalCalibrate",
        # composition + search
        "Pipeline", "Stacker", "WalkForward", "PipelineResult", "GridSearch",
        # adapters
        "BracketLadder",
        "BinaryAbove", "BinaryBelow", "Twin", "ThresholdLadder",
        # base
        "BaseEstimator", "clone",
    }
    missing = expected - set(bl.__all__)
    assert not missing, f"missing top-level exports: {sorted(missing)}"
    for name in expected:
        assert getattr(bl, name) is not None, f"{name} is None in bracketlearn.*"


# ---------------------------------------------------------------------------
# sklearn inheritance
# ---------------------------------------------------------------------------


def test_base_estimator_inherits_from_sklearn():
    from sklearn.base import BaseEstimator as SkBase

    from bracketlearn import BaseEstimator
    assert issubclass(BaseEstimator, SkBase)


def test_concrete_estimator_isinstance_of_sklearn_base():
    """A concrete bracketlearn estimator passes isinstance(est, sklearn BaseEstimator)."""
    from sklearn.base import BaseEstimator as SkBase

    from bracketlearn import EMOS, EmpiricalDistribution, QuantileReg
    for cls in (EMOS, EmpiricalDistribution, QuantileReg):
        assert isinstance(cls(), SkBase), f"{cls.__name__} not isinstance of sklearn BaseEstimator"


def test_sklearn_clone_works_on_bracketlearn_estimator():
    """sklearn.base.clone produces an unfitted copy."""
    from sklearn.base import clone as sk_clone

    from bracketlearn import EMOS
    est = EMOS()
    cloned = sk_clone(est)
    assert type(cloned) is EMOS
    assert cloned is not est


# ---------------------------------------------------------------------------
# Optional ids/timestamps (auto-fill)
# ---------------------------------------------------------------------------


def test_sklearn_point_fit_predict_without_ids_or_timestamps():
    from sklearn.linear_model import Ridge

    from bracketlearn import SklearnPoint
    X = np.random.default_rng(0).standard_normal((20, 4))
    y = np.zeros(20)
    sp = SklearnPoint(Ridge())
    sp.fit(X, y)  # NO ids=, NO timestamps=
    p = sp.predict(X)  # NO ids=, NO timestamps=
    assert p.mu.shape == (20,)


def test_empirical_distribution_predict_without_ids():
    from bracketlearn import EmpiricalDistribution
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)
    ed = EmpiricalDistribution()
    ed.fit(X, y)
    d = ed.predict_dist(X)  # no ids/timestamps
    assert d.ids.shape == (30,)
    np.testing.assert_array_equal(d.ids, np.arange(30))


def test_explicit_ids_still_honored_when_provided():
    """If caller passes ids/timestamps, they win over the auto-fill defaults."""
    from sklearn.linear_model import Ridge

    from bracketlearn import SklearnPoint
    X = np.random.default_rng(0).standard_normal((10, 2))
    y = np.zeros(10)
    sp = SklearnPoint(Ridge())
    sp.fit(X, y)
    custom_ids = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    p = sp.predict(X, ids=custom_ids, timestamps=custom_ids.astype(float))
    np.testing.assert_array_equal(p.ids, custom_ids)


# ---------------------------------------------------------------------------
# __sklearn_is_fitted__
# ---------------------------------------------------------------------------


def test_is_fitted_flips_after_fit():
    """check_is_fitted should detect post-fit state via the underscore convention."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    from bracketlearn import EmpiricalDistribution

    ed = EmpiricalDistribution()
    assert not ed.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        check_is_fitted(ed)

    ed.fit(np.zeros((10, 2)), np.arange(10).astype(float))
    assert ed.__sklearn_is_fitted__()
    check_is_fitted(ed)  # no raise


def test_is_fitted_for_sklearn_point():
    from sklearn.linear_model import Ridge

    from bracketlearn import SklearnPoint
    sp = SklearnPoint(Ridge())
    assert not sp.__sklearn_is_fitted__()
    sp.fit(np.zeros((5, 2)), np.zeros(5))
    assert sp.__sklearn_is_fitted__()


# ---------------------------------------------------------------------------
# n_features_in_ / feature_names_in_
# ---------------------------------------------------------------------------


def test_n_features_in_set_by_sklearn_point():
    from sklearn.linear_model import Ridge

    from bracketlearn import SklearnPoint
    sp = SklearnPoint(Ridge())
    sp.fit(np.zeros((8, 5)), np.zeros(8))
    assert sp.n_features_in_ == 5


def test_feature_names_in_set_when_x_is_dataframe():
    pytest.importorskip("pandas")
    import pandas as pd
    from sklearn.linear_model import Ridge

    from bracketlearn import SklearnPoint

    X = pd.DataFrame(
        np.zeros((6, 3)), columns=["a", "b", "c"],
    )
    y = np.zeros(6)
    sp = SklearnPoint(Ridge())
    sp.fit(X, y)
    assert sp.n_features_in_ == 3
    np.testing.assert_array_equal(sp.feature_names_in_, np.array(["a", "b", "c"], dtype=object))
