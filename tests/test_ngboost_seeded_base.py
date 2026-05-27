"""NGBoostNormal(base_random_state=...) produces reproducible fits and
matches the seeded-default-tree-learner reference.

NGBoost's ``random_state`` only seeds its minibatching / column-subsampling
RNG. Each boosting iteration ``clone(self.Base)`` with the default
``DecisionTreeRegressor(random_state=None)`` — tree split tie-breaking
draws from OS entropy and successive fits with the same NGBoost seed
still produce different μ̂/σ̂.

``base_random_state`` plugs that hole. This test pins both:
(a) two NGBoostNormal fits with both seeds set produce bit-identical μ̂/σ̂;
(b) the bracketlearn fit matches a reference that hand-constructs the
    same Base + NGBRegressor directly (so drift in either side fails loudly).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import ngboost  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("ngboost not installed", allow_module_level=True)

from bracketlearn.trainers import NGBoostNormal


def _make_synthetic(seed: int = 0, n: int = 200, d: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(0, 1.0, n)
    return X, y


def _fit_both_seeded(X, y):
    est = NGBoostNormal(
        n_estimators=50, random_seed=42, base_random_state=42,
    )
    est.fit(X, y)
    dist = est.predict_dist(
        X, ids=np.arange(len(y)), timestamps=np.arange(len(y), dtype=float),
    )
    return dist.params["mu"], dist.params["sigma"]


def test_both_seeds_make_fit_reproducible():
    """Two fits with the same (random_seed, base_random_state) → bit-identical."""
    X, y = _make_synthetic(seed=0, n=200)
    mu1, sigma1 = _fit_both_seeded(X, y)
    mu2, sigma2 = _fit_both_seeded(X, y)
    np.testing.assert_array_equal(mu1, mu2)
    np.testing.assert_array_equal(sigma1, sigma2)


def test_matches_hand_built_seeded_reference():
    """Bracketlearn's seeded fit matches a direct NGBRegressor+seeded-Base
    fit exactly. Pins the wiring: same Base hyperparameters, same seed
    threading, same call shape."""
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor

    X, y = _make_synthetic(seed=1, n=200)

    base = DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        splitter="best",
        random_state=42,
    )
    ref = NGBRegressor(
        Dist=Normal, Base=base,
        n_estimators=50, learning_rate=0.01,
        minibatch_frac=0.5, natural_gradient=True,
        random_state=42, verbose=False,
    )
    ref.fit(X, y)
    pred = ref.pred_dist(X)
    mu_ref = np.asarray(pred.loc, dtype=float)
    sigma_ref = np.maximum(np.asarray(pred.scale, dtype=float), 0.5)

    est = NGBoostNormal(
        n_estimators=50, learning_rate=0.01,
        minibatch_frac=0.5, natural_gradient=True,
        sigma_floor=0.5, random_seed=42, base_random_state=42,
    )
    est.fit(X, y)
    dist = est.predict_dist(
        X, ids=np.arange(len(y)), timestamps=np.arange(len(y), dtype=float),
    )

    np.testing.assert_array_equal(dist.params["mu"], mu_ref)
    np.testing.assert_array_equal(dist.params["sigma"], sigma_ref)


def test_default_base_random_state_is_none():
    """Backward compat: instantiating NGBoostNormal() without base_random_state
    keeps the v0.1 unseeded behavior (non-reproducible but matches users'
    existing expectations)."""
    est = NGBoostNormal()
    assert est.base_random_state is None
