"""QuantileReg(isotonic_method='sklearn_pava') matches the parent-repo
snowflake's per-row isotonic repair exactly.

The parent repo's prediction_market_weather/ml/trainers/quantile_reg.py
applies sklearn IsotonicRegression (pool-adjacent-violators) row-by-row
for quantile-crossing repair. bracketlearn's default is the faster
np.maximum.accumulate. These methods diverge whenever there are
crossings — sklearn averages across the violator pool, accumulate
clamps to the running max.

This test pins the snowflake-matching mode so the swap stays a true
drop-in.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("lightgbm not installed", allow_module_level=True)

from bracketlearn.trainers import QuantileReg
from bracketlearn.trainers.quantile import _repair_quantile_crossings


_TAUS = (0.05, 0.1, 0.5, 0.9, 0.95)


def _reference_sklearn_repair(qvals, taus):
    from sklearn.isotonic import IsotonicRegression
    out = qvals.copy()
    for i in range(out.shape[0]):
        if np.any(np.diff(out[i]) < 0):
            iso = IsotonicRegression(increasing=True)
            out[i] = iso.fit_transform(taus, out[i])
    return out


def test_repair_function_matches_sklearn_reference():
    taus = np.array(_TAUS)
    rng = np.random.default_rng(0)
    qvals = rng.normal(size=(50, len(_TAUS)))  # random crossings
    ref = _reference_sklearn_repair(qvals, taus)
    out = _repair_quantile_crossings(qvals, taus, "sklearn_pava")
    np.testing.assert_array_equal(out, ref)


def test_repair_function_maximum_accumulate_default():
    taus = np.array(_TAUS)
    rng = np.random.default_rng(1)
    qvals = rng.normal(size=(50, len(_TAUS)))
    expected = np.maximum.accumulate(qvals, axis=1)
    out = _repair_quantile_crossings(qvals, taus, "maximum_accumulate")
    np.testing.assert_array_equal(out, expected)


def test_repair_function_rejects_unknown_method():
    with pytest.raises(ValueError, match="isotonic_method"):
        _repair_quantile_crossings(
            np.zeros((1, 5)), np.array(_TAUS), "monotone_regression",
        )


def test_qreg_default_isotonic_method_is_maximum_accumulate():
    """Backward compat: instantiating QuantileReg() keeps v0.1 behavior."""
    est = QuantileReg()
    assert est.isotonic_method == "maximum_accumulate"


def test_qreg_sklearn_pava_qvals_match_per_row_reference():
    """End-to-end: QuantileReg(sklearn_pava) produces qvals that, after
    bracketing, match the snowflake's per-row repair output exactly."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 6))
    y = X[:, 0] * 2.0 + rng.normal(0, 1.0, 200)

    est = QuantileReg(
        taus=_TAUS, n_estimators=30, learning_rate=0.1,
        num_leaves=7, min_child_samples=5,
        random_seed=42, isotonic_method="sklearn_pava",
    )
    est.fit(X, y)
    dist = est.predict_dist(
        X[:50], ids=np.arange(50), timestamps=np.arange(50, dtype=float),
    )

    # Reference: refit raw qvals, apply per-row sklearn repair.
    import lightgbm as lgb
    raw_models = {
        tau: lgb.LGBMRegressor(
            objective="quantile", alpha=tau, n_estimators=30,
            learning_rate=0.1, num_leaves=7, min_child_samples=5,
            verbose=-1, random_state=42,
        ).fit(X, y)
        for tau in _TAUS
    }
    raw_q = np.column_stack([raw_models[t].predict(X[:50]) for t in _TAUS])
    ref_q = _reference_sklearn_repair(raw_q, np.array(_TAUS))

    np.testing.assert_array_equal(dist.qvals, ref_q)
