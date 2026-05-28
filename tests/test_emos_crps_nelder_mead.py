"""EMOS(fit_method='crps_nelder_mead') matches the parent-repo snowflake
exactly.

The parent repo's prediction_market_weather/ml/trainers/emos.py uses a
specific CRPS-Nelder-Mead fit (Gneiting & Raftery 2005, Gneiting et al.
2005). Session 5 of the bracketlearn v0.3 refactor adds this algorithm
as a first-class option on bracketlearn.EMOS so the snowflake can be
retired without changing model numerics. This test pins the
floating-point equivalence so the drop-in stays a true drop-in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import norm

from bracketlearn.trainers import EMOS

# ---------------------------------------------------------------------------
# Reference implementation — copied verbatim from the snowflake at
# prediction_market_weather/ml/trainers/emos.py (functions _gaussian_crps,
# _crps_loss, _fit_emos). Kept here as a frozen reference so any drift
# in either side fails this test loudly.
# ---------------------------------------------------------------------------


def _ref_gaussian_crps(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0)
                    + 2.0 * norm.pdf(z)
                    - 1.0 / math.sqrt(math.pi))


def _ref_crps_loss(params, ens_mean, ens_std, y):
    a, b, c, d = params
    mu = a + b * ens_mean
    var = math.exp(c) + math.exp(d) * (ens_std ** 2)
    sigma = np.sqrt(var)
    return float(np.mean(_ref_gaussian_crps(mu, sigma, y)))


def _ref_fit_emos(ens_mean, ens_std, y):
    X = np.column_stack([np.ones_like(ens_mean), ens_mean])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, b0 = float(beta[0]), float(beta[1])
    resid_var = float(np.var(y - (a0 + b0 * ens_mean)))
    mean_spread_sq = float(np.mean(ens_std ** 2))
    c0 = math.log(max(resid_var / 2.0, 1e-6))
    d0 = math.log(max(resid_var / (2.0 * max(mean_spread_sq, 1e-6)), 1e-6))
    x0 = np.array([a0, b0, c0, d0], dtype=float)
    res = minimize(
        _ref_crps_loss, x0,
        args=(ens_mean, ens_std, y),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 5000},
    )
    return res.x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_synthetic(seed: int = 0, n: int = 200):
    """Realistic ensemble forecast data: ens_mean ≈ y + bias,
    ens_std varies row to row, residuals correlate with spread."""
    rng = np.random.default_rng(seed)
    y = 70.0 + rng.normal(0, 8.0, n)             # max-temperature-ish
    ens_mean = y + rng.normal(0, 1.5, n)         # 1.5°F bias noise
    ens_std = 0.5 + 2.5 * rng.random(n)          # spread in [0.5, 3.0]
    return ens_mean, ens_std, y


def test_fit_matches_reference_implementation_floating_point():
    """Same input → bracketlearn EMOS(crps_nelder_mead) and the frozen
    reference fit produce identical (a, b, c, d) parameters."""
    ens_mean, ens_std, y = _make_synthetic(seed=0, n=300)

    # Reference: standalone scipy minimize call.
    a_ref, b_ref, c_ref, d_ref = _ref_fit_emos(ens_mean, ens_std, y)

    # bracketlearn EMOS in aggregates mode + crps_nelder_mead.
    X = np.column_stack([ens_mean, ens_std])
    est = EMOS(fit_method="crps_nelder_mead", input_form="aggregates")
    est.fit(X, y)

    np.testing.assert_allclose(est.a_, a_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(est.b_, b_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(est.c_, c_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(est.d_, d_ref, rtol=0, atol=1e-12)


def test_predict_mu_sigma_matches_reference():
    """Per-row μ̂ and σ̂ on a held-out set match the reference exactly."""
    ens_mean_tr, ens_std_tr, y_tr = _make_synthetic(seed=1, n=300)
    ens_mean_te, ens_std_te, _ = _make_synthetic(seed=2, n=120)

    a, b, c, d = _ref_fit_emos(ens_mean_tr, ens_std_tr, y_tr)
    mu_ref = a + b * ens_mean_te
    var_ref = math.exp(c) + math.exp(d) * (ens_std_te ** 2)
    sigma_ref = np.sqrt(var_ref)

    X_tr = np.column_stack([ens_mean_tr, ens_std_tr])
    X_te = np.column_stack([ens_mean_te, ens_std_te])
    est = EMOS(fit_method="crps_nelder_mead", input_form="aggregates")
    est.fit(X_tr, y_tr)
    dist = est.predict_dist(
        X_te, ids=np.arange(len(ens_mean_te)),
        timestamps=np.arange(len(ens_mean_te), dtype=float),
    )

    np.testing.assert_allclose(dist.mu, mu_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dist.sigma, sigma_ref, rtol=0, atol=1e-12)


def test_ols_fit_method_still_default():
    """Backward compat: instantiating EMOS() without specifying fit_method
    keeps the v0.1 OLS algorithm."""
    est = EMOS()
    assert est.fit_method == "ols"
    assert est.input_form == "members"


def test_crps_nelder_mead_rejects_sample_weight():
    """The CRPS variant does not (yet) thread sample weights through the
    optimiser — raise loudly per Rule #0.5 rather than silently dropping."""
    ens_mean, ens_std, y = _make_synthetic(seed=3, n=80)
    X = np.column_stack([ens_mean, ens_std])
    est = EMOS(fit_method="crps_nelder_mead", input_form="aggregates")
    with pytest.raises(NotImplementedError, match="sample_weight"):
        est.fit(X, y, sample_weight=np.ones(len(y)))


def test_aggregates_input_form_rejects_wrong_shape():
    est = EMOS(input_form="aggregates")
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        est.fit(np.zeros((5, 3)), np.zeros(5))


def test_aggregates_input_form_rejects_nonpositive_std():
    est = EMOS(input_form="aggregates")
    X = np.column_stack([np.zeros(5), np.array([1.0, 1.0, 0.0, 1.0, 1.0])])
    with pytest.raises(ValueError, match="strictly positive"):
        est.fit(X, np.zeros(5))
