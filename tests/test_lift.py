"""Lifter + Calibrator tests.

Coverage focused on the two implementations that ship in v0.1:
- GlobalResidual: fits one σ from OOF residuals; lift produces a parametric
  normal with the same σ for every row.
- Isotonic: discretises a dist onto bracket edges, fits isotonic regression
  on flattened (p_pred, y_hit) pairs, returns a bracket-backed dist.
- ConformalCalibrate: per-τ offset that shifts quantile-backed dists for
  coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.forecast import (
    Backing,
    DistributionForecast,
    ParametricFamily,
    PointForecast,
    ProvenanceMeta,
    TailPolicy,
    TailRule,
)
from bracketlearn.lift import (
    ConformalCalibrate,
    GARCHResidual,
    GlobalResidual,
    Isotonic,
    StudentTResidual,
)


def _point(mu: np.ndarray, prov: ProvenanceMeta) -> PointForecast:
    n = mu.shape[0]
    return PointForecast(
        mu=mu, ids=np.arange(n), timestamps=np.arange(n, dtype=float),
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# GlobalResidual
# ---------------------------------------------------------------------------


class TestGlobalResidual:
    def test_fit_estimates_residual_sigma(self, prov, rng):
        n = 1000
        mu_hat = rng.normal(0, 1.0, n)
        y = mu_hat + rng.normal(0, 2.5, n)
        gr = GlobalResidual()
        gr.fit(_point(mu_hat, prov), y)
        # Fitted σ should be close to true 2.5.
        assert abs(gr.sigma_ - 2.5) < 0.15

    def test_lift_produces_constant_sigma(self, prov, rng):
        n = 500
        mu_hat = rng.normal(0, 1.0, n)
        y = mu_hat + rng.normal(0, 1.5, n)
        gr = GlobalResidual().fit(_point(mu_hat, prov), y)
        new_mu = np.linspace(-5, 5, 50)
        new_point = _point(new_mu, prov)
        dist = gr.lift(new_point)
        assert dist.backing == Backing.PARAMETRIC
        assert dist.family == ParametricFamily.NORMAL
        assert np.all(dist.params["sigma"] == gr.sigma_)
        np.testing.assert_array_equal(dist.params["mu"], new_mu)

    def test_lift_before_fit_raises(self, prov):
        with pytest.raises(RuntimeError, match="before fit"):
            GlobalResidual().lift(_point(np.array([1.0]), prov))

    def test_rejects_constant_residuals(self, prov):
        """All-equal residuals → σ=0 is degenerate; we raise."""
        mu_hat = np.zeros(10)
        y = np.zeros(10)
        with pytest.raises(ValueError, match="σ|sigma"):
            GlobalResidual().fit(_point(mu_hat, prov), y)


# ---------------------------------------------------------------------------
# StudentTResidual
# ---------------------------------------------------------------------------


class TestStudentTResidual:
    def test_fit_recovers_df_and_scale(self, prov, rng):
        # Draw genuine Student-t residuals with df=5, σ=1.5.
        n = 5000
        from scipy.stats import t as _t
        true_df, true_sigma = 5.0, 1.5
        mu_hat = rng.normal(0, 1.0, n)
        y = mu_hat + _t.rvs(df=true_df, scale=true_sigma, size=n, random_state=0)
        st = StudentTResidual().fit(_point(mu_hat, prov), y)
        assert abs(st.sigma_ - true_sigma) < 0.15
        # df estimation has wide CI; loose tolerance.
        assert 3.0 < st.df_ < 10.0

    def test_lift_produces_student_t_backing(self, prov, rng):
        n = 500
        mu_hat = rng.normal(0, 1.0, n)
        from scipy.stats import t as _t
        y = mu_hat + _t.rvs(df=6.0, scale=1.0, size=n, random_state=1)
        st = StudentTResidual().fit(_point(mu_hat, prov), y)
        new_mu = np.linspace(-3, 3, 40)
        dist = st.lift(_point(new_mu, prov))
        assert dist.backing == Backing.PARAMETRIC
        assert dist.family == ParametricFamily.STUDENT_T
        np.testing.assert_array_equal(dist.params["mu"], new_mu)
        assert np.all(dist.params["sigma"] == st.sigma_)
        assert np.all(dist.params["df"] == st.df_)
        # Variance is finite and consistent with σ² · df / (df - 2).
        expected_var = st.sigma_ ** 2 * st.df_ / (st.df_ - 2.0)
        np.testing.assert_allclose(dist.variance(), expected_var, rtol=1e-9)

    def test_lift_before_fit_raises(self, prov):
        with pytest.raises(RuntimeError, match="before fit"):
            StudentTResidual().lift(_point(np.array([1.0]), prov))

    def test_too_few_residuals_raises(self, prov):
        with pytest.raises(ValueError, match="at least 10"):
            StudentTResidual().fit(_point(np.zeros(5), prov), np.ones(5))


# ---------------------------------------------------------------------------
# GARCHResidual
# ---------------------------------------------------------------------------


class TestGARCHResidual:
    def _simulate_garch(self, T: int, omega: float, alpha: float, beta: float,
                       rng: np.random.Generator) -> np.ndarray:
        sigma2 = np.empty(T)
        r = np.empty(T)
        sigma2[0] = omega / (1.0 - alpha - beta)
        r[0] = rng.normal(0.0, np.sqrt(sigma2[0]))
        for t in range(1, T):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
            r[t] = rng.normal(0.0, np.sqrt(sigma2[t]))
        return r

    def test_fit_recovers_params_approximately(self, prov, rng):
        # Simulate a known GARCH(1,1) process and check MLE recovers params
        # within reasonable tolerance.
        true_omega, true_alpha, true_beta = 0.05, 0.1, 0.85
        T = 2000
        r = self._simulate_garch(T, true_omega, true_alpha, true_beta, rng)
        # mu_hat = 0 so residuals = y = r.
        mu_hat = np.zeros(T)
        y = r
        g = GARCHResidual().fit(_point(mu_hat, prov), y)
        # Persistence (α+β) is the cleanest target — sum tends to be well-identified.
        assert abs((g.alpha_ + g.beta_) - (true_alpha + true_beta)) < 0.05
        # ω is harder; just check it's positive and order-of-magnitude.
        assert 0.0 < g.omega_ < 1.0
        # σ²_next is positive and finite.
        assert g.sigma2_next_ > 0

    def test_lift_produces_constant_per_row_sigma(self, prov, rng):
        T = 500
        mu_hat = np.zeros(T)
        y = self._simulate_garch(T, 0.05, 0.1, 0.85, rng)
        g = GARCHResidual().fit(_point(mu_hat, prov), y)
        new_mu = np.linspace(-1, 1, 20)
        dist = g.lift(_point(new_mu, prov))
        assert dist.backing == Backing.PARAMETRIC
        assert dist.family == ParametricFamily.NORMAL
        # One-step semantics: every row gets the same σ.
        expected_sigma = float(np.sqrt(g.sigma2_next_))
        np.testing.assert_allclose(dist.params["sigma"], expected_sigma, rtol=1e-12)

    def test_student_t_family(self, prov, rng):
        T = 1500
        # Inject fat-tailed shocks to exercise the t-fit branch.
        from scipy.stats import t as _t
        true_omega, true_alpha, true_beta = 0.05, 0.1, 0.85
        sigma2 = np.empty(T)
        r = np.empty(T)
        sigma2[0] = true_omega / (1.0 - true_alpha - true_beta)
        z = _t.rvs(df=5.0, size=T, random_state=2)
        r[0] = np.sqrt(sigma2[0]) * z[0]
        for t in range(1, T):
            sigma2[t] = true_omega + true_alpha * r[t - 1] ** 2 + true_beta * sigma2[t - 1]
            r[t] = np.sqrt(sigma2[t]) * z[t]
        g = GARCHResidual(family="student_t").fit(_point(np.zeros(T), prov), r)
        assert g.df_ is not None and g.df_ > 2.1
        dist = g.lift(_point(np.zeros(5), prov))
        assert dist.family == ParametricFamily.STUDENT_T

    def test_lift_before_fit_raises(self, prov):
        with pytest.raises(RuntimeError, match="before fit"):
            GARCHResidual().lift(_point(np.array([1.0]), prov))

    def test_too_few_residuals_raises(self, prov):
        with pytest.raises(ValueError, match="at least 30"):
            GARCHResidual().fit(_point(np.zeros(10), prov), np.ones(10))


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------


class TestIsotonic:
    def test_fit_transform_roundtrip(self, prov, rng):
        """Calibrated bracket probs should renormalise to 1 per row."""
        n = 500
        edges = np.linspace(-3, 3, 11)
        y = rng.normal(0, 1.0, n)
        # Mis-calibrated forecast: too-narrow Gaussian.
        d_oof = DistributionForecast.from_normal(
            mu=np.zeros(n), sigma=np.full(n, 0.5),
            ids=np.arange(n), timestamps=np.arange(n, dtype=float),
            provenance=prov,
        )
        iso = Isotonic(edges=edges).fit(d_oof, y)

        # Apply to a new dist.
        d_new = DistributionForecast.from_normal(
            mu=np.zeros(20), sigma=np.full(20, 0.5),
            ids=np.arange(20), timestamps=np.arange(20, dtype=float),
            provenance=prov,
        )
        cal = iso.transform(d_new)
        assert cal.backing == Backing.BRACKET
        np.testing.assert_allclose(cal.probs.sum(axis=1), 1.0, atol=1e-9)

    def test_transform_before_fit_raises(self, prov):
        d = DistributionForecast.from_normal(
            mu=np.array([0.0]), sigma=np.array([1.0]),
            ids=np.array([0]), timestamps=np.array([0.0]), provenance=prov,
        )
        with pytest.raises(RuntimeError, match="before fit"):
            Isotonic(edges=np.array([-1.0, 0.0, 1.0])).transform(d)


# ---------------------------------------------------------------------------
# ConformalCalibrate
# ---------------------------------------------------------------------------


class TestConformalCalibrate:
    def _make_quantile_dist(self, n: int, qvals: np.ndarray, prov) -> DistributionForecast:
        taus = np.array([0.1, 0.5, 0.9])
        return DistributionForecast.from_quantiles(
            taus=taus, qvals=np.tile(qvals, (n, 1)),
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.arange(n), timestamps=np.arange(n, dtype=float),
            provenance=prov,
        )

    def test_fit_learns_per_tau_offset(self, prov, rng):
        n = 500
        # Forecast quantiles biased high by 1.0 — calibration should learn
        # offsets ≈ +1.0 across τ to recover coverage.
        y = rng.normal(0, 1.0, n)
        from scipy.stats import norm
        taus = np.array([0.1, 0.5, 0.9])
        true_q = norm.ppf(taus)
        biased_q = true_q + 1.0
        d_oof = DistributionForecast.from_quantiles(
            taus=taus, qvals=np.tile(biased_q, (n, 1)),
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.arange(n), timestamps=np.arange(n, dtype=float),
            provenance=prov,
        )
        cc = ConformalCalibrate().fit(d_oof, y)
        assert cc.fitted_
        np.testing.assert_allclose(cc.offsets_, 1.0, atol=0.2)

    def test_transform_rejects_non_quantile_backing(self, prov):
        cc = ConformalCalibrate()
        cc.offsets_ = np.array([0.0, 0.0, 0.0])
        cc.fitted_ = True
        d_normal = DistributionForecast.from_normal(
            mu=np.array([0.0]), sigma=np.array([1.0]),
            ids=np.array([0]), timestamps=np.array([0.0]), provenance=prov,
        )
        with pytest.raises(ValueError, match="quantile"):
            cc.transform(d_normal)
