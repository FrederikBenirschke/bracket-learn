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
)
from bracketlearn.lift import ConformalCalibrate, GlobalResidual, Isotonic
from bracketlearn.tail import TailPolicy, TailRule


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
        with pytest.raises(NotImplementedError, match="quantile"):
            cc.transform(d_normal)
