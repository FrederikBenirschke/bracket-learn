"""Tests for the v0.3 score additions:

- ``crps_mixture_normal``: MC CRPS for mixture-of-normals. Collapses to
  ``crps_gaussian`` when weights concentrate on one component.
- ``log_score_quantile``: piecewise-linear CDF → constant density per bin.
  Reproduces the Gaussian density when the quantile grid is dense.
- ``to_point``: every backing returns a 1-D mean/median/mode that lands
  near the true centre of a simple test distribution.

All three are now reachable from ``PipelineResult.score``, so we also
check that ``MixtureNormals`` and ``QuantileReg`` produce *finite*
CRPS and log_score columns end-to-end.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from bracketlearn.compose import WalkForward
from bracketlearn.forecast import DistributionForecast, ProvenanceMeta, TailPolicy, TailRule
from bracketlearn.pipeline import Pipeline
from bracketlearn.score import (
    crps_gaussian,
    crps_mixture_normal,
    log_score_gaussian,
    log_score_quantile,
    to_point,
)
from bracketlearn.trainers import MixtureNormals, QuantileReg


def _prov() -> ProvenanceMeta:
    return ProvenanceMeta(
        forecaster_name="test", forecaster_version="0.0",
        fit_window=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
        fold_idx=None, calibration_set_hash=None, random_seed=0,
        code_sha="dev", feature_matrix_hash="-", created_at=datetime.now(),
        sigma_source="native",
    )


def _normal_dist(mu, sigma):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ids = np.arange(mu.size)
    ts = ids.astype(float)
    return DistributionForecast.from_normal(mu, sigma, ids=ids, timestamps=ts,
                                            provenance=_prov())


def _quantile_dist(taus, qvals):
    qvals = np.asarray(qvals, dtype=float)
    ids = np.arange(qvals.shape[0])
    ts = ids.astype(float)
    return DistributionForecast.from_quantiles(
        taus=np.asarray(taus, dtype=float), qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=_prov(),
    )


def _mixture_dist(weights, mus, sigmas):
    weights = np.asarray(weights, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    ids = np.arange(weights.shape[0])
    ts = ids.astype(float)
    return DistributionForecast.from_mixture_normal(
        weights=weights, mus=mus, sigmas=sigmas,
        ids=ids, timestamps=ts, provenance=_prov(),
    )


def _bracket_dist(edges, probs):
    edges = np.asarray(edges, dtype=float)
    probs = np.asarray(probs, dtype=float)
    ids = np.arange(probs.shape[0])
    ts = ids.astype(float)
    return DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=_prov(),
    )


# ---------------------------------------------------------------------------
# crps_mixture_normal
# ---------------------------------------------------------------------------


class TestCRPSMixtureNormal:
    def test_collapsed_mixture_matches_gaussian(self):
        """When all weight sits on one component, MC-CRPS must approach the
        closed-form Gaussian CRPS within MC tolerance."""
        n = 20
        rng = np.random.default_rng(0)
        mu = rng.normal(0, 1, n)
        sigma = np.full(n, 1.5)
        y = rng.normal(0, 1, n)
        # Mixture with weight=1 on the first component.
        weights = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        mus = np.column_stack([mu, mu + 5, mu - 5])
        sigmas = np.column_stack([sigma, sigma, sigma])
        mix = _mixture_dist(weights, mus, sigmas)
        norm = _normal_dist(mu, sigma)
        crps_closed = crps_gaussian(norm, y)
        crps_mc = crps_mixture_normal(mix, y, n_samples=4000, random_state=0)
        # MC noise is roughly 1/sqrt(n_samples) of the dispersion scale.
        np.testing.assert_allclose(crps_mc, crps_closed, atol=0.05)

    def test_is_nonnegative(self):
        weights = np.array([[0.4, 0.6]] * 10)
        mus = np.tile(np.array([0.0, 2.0]), (10, 1))
        sigmas = np.tile(np.array([1.0, 1.0]), (10, 1))
        mix = _mixture_dist(weights, mus, sigmas)
        y = np.linspace(-3, 5, 10)
        c = crps_mixture_normal(mix, y, n_samples=1000)
        assert (c >= 0).all()

    def test_widening_components_raises_crps(self):
        """A more dispersed mixture should be punished more by CRPS."""
        weights = np.array([[0.5, 0.5]] * 5)
        mus = np.tile(np.array([0.0, 0.0]), (5, 1))
        sigmas_narrow = np.tile(np.array([0.5, 0.5]), (5, 1))
        sigmas_wide = np.tile(np.array([3.0, 3.0]), (5, 1))
        narrow = _mixture_dist(weights, mus, sigmas_narrow)
        wide = _mixture_dist(weights, mus, sigmas_wide)
        y = np.zeros(5)
        c_n = crps_mixture_normal(narrow, y, n_samples=4000, random_state=0).mean()
        c_w = crps_mixture_normal(wide, y, n_samples=4000, random_state=0).mean()
        assert c_w > c_n


# ---------------------------------------------------------------------------
# log_score_quantile
# ---------------------------------------------------------------------------


class TestLogScoreQuantile:
    def test_uniform_quantiles_recover_uniform_density(self):
        """If qvals matches the (interior) taus directly, the implied CDF
        is the identity on (0, 1) so density is 1.0 in every bin and
        log_score → 0 on y values inside the support."""
        taus = np.linspace(0.05, 0.95, 19)
        qvals = np.tile(taus, (5, 1))     # row r has q_τ = τ
        d = _quantile_dist(taus, qvals)
        y = np.full(5, 0.5)
        ls = log_score_quantile(d, y)
        # density per bin = dτ / dq = 1.0 → log_score = 0 within the support.
        np.testing.assert_allclose(ls, 0.0, atol=1e-6)

    def test_in_support_density_scales_with_range(self):
        """A narrow quantile range has higher density per bin than a wide
        one (smaller dq for the same dτ). For y inside both supports the
        narrow dist should report a *lower* log-score (higher density)."""
        taus = np.linspace(0.05, 0.95, 19)
        narrow = np.tile(np.linspace(-1, 1, taus.size), (3, 1))
        wide = np.tile(np.linspace(-10, 10, taus.size), (3, 1))
        n = _quantile_dist(taus, narrow)
        w = _quantile_dist(taus, wide)
        y_in = np.zeros(3)                # centre of both supports
        ls_n = log_score_quantile(n, y_in).mean()
        ls_w = log_score_quantile(w, y_in).mean()
        assert ls_n < ls_w                # narrow has higher density at 0

    def test_matches_gaussian_logscore_on_dense_quantile_grid(self):
        """A quantile dist sampled from N(0, 1) at 99 taus should give a
        log-score close to log_score_gaussian on the same y values."""
        taus = np.linspace(0.01, 0.99, 99)
        from scipy.stats import norm
        z = norm.ppf(taus)
        qvals = np.tile(z, (10, 1))         # 10 identical rows
        qdist = _quantile_dist(taus, qvals)
        ndist = _normal_dist(np.zeros(10), np.ones(10))
        y = np.linspace(-1.5, 1.5, 10)
        ls_q = log_score_quantile(qdist, y)
        ls_n = log_score_gaussian(ndist, y)
        # Piecewise-constant density approximation is coarse but should be
        # within ~0.2 nats of the true Gaussian.
        np.testing.assert_allclose(ls_q, ls_n, atol=0.2)


# ---------------------------------------------------------------------------
# to_point
# ---------------------------------------------------------------------------


class TestToPoint:
    def test_normal_mean_equals_mu(self):
        d = _normal_dist(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]))
        np.testing.assert_array_equal(to_point(d, how="mean"), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(to_point(d, how="median"), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(to_point(d, how="mode"), [1.0, 2.0, 3.0])

    def test_mixture_mean_is_weighted_sum(self):
        weights = np.array([[0.3, 0.7]])
        mus = np.array([[0.0, 10.0]])
        sigmas = np.array([[1.0, 1.0]])
        d = _mixture_dist(weights, mus, sigmas)
        np.testing.assert_allclose(to_point(d, how="mean"), [7.0])

    def test_mixture_mode_is_highest_weight_component(self):
        weights = np.array([[0.2, 0.8]])
        mus = np.array([[-5.0, 5.0]])
        sigmas = np.array([[1.0, 1.0]])
        d = _mixture_dist(weights, mus, sigmas)
        np.testing.assert_array_equal(to_point(d, how="mode"), [5.0])

    def test_bracket_mean_uses_midpoints(self):
        edges = np.array([0.0, 1.0, 2.0])
        probs = np.array([[0.5, 0.5], [0.0, 1.0]])
        d = _bracket_dist(edges, probs)
        # Mids = [0.5, 1.5]; means = [0.5*0.5 + 0.5*1.5, 0*0.5 + 1*1.5] = [1.0, 1.5]
        np.testing.assert_allclose(to_point(d, how="mean"), [1.0, 1.5])

    def test_quantile_median(self):
        taus = np.array([0.1, 0.5, 0.9])
        qvals = np.array([[-1, 0, 1], [0, 5, 10]], dtype=float)
        d = _quantile_dist(taus, qvals)
        np.testing.assert_array_equal(to_point(d, how="median"), [0.0, 5.0])

    def test_quantile_mean_close_to_symmetric_centre(self):
        """A symmetric quantile grid should give mean ≈ median."""
        taus = np.linspace(0.05, 0.95, 19)
        qvals = np.tile(np.linspace(-3, 3, taus.size), (2, 1))
        d = _quantile_dist(taus, qvals)
        np.testing.assert_allclose(to_point(d, how="mean"), [0.0, 0.0], atol=0.1)

    def test_invalid_how_raises(self):
        d = _normal_dist(np.array([0.0]), np.array([1.0]))
        with pytest.raises(ValueError, match="how="):
            to_point(d, how="argmax")


# ---------------------------------------------------------------------------
# Pipeline integration — the n/a's are gone
# ---------------------------------------------------------------------------


def _signal_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, n)
    return X, y, np.arange(n), np.arange(n, dtype=float)


class TestPipelineNoMoreNaN:
    def test_mixture_crps_is_finite(self):
        X, y, ids, ts = _signal_dataset()
        result = WalkForward(
            cv="kfold", n_folds=3, shuffle=True, random_state=0, refit_on_full=False,
        ).fit_predict(
            Pipeline([MixtureNormals()], name="mix"), X, y, ids=ids, timestamps=ts,
        )
        s = result.score(y, metrics=["crps", "log_score"])
        assert np.isfinite(s["mix"]["crps"])
        assert np.isfinite(s["mix"]["log_score"])

    def test_quantile_log_score_is_finite(self):
        X, y, ids, ts = _signal_dataset()
        result = WalkForward(
            cv="kfold", n_folds=3, shuffle=True, random_state=0, refit_on_full=False,
        ).fit_predict(
            Pipeline([QuantileReg(n_estimators=40, random_seed=0)], name="qreg"),
            X, y, ids=ids, timestamps=ts,
        )
        s = result.score(y, metrics=["crps", "log_score"])
        assert np.isfinite(s["qreg"]["crps"])
        assert np.isfinite(s["qreg"]["log_score"])
