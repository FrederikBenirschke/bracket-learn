"""Baseline trainer tests.

Pins:

- ``EmpiricalDistribution.predict_dist`` returns the configured taus and
  the *training* y quantiles broadcast across rows. Same dist on every row.
- Weighted fit shifts the recovered quantiles toward heavily-weighted rows.
- ``Persistence`` raises on non-positive lag and on too-few training rows.
- Lag-1 ``Persistence`` predicts the last training y for every inference row.
- Pipeline integration: an ``EmpiricalDistribution`` stage produces a
  finite OOF CRPS; a learned ``QuantileReg`` beats it on a clearly
  signal-bearing X.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.baselines import EmpiricalDistribution, Persistence, PersistenceDist
from bracketlearn.compose import WalkForward
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import Pipeline
from bracketlearn.trainers import QuantileReg


def _signal_dataset(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    y = X.sum(axis=1) * 2.0 + rng.normal(0, 0.5, n)  # strong, learnable signal
    return X, y, np.arange(n), np.arange(n, dtype=float)


class TestEmpiricalDistribution:
    def test_unfit_predict_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            EmpiricalDistribution().predict_dist(
                np.zeros((3, 2)), ids=np.arange(3),
                timestamps=np.arange(3, dtype=float),
            )

    def test_quantiles_match_training_y(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 1000)
        ed = EmpiricalDistribution(taus=(0.1, 0.5, 0.9)).fit(np.zeros((1000, 1)), y)
        expected = np.quantile(y, [0.1, 0.5, 0.9])
        np.testing.assert_allclose(ed.quantiles_, expected)

    def test_dist_broadcasts_across_rows(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 500)
        ed = EmpiricalDistribution(taus=(0.25, 0.5, 0.75)).fit(np.zeros((500, 1)), y)
        dist = ed.predict_dist(
            np.zeros((4, 1)), ids=np.arange(4), timestamps=np.arange(4, dtype=float),
        )
        # All rows must share the same quantile vector.
        for row in dist.qvals:
            np.testing.assert_allclose(row, ed.quantiles_)

    def test_weighted_fit_shifts_quantiles(self):
        """Heavy weights on the high-y rows pull the median upward."""
        n = 200
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        w = np.ones(n)
        w[n // 2:] = 1000.0
        ed_un = EmpiricalDistribution(taus=(0.5,)).fit(np.zeros((n, 1)), y)
        ed_w = EmpiricalDistribution(taus=(0.5,)).fit(np.zeros((n, 1)), y, sample_weight=w)
        assert ed_w.quantiles_[0] > ed_un.quantiles_[0]

    def test_in_pipeline_beats_uniform(self):
        """Empirical is a real baseline — its CRPS must be finite, and a
        properly-fit QuantileReg must beat it on a signal-bearing dataset."""
        X, y, ids, ts = _signal_dataset()
        model = [
            Pipeline([EmpiricalDistribution()], name="emp"),
            Pipeline([QuantileReg(n_estimators=80, random_seed=0)], name="qreg"),
        ]
        result = WalkForward(cv="kfold", n_folds=3, refit_on_full=False).fit_predict(
            model, X, y, ids=ids, timestamps=ts,
        )
        s = result.score(y, metrics=["crps"])
        assert np.isfinite(s["emp"]["crps"])
        assert s["qreg"]["crps"] < s["emp"]["crps"]


class TestPersistence:
    def test_zero_lag_rejected(self):
        with pytest.raises(ValueError, match="lag"):
            Persistence(lag=0).fit(np.zeros((10, 1)), np.zeros(10))

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="lag"):
            Persistence(lag=5).fit(np.zeros((3, 1)), np.array([1., 2., 3.]))

    def test_lag1_predicts_last_train_y(self):
        y = np.array([10., 11., 12., 99.])
        p = Persistence(lag=1).fit(np.zeros((4, 2)), y)
        pred = p.predict(
            np.zeros((5, 2)), ids=np.arange(5), timestamps=np.arange(5, dtype=float),
        )
        np.testing.assert_array_equal(pred.mu, np.full(5, 99.0))

    def test_lag_k_cycles(self):
        """lag=k tiles the last k training y values across the inference horizon."""
        y = np.arange(10, 20, dtype=float)   # ..., 17, 18, 19
        p = Persistence(lag=3).fit(np.zeros((10, 1)), y)
        pred = p.predict(
            np.zeros((7, 1)), ids=np.arange(7), timestamps=np.arange(7, dtype=float),
        )
        # tail = [17, 18, 19]; inference rows pick tail[i mod 3] →
        # 17, 18, 19, 17, 18, 19, 17.
        np.testing.assert_array_equal(
            pred.mu, np.array([17., 18., 19., 17., 18., 19., 17.])
        )

    def test_lag24_diurnal_cycle(self):
        """lag=24 replays the last 24 hours — the diurnal-cycle baseline."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 200)
        p = Persistence(lag=24).fit(np.zeros((200, 1)), y)
        pred = p.predict(
            np.zeros((48, 1)), ids=np.arange(48), timestamps=np.arange(48, dtype=float),
        )
        # First 24 inference hours are exactly y[-24:].
        np.testing.assert_array_equal(pred.mu[:24], y[-24:])
        # Hours 24-47 are exactly y[-24:] again (full cycle repeated).
        np.testing.assert_array_equal(pred.mu[24:48], y[-24:])

    def test_lifted_in_pipeline(self):
        """Wrapped with GlobalResidual, Persistence drops into a pipeline."""
        X, y, ids, ts = _signal_dataset(n=300)
        persist = Pipeline([Persistence(lag=1), GlobalResidual()], name="persist")
        result = WalkForward(
            cv="expanding-window", n_folds=3, refit_on_full=False,
        ).fit_predict(persist, X, y, ids=ids, timestamps=ts)
        s = result.score(y, metrics=["crps"])
        assert np.isfinite(s["persist"]["crps"])


class TestPersistenceDist:
    def test_zero_lag_rejected(self):
        with pytest.raises(ValueError, match="lag"):
            PersistenceDist(lag=0).fit(np.zeros((10, 1)), np.zeros(10))

    def test_too_few_rows_raises(self):
        # need at least lag+2 = 3 rows for lag=1
        with pytest.raises(ValueError, match="rows"):
            PersistenceDist(lag=1).fit(np.zeros((2, 1)), np.array([1., 2.]))

    def test_unfit_predict_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            PersistenceDist().predict_dist(
                np.zeros((3, 1)), ids=np.arange(3),
                timestamps=np.arange(3, dtype=float),
            )

    def test_constant_y_rejected(self):
        """If y is constant, lag-residuals are all zero → σ=0 → raise (Rule #0.5)."""
        with pytest.raises(ValueError, match="non-positive|σ|sigma"):
            PersistenceDist(lag=1).fit(np.zeros((20, 1)), np.full(20, 5.0))

    def test_sigma_estimates_lag_residual_std(self):
        """For a random walk y_t = y_{t-1} + ε, σ̂ should recover Var(ε)^½."""
        rng = np.random.default_rng(0)
        innov = rng.normal(0, 2.0, 5000)
        y = np.cumsum(innov)
        p = PersistenceDist(lag=1).fit(np.zeros((5000, 1)), y)
        # σ should be close to true innovation std 2.0.
        assert abs(p.sigma_ - 2.0) < 0.1

    def test_mu_tiles_tail_y(self):
        """μ rule matches Persistence — last lag y's tiled across inference."""
        y = np.array([10., 11., 12., 13., 14., 99.])
        p = PersistenceDist(lag=1).fit(np.zeros((6, 1)), y)
        dist = p.predict_dist(
            np.zeros((5, 1)), ids=np.arange(5),
            timestamps=np.arange(5, dtype=float),
        )
        np.testing.assert_array_equal(dist.params["mu"], np.full(5, 99.0))
        assert np.all(dist.params["sigma"] == p.sigma_)

    def test_lag_k_cycles(self):
        """lag=k tiles tail-y as in Persistence."""
        # Use random y so lag-3 residuals are non-degenerate (a perfect arange
        # gives constant lag-residual=3 → σ=0 → Rule #0.5 raise).
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 20)
        p = PersistenceDist(lag=3).fit(np.zeros((20, 1)), y)
        dist = p.predict_dist(
            np.zeros((7, 1)), ids=np.arange(7),
            timestamps=np.arange(7, dtype=float),
        )
        tail = y[-3:]
        np.testing.assert_array_equal(
            dist.params["mu"],
            np.array([tail[0], tail[1], tail[2], tail[0], tail[1], tail[2], tail[0]]),
        )

    def test_weighted_sigma_concentrates(self):
        """Heavy weights on a low-noise slice should shrink σ̂."""
        rng = np.random.default_rng(0)
        # First half: σ_innov=0.1; second half: σ_innov=2.0
        innov = np.concatenate([
            rng.normal(0, 0.1, 500),
            rng.normal(0, 2.0, 500),
        ])
        y = np.cumsum(innov)
        # Weight the low-noise slice heavily.
        w = np.concatenate([np.full(500, 100.0), np.full(500, 1.0)])
        p_un = PersistenceDist(lag=1).fit(np.zeros((1000, 1)), y)
        p_w = PersistenceDist(lag=1).fit(np.zeros((1000, 1)), y, sample_weight=w)
        assert p_w.sigma_ < p_un.sigma_

    def test_in_pipeline_yields_finite_crps(self):
        """End-to-end pipeline run on autocorrelated y."""
        rng = np.random.default_rng(0)
        n = 300
        innov = rng.normal(0, 1.0, n)
        y = np.cumsum(innov)
        X = np.zeros((n, 1))
        ids = np.arange(n)
        ts = np.arange(n, dtype=float)
        result = WalkForward(
            cv="expanding-window", n_folds=3, refit_on_full=False,
        ).fit_predict(
            Pipeline([PersistenceDist(lag=1)], name="pdist"),
            X, y, ids=ids, timestamps=ts,
        )
        s = result.score(y, metrics=["crps"])
        assert np.isfinite(s["pdist"]["crps"])
