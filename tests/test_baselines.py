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

from bracketlearn.baselines import EmpiricalDistribution, Persistence
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import ForecastPipeline, LiftedForecaster
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
        p = ForecastPipeline(
            steps=[
                ("emp", EmpiricalDistribution()),
                ("qreg", QuantileReg(n_estimators=80, random_seed=0)),
            ],
            cv="kfold", n_folds=3, refit_on_full=False,
        )
        result = p.fit_predict(X, y, ids=ids, timestamps=ts)
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
        p = ForecastPipeline(
            steps=[
                ("persist", LiftedForecaster(
                    Persistence(lag=1), GlobalResidual(), name="persist",
                )),
            ],
            cv="expanding-window", n_folds=3, refit_on_full=False,
        )
        result = p.fit_predict(X, y, ids=ids, timestamps=ts)
        s = result.score(y, metrics=["crps"])
        assert np.isfinite(s["persist"]["crps"])
