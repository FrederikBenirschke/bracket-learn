"""Trivial baselines.

Every probabilistic-forecasting paper compares against a baseline that
ignores most of the signal. These two are the floors a real model should
clear by a wide margin — if your fancy quantile-regression-stacked-ensemble
ties ``EmpiricalDistribution``, the features aren't predictive.

- ``EmpiricalDistribution``: emits the marginal distribution of training
  ``y`` as a fixed quantile-backed forecast. Ignores ``X`` completely.
  The "you should always beat this" floor for distributional skill.

- ``Persistence``: ``mu_t = y_{t - lag}`` (defaults to lag=1). Point-only;
  pair with ``GlobalResidual`` (or another ``Lifter``) for distributional
  output. Trivial on i.i.d. data, surprisingly strong on autocorrelated
  series — use it to spot autocorrelation you weren't modelling.

Both inherit ``BaseEstimator`` so they slot into ``ForecastPipeline``
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import DistributionForecast, PointForecast, ProvenanceMeta

_DEFAULT_TAUS: tuple[float, ...] = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
)


@dataclass
class EmpiricalDistribution(BaseEstimator):
    """Marginal-y baseline: ignore X, emit the empirical CDF of training y.

    Stores ``np.quantile(y_train, taus)`` once at fit time; ``predict_dist``
    broadcasts that quantile vector across every row of the inference X.
    No regression, no calibration, no per-row variation.

    Despite being trivial, this is the standard CRPS floor in weather and
    forecasting literature ("climatology"). A model that doesn't beat it
    has zero distributional skill.

    Output: quantile-backed ``DistributionForecast`` with the configured
    tail policy (clip by default).
    """

    taus: tuple[float, ...] = _DEFAULT_TAUS
    name: str = "Empirical"
    depends_on: tuple[str, ...] = ()
    quantiles_: np.ndarray | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        y = np.asarray(y, dtype=float)
        taus = np.asarray(self.taus, dtype=float)
        if sample_weight is None:
            self.quantiles_ = np.quantile(y, taus)
        else:
            # Weighted quantiles via sorted-y / cumulative-weight interpolation.
            w = np.asarray(sample_weight, dtype=float)
            order = np.argsort(y)
            ys, ws = y[order], w[order]
            cum = np.cumsum(ws)
            cum /= cum[-1]
            self.quantiles_ = np.interp(taus, cum, ys)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        from bracketlearn.forecast import TailPolicy, TailRule

        if self.quantiles_ is None:
            raise RuntimeError("EmpiricalDistribution.predict_dist called before fit")
        X = np.asarray(X)
        N = X.shape[0]
        qvals = np.broadcast_to(self.quantiles_, (N, self.quantiles_.shape[0])).copy()
        taus = np.asarray(self.taus, dtype=float)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_quantiles(
            taus=taus, qvals=qvals,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


@dataclass
class Persistence(BaseEstimator):
    """``mu_t = y_{t - lag}``. PointForecaster — wrap with a Lifter for σ.

    At fit time we record the *last* ``lag`` training ``y`` values. At
    predict time we tile that vector across the inference horizon:
    inference row ``i`` gets ``tail_y_[i mod lag]``. This means:

    - lag=1 collapses to "predict the last training y everywhere" (the
      classical random-walk baseline).
    - lag=24 on hourly data emits ``[y_{T-24}, y_{T-23}, ..., y_{T-1},
      y_{T-24}, y_{T-23}, ...]`` — the last full day repeated, which is
      the standard "yesterday's diurnal cycle" baseline used in
      bike-share / load-forecasting benchmarks.

    The cycle is deterministic and ignores any inference y (the model
    sees only X and timestamps). For a strictly causal autoregressive
    forecaster, pair this with ``cv="expanding-window"`` or
    ``"rolling-window"`` — ``"kfold"`` on shuffled rows makes the "last
    y" meaningless.

    Trivial on i.i.d. shuffles; standard time-series baseline whenever
    rows are autocorrelated. Pair with ``GlobalResidual`` (fit on OOF
    residuals) for a proper distributional forecast.
    """

    lag: int = 1
    name: str = "Persistence"
    depends_on: tuple[str, ...] = ()
    tail_y_: np.ndarray | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1; got {self.lag}")
        y = np.asarray(y, dtype=float)
        if y.shape[0] < self.lag:
            raise ValueError(
                f"need at least lag={self.lag} training rows; got {y.shape[0]}"
            )
        self.tail_y_ = y[-self.lag:].copy()
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> PointForecast:
        if self.tail_y_ is None:
            raise RuntimeError("Persistence.predict called before fit")
        X = np.asarray(X)
        N = X.shape[0]
        # Tile the recorded tail across the inference horizon. For lag=1
        # this gives a constant prediction; for lag=24 it repeats yesterday.
        mu = self.tail_y_[np.arange(N) % self.lag]
        prov = ProvenanceMeta.placeholder(self.name)
        return PointForecast(
            mu=mu, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


@dataclass
class PersistenceDist(BaseEstimator):
    """Distributional persistence: ``y_t ~ N(y_{t-lag}, σ̂²)``.

    Same μ rule as ``Persistence`` — tiles the last ``lag`` training y's
    across the inference horizon. σ̂ is the std of in-sample
    persistence residuals ``y_t − y_{t-lag}`` over the training window,
    so it captures the empirical scale of single-lag innovations.

    Use when you need a distributional baseline (CRPS, bracket-prob
    eval) and the series is autocorrelated. For i.i.d. data, σ̂ collapses
    to ``std(y)`` and this becomes a constant-Normal climatology — use
    ``EmpiricalDistribution`` instead, which doesn't pretend symmetry.

    Lag=24 on hourly data gives "yesterday's diurnal + Gaussian noise"
    — a strong baseline for load/temperature forecasting.
    """

    lag: int = 1
    name: str = "PersistenceDist"
    depends_on: tuple[str, ...] = ()
    tail_y_: np.ndarray | None = field(default=None, init=False)
    sigma_: float | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1; got {self.lag}")
        y = np.asarray(y, dtype=float)
        if y.shape[0] < self.lag + 2:
            raise ValueError(
                f"PersistenceDist: need at least lag+2={self.lag + 2} "
                f"training rows to estimate residual σ; got {y.shape[0]}"
            )
        resid = y[self.lag:] - y[:-self.lag]
        if sample_weight is None:
            sigma = float(np.std(resid, ddof=1))
        else:
            w = np.asarray(sample_weight, dtype=float)[self.lag:]
            if w.sum() <= 0:
                raise ValueError(
                    "PersistenceDist.fit: post-lag sample_weight sums to 0"
                )
            mean = float((w * resid).sum() / w.sum())
            var = float((w * (resid - mean) ** 2).sum() / w.sum())
            sigma = float(np.sqrt(var))
        if sigma <= 0:
            raise ValueError(
                "PersistenceDist.fit: residual std is non-positive — "
                "y_t == y_{t-lag} on every training row. The series has "
                "no lag-step innovation; persistence is a deterministic "
                "function and σ is undefined. Refusing to substitute a "
                "floor (Rule #0.5)."
            )
        self.tail_y_ = y[-self.lag:].copy()
        self.sigma_ = sigma
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        if self.tail_y_ is None or self.sigma_ is None:
            raise RuntimeError("PersistenceDist.predict_dist called before fit")
        X = np.asarray(X)
        N = X.shape[0]
        mu = self.tail_y_[np.arange(N) % self.lag]
        sigma = np.full(N, self.sigma_, dtype=float)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_normal(
            mu, sigma,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )
