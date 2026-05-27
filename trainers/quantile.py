"""Quantile-backed DistForecasters.

QuantileReg (per-tau LightGBM heads), QuantileForest (forest-leaf empirical).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    DistributionForecast,
    ProvenanceMeta,
)

# ---------------------------------------------------------------------------
# QuantileReg — per-τ LightGBM heads. Quantile-backed DistForecaster.
# ---------------------------------------------------------------------------


_DEFAULT_QUANTILES: tuple[float, ...] = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
)


@dataclass
class QuantileReg(BaseEstimator):
    """Per-τ LightGBM quantile-regression heads.

    Fits one LightGBM regressor per τ ∈ taus with objective='quantile',
    alpha=τ; isotonic-repairs predicted quantiles per row.

    Output: quantile-backed DistributionForecast with TailRule.clip on both
    sides (extreme bracket mass stays at the outermost quantile; switch to
    gpd/gaussian_match in v0.3).
    """

    taus: tuple[float, ...] = _DEFAULT_QUANTILES
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    random_seed: int | None = None
    name: str = "QuantileReg"
    depends_on: tuple[str, ...] = ()
    models_: dict[float, Any] = field(default_factory=dict, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.models_ = {}
        for tau in self.taus:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=tau,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                verbose=-1,
                random_state=self.random_seed,
            )
            if sample_weight is not None:
                m.fit(X, y, sample_weight=sample_weight)
            else:
                m.fit(X, y)
            self.models_[tau] = m
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        from bracketlearn.forecast import TailPolicy, TailRule

        if not self.models_:
            raise RuntimeError("QuantileReg.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        qvals = np.column_stack([self.models_[t].predict(X) for t in self.taus])
        # Repair crossings across rows in one vectorised pass.
        qvals = np.maximum.accumulate(qvals, axis=1)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native", random_seed=self.random_seed)
        return DistributionForecast.from_quantiles(
            taus=np.asarray(self.taus, dtype=float),
            qvals=qvals,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# QuantileForest — single random forest, quantiles from leaf empirical CDFs.
# ---------------------------------------------------------------------------


@dataclass
class QuantileForest(BaseEstimator):
    """Quantile Regression Forest (Meinshausen 2006).

    Fits one quantile-forest model; predicts per-row quantiles at fixed taus.
    """

    taus: tuple[float, ...] = _DEFAULT_QUANTILES
    n_estimators: int = 300
    min_samples_leaf: int = 10
    random_seed: int | None = 0
    name: str = "QuantileForest"
    depends_on: tuple[str, ...] = ()
    model_: Any = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        from quantile_forest import RandomForestQuantileRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.model_ = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=self.random_seed,
        )
        if sample_weight is not None:
            self.model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X, y)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        from bracketlearn.forecast import TailPolicy, TailRule

        if self.model_ is None:
            raise RuntimeError("QuantileForest.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        qpred = np.asarray(self.model_.predict(X, quantiles=list(self.taus)),
                           dtype=float)
        # quantile-forest can return shape (N, Q) or (N,) depending on Q==1; coerce.
        if qpred.ndim == 1:
            qpred = qpred.reshape(-1, 1)
        qpred = np.maximum.accumulate(qpred, axis=1)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native", random_seed=self.random_seed)
        return DistributionForecast.from_quantiles(
            taus=np.asarray(self.taus, dtype=float),
            qvals=qpred,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


