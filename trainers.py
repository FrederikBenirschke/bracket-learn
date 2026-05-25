"""Three v0.1 trainers exercising the framework's three main code paths.

- Ridge       — PointForecaster, wrapped via LiftedForecaster + GlobalResidual.
- EMOS        — DistForecaster (native parametric normal).
- Stacking    — DistForecaster with depends_on; combines upstream OOF.

These are intentionally simple; they exist to prove the framework holds
end-to-end, not to be best-in-class predictors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

import numpy as np

from bracketlearn.forecast import DistributionForecast, PointForecast, ProvenanceMeta


# ---------------------------------------------------------------------------
# SklearnPoint — wrap any sklearn-style regressor as a PointForecaster.
# ---------------------------------------------------------------------------


@dataclass
class SklearnPoint:
    """Adapter: any object with sklearn's fit(X, y) + predict(X) is a
    PointForecaster.

    Works with sklearn.linear_model.{Ridge, Lasso, LinearRegression, ...},
    LightGBM/XGBoost regressors, sklearn ensembles, custom estimators —
    anything matching the sklearn contract.

    Examples:
        SklearnPoint(sklearn.linear_model.Ridge(alpha=1.0))
        SklearnPoint(sklearn.ensemble.GradientBoostingRegressor())
        SklearnPoint(lightgbm.LGBMRegressor(n_estimators=200))
    """

    estimator: Any
    name: str | None = None
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = type(self.estimator).__name__

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Forward sample_weight only if the estimator accepts it.
        if sample_weight is not None:
            try:
                self.estimator.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                self.estimator.fit(X, y)
        else:
            self.estimator.fit(X, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> PointForecast:
        mu = np.asarray(self.estimator.predict(np.asarray(X, dtype=float)), dtype=float)
        prov = ProvenanceMeta(
            forecaster_name=self.name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), datetime.now()),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=None,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=datetime.now(),
        )
        return PointForecast(
            mu=mu,
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# EMOS — Ensemble Model Output Statistics. Native parametric-normal DistForecaster.
# ---------------------------------------------------------------------------


@dataclass
class EMOS:
    """Minimal EMOS:
        μ̂(x) = a + b·x_mean(features),
        σ̂²(x) = c + d·var(features)  (clipped to [eps, ∞))

    where 'features' are the columns of X. Treats X as raw ensemble members
    in the simplest case; in practice users would build X to be the ensemble
    spread/mean directly.

    Coefficients fit by minimising the CRPS of a Gaussian under squared-error
    on (a + b·μ̄ − y) and a separate non-negative LS on the variance.
    Simpler v0.1 fit: OLS for (a, b) and method-of-moments for (c, d).
    """

    name: str = "EMOS"
    depends_on: tuple[str, ...] = ()
    a_: float | None = field(default=None, init=False)
    b_: float | None = field(default=None, init=False)
    c_: float | None = field(default=None, init=False)
    d_: float | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ens_mean = X.mean(axis=1)
        ens_var = X.var(axis=1, ddof=0)

        # OLS for μ: y ≈ a + b·ens_mean
        A_mu = np.column_stack([np.ones_like(ens_mean), ens_mean])
        sol_mu, *_ = np.linalg.lstsq(A_mu, y, rcond=None)
        self.a_, self.b_ = float(sol_mu[0]), float(sol_mu[1])

        # Squared residuals → σ².
        resid = y - (self.a_ + self.b_ * ens_mean)
        r2 = resid ** 2
        # Non-negative LS for variance: r² ≈ c + d·ens_var
        A_var = np.column_stack([np.ones_like(ens_var), ens_var])
        sol_var, *_ = np.linalg.lstsq(A_var, r2, rcond=None)
        self.c_, self.d_ = float(sol_var[0]), float(sol_var[1])
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        if self.a_ is None:
            raise RuntimeError("EMOS.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        ens_mean = X.mean(axis=1)
        ens_var = X.var(axis=1, ddof=0)
        mu = self.a_ + self.b_ * ens_mean
        var = np.clip(self.c_ + self.d_ * ens_var, 1e-6, None)
        sigma = np.sqrt(var)
        prov = ProvenanceMeta(
            forecaster_name=self.name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), datetime.now()),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=None,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=datetime.now(),
            sigma_source="native",
        )
        return DistributionForecast.from_normal(
            mu, sigma, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# Stacking — DistForecaster with depends_on. Linear meta-learner over upstream μ.
# ---------------------------------------------------------------------------


@dataclass
class Stacking:
    """Meta-learner. Features = upstream forecasters' OOF μ (and optionally σ).

    Linear regression of y on stacked upstream μ vectors. Output σ is a
    constant fitted from in-sample residuals (simplification of full
    distributional stacking).

    Pipeline injects deps_oof: dict[name → DistributionForecast] at fit
    time; this Stacking uses the .params['mu'] of each.
    """

    deps: tuple[str, ...]
    name: str = "Stacking"
    weights_: np.ndarray | None = field(default=None, init=False)
    intercept_: float | None = field(default=None, init=False)
    sigma_: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.depends_on = tuple(self.deps)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"Stacking.fit needs deps_oof for {self.depends_on}; got {list(deps_oof or [])}"
            )
        y = np.asarray(y, dtype=float)
        # Stack upstream μ predictions row-aligned. We rely on pipeline
        # passing the test-fold dist for each upstream; align by .ids ordering.
        # v0.1 simplification: assume deps_oof[name] has the same row order
        # as our (X, y).
        cols = []
        for name in self.depends_on:
            d = deps_oof[name]
            if d.backing.value != "parametric":
                raise NotImplementedError(
                    f"Stacking expects parametric upstream; {name} is {d.backing}"
                )
            cols.append(d.params["mu"])
        Z = np.column_stack(cols)  # (N, K)
        # OLS with intercept.
        A = np.column_stack([np.ones(Z.shape[0]), Z])
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.intercept_ = float(sol[0])
        self.weights_ = sol[1:]
        resid = y - (self.intercept_ + Z @ self.weights_)
        self.sigma_ = float(np.std(resid, ddof=1))
        if self.sigma_ <= 0:
            self.sigma_ = 1e-3
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        # At predict time, the pipeline must have re-run the upstream stages
        # on the current X; it passes their dist via deps_oof again.
        if not deps_oof:
            raise ValueError("Stacking.predict_dist needs deps_oof")
        cols = [deps_oof[name].params["mu"] for name in self.depends_on]
        Z = np.column_stack(cols)
        mu = self.intercept_ + Z @ self.weights_
        sigma = np.full_like(mu, self.sigma_)
        prov = ProvenanceMeta(
            forecaster_name=self.name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), datetime.now()),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=None,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=datetime.now(),
            sigma_source="native",
        )
        return DistributionForecast.from_normal(
            mu, sigma, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )
