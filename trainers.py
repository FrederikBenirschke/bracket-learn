"""v0.1 trainers covering the framework's main protocol shapes.

Core building blocks:
- SklearnPoint    — wrap any sklearn-style regressor as a PointForecaster.
- EMOS            — native parametric-normal DistForecaster.
- Stacking        — DistForecaster with depends_on (linear meta-learner).
- NGBoostNormal   — non-linear EMOS via NGBoost; native parametric normal.
- MixtureNormals  — per-vendor Gaussian mixture; native parametric mixture.

Convenience builders pre-wrap common combinations:
- ridge()           — RidgeCV + GlobalResidual via LiftedForecaster.
- market_ols()      — OLS + GlobalResidual.
- emos_calibrated() — EMOS + Isotonic via CalibratedForecaster (needs edges).

These trainers exist to prove the framework holds across protocol shapes;
they mirror the algorithmic essence of the equivalents in
prediction_market_weather/ml/trainers/ but without the parquet/registry I/O.
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


# ---------------------------------------------------------------------------
# NGBoostNormal — non-linear EMOS. μ̂ and σ̂ both boosted as f(X).
# ---------------------------------------------------------------------------


@dataclass
class NGBoostNormal:
    """Native parametric-normal DistForecaster backed by NGBoost.

    Mirrors prediction_market_weather/ml/trainers/ngboost_normal.py: boosts
    (μ̂, σ̂) as non-linear functions of the full feature vector so
    dispersion can be regime-conditional.

    sigma_floor clamps σ̂ above a minimum (NGBoost can collapse σ̂ → 0 on
    overfit folds; the floor mirrors the original trainer's SIGMA_FLOOR).
    """

    n_estimators: int = 400
    learning_rate: float = 0.01
    minibatch_frac: float = 0.5
    natural_gradient: bool = True
    sigma_floor: float = 0.5
    random_seed: int | None = None
    name: str = "NGBoostNormal"
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
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.model_ = NGBRegressor(
            Dist=Normal,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            natural_gradient=self.natural_gradient,
            random_state=self.random_seed,
            verbose=False,
        )
        self.model_.fit(X, y)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        if self.model_ is None:
            raise RuntimeError("NGBoostNormal.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        dist = self.model_.pred_dist(X)
        mu = np.asarray(dist.loc, dtype=float)
        sigma = np.maximum(np.asarray(dist.scale, dtype=float), self.sigma_floor)
        prov = ProvenanceMeta(
            forecaster_name=self.name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), datetime.now()),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=self.random_seed,
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
# MixtureNormals — one Gaussian component per "vendor" (column of X).
# ---------------------------------------------------------------------------


@dataclass
class MixtureNormals:
    """Per-vendor Gaussian mixture.

        p(y | x) = (1/K) Σ_v N(y; x_v, σ_v²)

    where x_v is the v-th column of X (vendor's point forecast) and σ_v is
    that vendor's train-slice RMSE against y. Equal weights over the K
    columns of X.

    Mirrors prediction_market_weather/ml/trainers/mixture_normals.py except
    we treat X as already-curated vendor columns: the original handled
    missing values per row, which is a data-shape concern that doesn't fit
    the (N, K) dense array contract here.

    Rule #0.5: rows with all-NaN columns or all-zero variances raise.
    """

    name: str = "MixtureNormals"
    depends_on: tuple[str, ...] = ()
    sigma_floor: float = 0.5
    sigma_v_: np.ndarray | None = field(default=None, init=False)
    K_: int | None = field(default=None, init=False)

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
        if X.ndim != 2:
            raise ValueError(f"MixtureNormals expects 2-D X; got shape {X.shape}")
        K = X.shape[1]
        diffs = X - y[:, None]
        sigma_v = np.sqrt(np.mean(diffs ** 2, axis=0))
        sigma_v = np.maximum(sigma_v, self.sigma_floor)
        if np.any(~np.isfinite(sigma_v)):
            raise RuntimeError(
                f"MixtureNormals: non-finite σ_v after fit ({sigma_v}); "
                f"check X has no NaNs"
            )
        self.sigma_v_ = sigma_v
        self.K_ = K
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        if self.sigma_v_ is None:
            raise RuntimeError("MixtureNormals.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.K_:
            raise ValueError(
                f"MixtureNormals: predict X has K={X.shape[1]}, train had K={self.K_}"
            )
        N = X.shape[0]
        mus = X
        sigmas = np.broadcast_to(self.sigma_v_, (N, self.K_)).copy()
        weights = np.full((N, self.K_), 1.0 / self.K_)
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
        return DistributionForecast.from_mixture_normal(
            weights=weights, mus=mus, sigmas=sigmas,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# Convenience builders — pre-wrap common (forecaster, lifter/calibrator) combos.
# ---------------------------------------------------------------------------


def ridge(
    *,
    alphas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    name: str = "ridge",
) -> Any:
    """RidgeCV + GlobalResidual wrapped in LiftedForecaster.

    Picks α from `alphas` via leave-one-out CV on the inner-fit slice.
    Returns a LiftedForecaster ready to register with a ForecastPipeline.
    """
    from sklearn.linear_model import RidgeCV

    from bracketlearn.composite import LiftedForecaster
    from bracketlearn.lift import GlobalResidual

    return LiftedForecaster(
        base=SklearnPoint(RidgeCV(alphas=np.asarray(alphas))),
        lifter=GlobalResidual(family="normal"),
        name=name,
    )


def market_ols(*, name: str = "market_ols") -> Any:
    """Plain OLS + GlobalResidual. Mirrors market_ols Q2 (target = realized).

    Q1 (target = market_implied) is the same model fit with a different
    target — out of scope for bracketlearn (the target choice is a calling
    convention, not a model class).
    """
    from sklearn.linear_model import LinearRegression

    from bracketlearn.composite import LiftedForecaster
    from bracketlearn.lift import GlobalResidual

    return LiftedForecaster(
        base=SklearnPoint(LinearRegression()),
        lifter=GlobalResidual(family="normal"),
        name=name,
    )


def emos_calibrated(*, edges: np.ndarray, name: str = "emos_calibrated") -> Any:
    """EMOS wrapped with Isotonic on the given bracket ladder.

    `edges` defines the ladder used for isotonic calibration. The pipeline
    fits the isotonic on a held-out tail of each training fold and applies
    it to the test fold.
    """
    from bracketlearn.composite import CalibratedForecaster
    from bracketlearn.lift import Isotonic

    return CalibratedForecaster(
        forecaster=EMOS(),
        calibrator=Isotonic(edges=np.asarray(edges, dtype=float)),
        name=name,
    )


# ---------------------------------------------------------------------------
# QuantileReg — per-τ LightGBM heads. Quantile-backed DistForecaster.
# ---------------------------------------------------------------------------


_DEFAULT_QUANTILES: tuple[float, ...] = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
)


def _isotonic_repair_row(qvals: np.ndarray) -> np.ndarray:
    """In-place fix for non-monotone quantile crossings (PAV across τ)."""
    return np.maximum.accumulate(qvals)


@dataclass
class QuantileReg:
    """Per-τ LightGBM quantile-regression heads.

    Mirrors prediction_market_weather/ml/trainers/quantile_reg.py: fits one
    LightGBM regressor per τ ∈ taus with objective='quantile', alpha=τ;
    isotonic-repairs predicted quantiles per row.

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
        from bracketlearn.tail import TailPolicy, TailRule

        if not self.models_:
            raise RuntimeError("QuantileReg.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        qvals = np.column_stack([self.models_[t].predict(X) for t in self.taus])
        # Repair crossings row-by-row.
        for i in range(qvals.shape[0]):
            qvals[i] = _isotonic_repair_row(qvals[i])
        prov = ProvenanceMeta(
            forecaster_name=self.name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), datetime.now()),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=self.random_seed,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=datetime.now(),
            sigma_source="native",
        )
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
class QuantileForest:
    """Quantile Regression Forest (Meinshausen 2006).

    Mirrors prediction_market_weather/ml/trainers/quantile_forest.py: fits
    one quantile-forest model; predicts per-row quantiles at fixed taus.
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
        self.model_.fit(X, y)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        from bracketlearn.tail import TailPolicy, TailRule

        if self.model_ is None:
            raise RuntimeError("QuantileForest.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        qpred = np.asarray(self.model_.predict(X, quantiles=list(self.taus)),
                           dtype=float)
        # quantile-forest can return shape (N, Q) or (N,) depending on Q==1; coerce.
        if qpred.ndim == 1:
            qpred = qpred.reshape(-1, 1)
        for i in range(qpred.shape[0]):
            qpred[i] = _isotonic_repair_row(qpred[i])
        prov = ProvenanceMeta(
            forecaster_name=self.name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), datetime.now()),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=self.random_seed,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=datetime.now(),
            sigma_source="native",
        )
        return DistributionForecast.from_quantiles(
            taus=np.asarray(self.taus, dtype=float),
            qvals=qpred,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# CumulativeBinary — one classifier on (X ⊕ cutpoint) → 1[y ≤ cutpoint].
# ---------------------------------------------------------------------------


@dataclass
class CumulativeBinary:
    """Single LightGBM binary classifier on augmented features.

    Mirrors prediction_market_weather/ml/trainers/cumulative_binary.py: fits
    one classifier over (X, cutpoint) → 1[y ≤ cutpoint] using a fixed grid
    of cutpoints (derived from the bracket ladder); at predict time, queries
    P(y ≤ k) for each k in the grid and emits a quantile-backed dist where
    taus = the cdf values at the configured cutpoints.

    Constructed with a fixed `cutpoints` array — typically the interior
    edges of the eval bracket ladder.
    """

    cutpoints: np.ndarray
    n_estimators: int = 80
    learning_rate: float = 0.05
    num_leaves: int = 7
    min_child_samples: int = 100
    monotone: bool = True
    name: str = "CumulativeBinary"
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
        import lightgbm as lgb

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        cuts = np.asarray(self.cutpoints, dtype=float)
        if cuts.size == 0:
            raise ValueError("CumulativeBinary requires at least one cutpoint")
        N = X.shape[0]
        # Build augmented training set: each train row × each cutpoint.
        X_aug = np.repeat(X, cuts.size, axis=0)
        cut_col = np.tile(cuts, N).reshape(-1, 1)
        X_aug = np.hstack([X_aug, cut_col])
        y_aug = (np.repeat(y, cuts.size) <= np.tile(cuts, N)).astype(int)
        n_feat = X_aug.shape[1]
        monotone = [0] * (n_feat - 1) + [1] if self.monotone else None
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            objective="binary",
            verbose=-1,
            monotone_constraints=monotone,
        )
        self.model_.fit(X_aug, y_aug)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        from bracketlearn.tail import TailPolicy, TailRule

        if self.model_ is None:
            raise RuntimeError("CumulativeBinary.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        cuts = np.asarray(self.cutpoints, dtype=float)
        N = X.shape[0]
        # Build query: each test row × each cutpoint.
        X_aug = np.repeat(X, cuts.size, axis=0)
        cut_col = np.tile(cuts, N).reshape(-1, 1)
        X_aug = np.hstack([X_aug, cut_col])
        proba = self.model_.predict_proba(X_aug)[:, 1].reshape(N, cuts.size)
        # Isotonic repair per row (non-monotonicity is rare with monotone=True
        # but the LGBM contraint isn't strict for binary).
        for i in range(N):
            proba[i] = np.maximum.accumulate(proba[i])
        # taus = proba values at the cutpoints (per row), but quantile-backed
        # storage assumes shared taus across rows. Reverse the roles: store
        # cutpoints as qvals, and use proba as per-row taus — but that breaks
        # the shared-tau invariant too. Instead, define a *fixed tau grid*
        # using rank of cutpoints (e.g. (1..K)/(K+1)) and let qvals = cutpoints.
        # Simpler: emit a bracket-backed dist on edges = [-inf-equivalent,
        # cuts, +inf-equivalent], probs = diff(0, proba_at_cuts, 1).
        #
        # To match the existing trainer's semantics (bracket probs from
        # cumulative classifier output), emit a bracket dist over
        # [cuts[0]-pad, cuts[0], cuts[1], ..., cuts[-1], cuts[-1]+pad].
        # The two outer bins absorb the tail mass under TailRule.clip.
        pad = max(1.0, float(np.diff(cuts).mean()) if cuts.size > 1 else 1.0)
        edges = np.concatenate([[cuts[0] - pad], cuts, [cuts[-1] + pad]])
        # cdf at the inner edges = proba; at the outer edges = 0 and 1.
        cdf_at_edges = np.column_stack([
            np.zeros(N),
            proba,
            np.ones(N),
        ])
        probs = np.diff(cdf_at_edges, axis=1)
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        probs = probs / row_sum
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
        return DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# TailSpecialist — EMOS body + LightGBM tail classifiers (DistForecaster).
# ---------------------------------------------------------------------------


@dataclass
class TailSpecialist:
    """Gaussian body (from upstream EMOS μ̂/σ̂) + LightGBM tail classifiers.

    Mirrors prediction_market_weather/ml/trainers/tail_specialist.py. depends_on
    a parametric-normal upstream (typically named 'emos') and a ladder
    (edges) — fits two binary classifiers for the first/last bracket
    indicators, then rescales the Gaussian body probs to (1 - p_lo - p_hi).
    """

    edges: np.ndarray
    upstream: str = "emos"
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    name: str = "TailSpecialist"
    clf_lo_: Any = field(default=None, init=False)
    clf_hi_: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.depends_on = (self.upstream,)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        if not deps_oof or self.upstream not in deps_oof:
            raise ValueError(
                f"TailSpecialist.fit needs deps_oof[{self.upstream!r}]"
            )
        edges = np.asarray(self.edges, dtype=float)
        if edges.size < 4:
            raise ValueError(
                f"TailSpecialist needs ladder with ≥3 brackets (4 edges); got {edges.size}"
            )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # First-bracket indicator: y < edges[1]. Last-bracket: y >= edges[-2].
        y_lo = (y < edges[1]).astype(int)
        y_hi = (y >= edges[-2]).astype(int)
        if y_lo.sum() < 5 or y_hi.sum() < 5:
            raise RuntimeError(
                f"TailSpecialist: too few tail positives "
                f"(lo={int(y_lo.sum())}, hi={int(y_hi.sum())})"
            )
        common = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            objective="binary",
            class_weight="balanced",
            verbose=-1,
        )
        self.clf_lo_ = lgb.LGBMClassifier(**common)
        self.clf_lo_.fit(X, y_lo)
        self.clf_hi_ = lgb.LGBMClassifier(**common)
        self.clf_hi_.fit(X, y_hi)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        if self.clf_lo_ is None:
            raise RuntimeError("TailSpecialist.predict_dist called before fit")
        if not deps_oof or self.upstream not in deps_oof:
            raise ValueError(
                f"TailSpecialist.predict_dist needs deps_oof[{self.upstream!r}]"
            )
        upstream = deps_oof[self.upstream]
        # Discretise EMOS dist onto ladder.
        edges = np.asarray(self.edges, dtype=float)
        cdf_hi = upstream.cdf(edges[1:])
        cdf_lo = upstream.cdf(edges[:-1])
        body_probs = np.clip(cdf_hi - cdf_lo, 0.0, 1.0)
        N, B = body_probs.shape
        # Tail probs from classifiers.
        X = np.asarray(X, dtype=float)
        p_lo = np.clip(self.clf_lo_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        p_hi = np.clip(self.clf_hi_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        # Rescale inner bins (1..B-1, i.e. all but first and last) to
        # (1 - p_lo - p_hi).
        inner = body_probs[:, 1:-1]
        inner_sum = inner.sum(axis=1, keepdims=True)
        body_total = np.maximum(0.0, 1.0 - p_lo - p_hi)[:, None]
        # Avoid divide-by-zero: if inner_sum=0, redistribute body_total uniformly.
        safe = inner_sum > 0
        inner_scaled = np.where(
            safe, inner * (body_total / np.where(safe, inner_sum, 1.0)),
            body_total / max(inner.shape[1], 1),
        )
        probs = np.concatenate(
            [p_lo[:, None], inner_scaled, p_hi[:, None]], axis=1,
        )
        # Final renorm as a safety net.
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        probs = probs / row_sum
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
        return DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# OnlineAggregator — sleeping-experts AdaHedge (PointForecaster).
# ---------------------------------------------------------------------------


@dataclass
class OnlineAggregator:
    """AdaHedge over forecast experts (columns of X).

    Mirrors prediction_market_weather/ml/trainers/online_aggregator.py:
    walks rows in order, treats each column of X as an expert's point
    prediction (NaN = asleep on that row), accumulates per-expert squared
    losses, updates the mixability-gap learning rate, and produces an
    aggregated prediction per row.

    Predict-time behavior mirrors the original's `predict_inference_side`
    path: at fit time the final weight vector is snapshotted; at predict
    time we compute weighted mean over awake experts, renormalising the
    snapshot weights to the active subset. This is what the original ships
    to inference — pure online behavior during fit, snapshot-and-apply at
    predict.

    Output: PointForecaster — pair with GlobalResidual (or other Lifter)
    for distribution coverage. Composition is explicit, not baked in.
    """

    min_experts: int = 2
    name: str = "OnlineAggregator"
    depends_on: tuple[str, ...] = ()
    final_w_: np.ndarray | None = field(default=None, init=False)
    K_: int | None = field(default=None, init=False)

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
        if X.ndim != 2:
            raise ValueError(f"OnlineAggregator expects 2-D X (rows × experts); got {X.shape}")
        T, K = X.shape
        L = np.zeros(K)
        delta = 0.0
        eta = float("inf")
        log_K = float(np.log(max(K, 2)))
        last_w_per_expert = np.zeros(K)
        seen_per_expert = np.zeros(K, dtype=int)

        for t in range(T):
            f_t = X[t]
            y_t = y[t]
            awake = ~np.isnan(f_t)
            n_awake = int(awake.sum())
            if n_awake < self.min_experts:
                continue
            awake_idx = np.where(awake)[0]
            L_awake = L[awake_idx]
            w_awake = self._softmin(eta, L_awake)
            last_w_per_expert[awake_idx] = w_awake
            seen_per_expert[awake_idx] += 1
            f_awake = f_t[awake_idx]
            pred = float(np.dot(w_awake, f_awake))
            ell_awake = (f_awake - y_t) ** 2
            hedge_loss_t = float(np.dot(w_awake, ell_awake))
            mix_loss_t = self._mix_loss(eta, w_awake, ell_awake)
            delta += max(0.0, hedge_loss_t - mix_loss_t)
            if delta > 0:
                eta = log_K / delta
            L[awake_idx] += ell_awake

        if seen_per_expert.sum() == 0:
            raise RuntimeError(
                f"OnlineAggregator: no rows had ≥{self.min_experts} awake experts"
            )
        # Final weights: per AdaHedge semantics, take the *current* posterior
        # over all experts (those never awake get 0). Renormalise.
        w_final = self._softmin(eta, L)
        # Zero out experts never seen — guards against giving cold-start
        # vendors any weight at predict time.
        w_final[seen_per_expert == 0] = 0.0
        s = w_final.sum()
        if s <= 0:
            raise RuntimeError("OnlineAggregator: final weight vector sums to 0")
        self.final_w_ = w_final / s
        self.K_ = K
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> PointForecast:
        if self.final_w_ is None:
            raise RuntimeError("OnlineAggregator.predict called before fit")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.K_:
            raise ValueError(
                f"OnlineAggregator: predict X has K={X.shape[1]}, train had K={self.K_}"
            )
        N = X.shape[0]
        mu = np.full(N, np.nan)
        for t in range(N):
            awake = ~np.isnan(X[t])
            if int(awake.sum()) < self.min_experts:
                continue
            w = self.final_w_[awake]
            s = w.sum()
            if s <= 0:
                continue
            mu[t] = float(np.dot(w / s, X[t][awake]))
        # Rule #0.5: leftover NaNs are a real coverage hole — raise.
        if np.isnan(mu).any():
            n_miss = int(np.isnan(mu).sum())
            raise RuntimeError(
                f"OnlineAggregator.predict: {n_miss}/{N} rows had < {self.min_experts} awake experts"
            )
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
            mu=mu, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )

    @staticmethod
    def _softmin(eta: float, losses: np.ndarray) -> np.ndarray:
        if not np.isfinite(eta):
            w = np.ones_like(losses)
            return w / w.sum()
        scaled = -eta * losses
        scaled = scaled - scaled.max()
        w = np.exp(scaled)
        return w / w.sum()

    @staticmethod
    def _mix_loss(eta: float, weights: np.ndarray, losses: np.ndarray) -> float:
        if not np.isfinite(eta):
            return float(losses.min())
        z = -eta * losses
        z_max = z.max()
        return float(-(np.log(np.sum(weights * np.exp(z - z_max))) + z_max) / eta)


# ---------------------------------------------------------------------------
# RNNHourly — GRU on (24, C) hourly tensor (PointForecaster).
# ---------------------------------------------------------------------------


@dataclass
class RNNHourly:
    """Tiny GRU on a (24, C) hourly tensor → residual-corrected point forecast.

    Mirrors prediction_market_weather/ml/trainers/rnn_hourly.py: GRU reads
    the 24-hour sequence, concatenates a station embedding (if station_ids
    is passed via the `station_ids` argument at fit), MLP head outputs a
    scalar residual to the channel-0 max (HRRR's max-T baseline). Final
    prediction = channel_0_max + residual.

    Expects X.ndim == 3 with shape (N, T, C). For weather: T=24 hours,
    C=6 (temperature, dewpoint, RH, wind, cloud, CAPE).

    `baseline_channel`: which channel's max provides the residual anchor
    (default 0 = temperature, matching the original trainer).

    `station_ids` (optional, passed at fit/predict via `meta=...` arg):
    integer-encoded station for the embedding. If absent, embedding is
    skipped and the model uses GRU only.

    Output: PointForecaster — pair with GlobalResidual (or other Lifter)
    for distribution coverage.
    """

    hidden: int = 32
    embed: int = 4
    dropout: float = 0.3
    epochs: int = 200
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-4
    baseline_channel: int = 0
    seed: int = 17
    name: str = "RNNHourly"
    depends_on: tuple[str, ...] = ()
    model_: Any = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    std_: np.ndarray | None = field(default=None, init=False)
    n_stations_: int | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
        station_ids: np.ndarray | None = None,
    ) -> Self:
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"RNNHourly expects 3-D X (N, T, C); got {X.shape}")
        N, T, C = X.shape
        # Residual target = realized - baseline_channel_max.
        baseline = X[:, :, self.baseline_channel].max(axis=1)
        residual = y - baseline

        # Per-channel normaliser fit on train only.
        flat = X.reshape(-1, C)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.mean_, self.std_ = mean.astype(np.float32), std

        if station_ids is not None:
            sid = np.asarray(station_ids, dtype=np.int64)
            if sid.shape[0] != N:
                raise ValueError(f"station_ids length {sid.shape[0]} != N={N}")
            n_stations = int(sid.max()) + 1
        else:
            sid = np.zeros(N, dtype=np.int64)
            n_stations = 1
        self.n_stations_ = n_stations

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model_ = _HourlyGRU(
            n_channels=C, n_stations=n_stations,
            hidden=self.hidden, embed=self.embed, dropout=self.dropout,
        )
        opt = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.SmoothL1Loss(beta=1.0)

        Xn = (X - self.mean_) / self.std_
        Xt = torch.from_numpy(Xn.astype(np.float32))
        yt = torch.from_numpy(residual.astype(np.float32))
        st = torch.from_numpy(sid)

        for _ in range(self.epochs):
            perm = torch.randperm(N)
            self.model_.train()
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self.model_(Xt[idx], st[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        station_ids: np.ndarray | None = None,
    ) -> PointForecast:
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch

        if self.model_ is None:
            raise RuntimeError("RNNHourly.predict called before fit")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"RNNHourly.predict expects 3-D X; got {X.shape}")
        N = X.shape[0]
        baseline = X[:, :, self.baseline_channel].max(axis=1)
        Xn = (X - self.mean_) / self.std_
        if station_ids is not None:
            sid = np.asarray(station_ids, dtype=np.int64)
            # Cold-start guard: clamp to known range.
            sid = np.clip(sid, 0, self.n_stations_ - 1)
        else:
            sid = np.zeros(N, dtype=np.int64)
        self.model_.eval()
        with torch.no_grad():
            pred_resid = self.model_(
                torch.from_numpy(Xn.astype(np.float32)),
                torch.from_numpy(sid),
            ).numpy()
        mu = (baseline + pred_resid).astype(float)
        prov = ProvenanceMeta(
            forecaster_name=self.name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), datetime.now()),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=self.seed,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=datetime.now(),
        )
        return PointForecast(
            mu=mu, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


class _HourlyGRU:
    """Inner torch module (built lazily via __new__ trick to avoid eager
    torch import at module import time). Mirrors weather/rnn_hourly.HourlyGRU.
    """

    def __new__(cls, n_channels: int, n_stations: int, hidden: int, embed: int, dropout: float):
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch
        from torch import nn

        class HourlyGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.station_embed = nn.Embedding(n_stations, embed)
                self.gru = nn.GRU(input_size=n_channels, hidden_size=hidden, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Sequential(
                    nn.Linear(hidden + embed, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x, sid_idx):
                _, h_n = self.gru(x)
                h = h_n[-1]
                emb = self.station_embed(sid_idx)
                z = self.dropout(torch.cat([h, emb], dim=-1))
                return self.head(z).squeeze(-1)

        return HourlyGRU()
