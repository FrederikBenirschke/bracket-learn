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

These trainers exist to prove the framework holds across protocol shapes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    DistributionForecast,
    PointForecast,
    ProvenanceMeta,
    bracket_probs_from_cdf_at_edges,
)


def _estimator_accepts_sample_weight(estimator: Any) -> bool:
    """Inspect fit signature for a sample_weight parameter.

    Replaces the v0.1 bare ``except TypeError`` pattern (any TypeError
    raised *inside* fit silently dropped the weights). Now we either pass
    weights or skip them by explicit signature check.
    """
    try:
        sig = inspect.signature(estimator.fit)
    except (ValueError, TypeError):
        return False
    return "sample_weight" in sig.parameters


def _weighted_lstsq2(A: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> tuple[float, float]:
    """Weighted least squares for 2-column design matrices. Returns (a, b)."""
    if w is None:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    else:
        sw = np.sqrt(np.asarray(w, dtype=float))
        sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return float(sol[0]), float(sol[1])


# ---------------------------------------------------------------------------
# SklearnPoint — wrap any sklearn-style regressor as a PointForecaster.
# ---------------------------------------------------------------------------


@dataclass
class SklearnPoint(BaseEstimator):
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
        # Record input signature BEFORE np.asarray strips the columns
        # attribute (sklearn convention: feature_names_in_ from DataFrame).
        self._record_input_signature(X)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Forward sample_weight only if the estimator accepts it. We
        # introspect the signature (no silent TypeError swallow).
        if sample_weight is not None and _estimator_accepts_sample_weight(self.estimator):
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y)
        self.fitted_ = True
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
class EMOS(BaseEstimator):
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

        # OLS for μ: y ≈ a + b·ens_mean (weighted if sample_weight given).
        A_mu = np.column_stack([np.ones_like(ens_mean), ens_mean])
        self.a_, self.b_ = _weighted_lstsq2(A_mu, y, sample_weight)

        # Squared residuals → σ². Method-of-moments OLS for variance:
        # r² ≈ c + d·ens_var. Unconstrained OLS can return c_<0 or d_<0,
        # which makes σ²(x) negative somewhere in the training range —
        # silently clipping that at predict time hides a bad fit (Rule
        # #0.5). Solve unconstrained first; if either coefficient is
        # negative, fall back to a constant variance (mean of r²) and
        # record that we did so via ``sigma_source``.
        resid = y - (self.a_ + self.b_ * ens_mean)
        r2 = resid ** 2
        A_var = np.column_stack([np.ones_like(ens_var), ens_var])
        c_unc, d_unc = _weighted_lstsq2(A_var, r2, sample_weight)
        # Reject the linear-in-variance fit if it would emit negative
        # variance anywhere on the *training* spread range.
        var_train = c_unc + d_unc * ens_var
        if c_unc < 0 or d_unc < 0 or np.any(var_train <= 0):
            if sample_weight is None:
                c_fallback = float(np.mean(r2))
            else:
                w = np.asarray(sample_weight, dtype=float)
                c_fallback = float((w * r2).sum() / w.sum())
            if c_fallback <= 0:
                raise ValueError(
                    "EMOS: mean squared residual non-positive — y is a "
                    "perfect linear function of X.mean(axis=1) on the "
                    "training set; no variance left to fit."
                )
            self.c_, self.d_ = c_fallback, 0.0
            self.sigma_fit_was_constant_ = True
        else:
            self.c_, self.d_ = c_unc, d_unc
            self.sigma_fit_was_constant_ = False
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
        var = self.c_ + self.d_ * ens_var
        # var should be > 0 by construction (fit guards both coefficients
        # and rechecks on training data). Negative here means the
        # inference X.var() went outside the training range — a real
        # extrapolation problem, not a numerical-noise floor. Raise.
        if np.any(var <= 0):
            n_bad = int(np.sum(var <= 0))
            min_var = float(var.min())
            raise ValueError(
                f"EMOS.predict_dist: linear-in-variance fit emits "
                f"non-positive variance on {n_bad} rows "
                f"(min var = {min_var:.3g}). The inference X has lower "
                f"ensemble spread than any training row; refit on a "
                f"wider spread range or use a constant-σ fallback."
            )
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
class Stacking(BaseEstimator):
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
        ids: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"Stacking.fit needs deps_oof for {self.depends_on}; got {list(deps_oof or [])}"
            )
        y = np.asarray(y, dtype=float)
        # Stack upstream μ predictions row-aligned. We REQUIRE that each
        # upstream dist's .ids matches our (X, y) row order (no silent
        # misalignment). If the caller passes ids, we check
        # them; if not, we still require all upstream dists to agree on
        # their own ids vectors (else the meta-learner builds rows from
        # mis-zipped predictions).
        upstream_ids = None
        for name in self.depends_on:
            d = deps_oof[name]
            if d.backing.value != "parametric":
                raise NotImplementedError(
                    f"Stacking expects parametric upstream; {name} is {d.backing}"
                )
            if d.params["mu"].shape[0] != y.shape[0]:
                raise ValueError(
                    f"Stacking.fit: deps_oof[{name!r}] has N={d.params['mu'].shape[0]} "
                    f"but y has N={y.shape[0]}"
                )
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"Stacking.fit: deps_oof[{name!r}].ids does not match the "
                    f"first upstream's ids — meta-learner rows would be misaligned"
                )
        if ids is not None and upstream_ids is not None:
            if not np.array_equal(np.asarray(ids), upstream_ids):
                raise ValueError(
                    "Stacking.fit: caller's ids do not match deps_oof ids — "
                    "rows would be misaligned"
                )
        cols = [deps_oof[name].params["mu"] for name in self.depends_on]
        Z = np.column_stack(cols)  # (N, K)
        # OLS with intercept (weighted if sample_weight given).
        A = np.column_stack([np.ones(Z.shape[0]), Z])
        if sample_weight is None:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            sw = np.sqrt(np.asarray(sample_weight, dtype=float))
            sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        self.intercept_ = float(sol[0])
        self.weights_ = sol[1:]
        resid = y - (self.intercept_ + Z @ self.weights_)
        self.sigma_ = float(np.std(resid, ddof=1))
        # Refuse degenerate σ̂. v0.1 floored σ̂≤0 to 1e-3,
        # which produced near-deterministic forecasts and masked
        # upstream-μ-vs-y collinearity (data leak). We raise when σ̂
        # falls below a small fraction of y's scale — covers exact-zero
        # AND float-noise-positive cases.
        y_scale = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
        if self.sigma_ <= max(1e-9 * max(y_scale, 1.0), 1e-12):
            raise ValueError(
                f"Stacking.fit: residual std is degenerate "
                f"(sigma_={self.sigma_:.3g}, y_scale={y_scale:.3g}); "
                f"meta-learner perfectly fits training y, which means either "
                f"upstream μ collinearity with y (data leak) or N is too small. "
                f"Refusing to substitute a 1e-3 floor."
            )
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
        # Row-alignment check: each upstream's ids must match the caller's ids
        # exactly (no silent misalignment).
        ids_arr = np.asarray(ids)
        for name in self.depends_on:
            d = deps_oof[name]
            if not np.array_equal(d.ids, ids_arr):
                raise ValueError(
                    f"Stacking.predict_dist: deps_oof[{name!r}].ids does not "
                    f"match caller ids — rows would be misaligned"
                )
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
class NGBoostNormal(BaseEstimator):
    """Native parametric-normal DistForecaster backed by NGBoost.

    Boosts (μ̂, σ̂) as non-linear functions of the full feature vector so
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
class MixtureNormals(BaseEstimator):
    """Per-vendor Gaussian mixture.

        p(y | x) = (1/K) Σ_v N(y; x_v, σ_v²)

    where x_v is the v-th column of X (vendor's point forecast) and σ_v is
    that vendor's train-slice RMSE against y. Equal weights over the K
    columns of X.

    Treats X as already-curated vendor columns; missing-value handling per
    row is out of scope for the (N, K) dense array contract.

    Rows with all-NaN columns or all-zero variances raise.
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
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float)
            sigma_v = np.sqrt((w[:, None] * diffs ** 2).sum(axis=0) / w.sum())
        else:
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
        from bracketlearn.tail import TailPolicy, TailRule

        if not self.models_:
            raise RuntimeError("QuantileReg.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        qvals = np.column_stack([self.models_[t].predict(X) for t in self.taus])
        # Repair crossings across rows in one vectorised pass.
        qvals = np.maximum.accumulate(qvals, axis=1)
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
        from bracketlearn.tail import TailPolicy, TailRule

        if self.model_ is None:
            raise RuntimeError("QuantileForest.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        qpred = np.asarray(self.model_.predict(X, quantiles=list(self.taus)),
                           dtype=float)
        # quantile-forest can return shape (N, Q) or (N,) depending on Q==1; coerce.
        if qpred.ndim == 1:
            qpred = qpred.reshape(-1, 1)
        qpred = np.maximum.accumulate(qpred, axis=1)
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
class CumulativeBinary(BaseEstimator):
    """Single LightGBM binary classifier on augmented features.

    Fits one classifier over (X, cutpoint) → 1[y ≤ cutpoint] using a fixed grid
    of cutpoints (derived from the bracket ladder); at predict time, queries
    P(y ≤ k) for each k in the grid and emits a quantile-backed dist where
    taus = the cdf values at the configured cutpoints.

    Constructed with a fixed `cutpoints` array — typically the interior
    edges of the eval bracket ladder. Caller MUST also pass
    ``outer_edges=(low, high)`` defining the bracket boundaries below
    ``cutpoints[0]`` and above ``cutpoints[-1]``. The two outer bins
    absorb the tail mass under TailRule.clip semantics; without explicit
    outer edges, downstream mean/variance would be biased by an invented
    pad.
    """

    cutpoints: np.ndarray
    outer_edges: tuple[float, float]
    n_estimators: int = 80
    learning_rate: float = 0.05
    num_leaves: int = 7
    min_child_samples: int = 100
    monotone: bool = True
    name: str = "CumulativeBinary"
    depends_on: tuple[str, ...] = ()
    model_: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        lo, hi = self.outer_edges
        cuts = np.asarray(self.cutpoints, dtype=float)
        if cuts.size == 0:
            return  # fit() will raise on this; defer.
        if not (lo < cuts[0]):
            raise ValueError(
                f"outer_edges[0]={lo} must be strictly less than cutpoints[0]={cuts[0]}"
            )
        if not (hi > cuts[-1]):
            raise ValueError(
                f"outer_edges[1]={hi} must be strictly greater than cutpoints[-1]={cuts[-1]}"
            )

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
        if sample_weight is not None:
            sw_aug = np.repeat(sample_weight, cuts.size)
            self.model_.fit(X_aug, y_aug, sample_weight=sw_aug)
        else:
            self.model_.fit(X_aug, y_aug)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:

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
        # Isotonic repair across rows in one pass (non-monotonicity is rare
        # with monotone=True but the LGBM constraint isn't strict for binary).
        proba = np.maximum.accumulate(proba, axis=1)
        # taus = proba values at the cutpoints (per row), but quantile-backed
        # storage assumes shared taus across rows. Reverse the roles: store
        # cutpoints as qvals, and use proba as per-row taus — but that breaks
        # the shared-tau invariant too. Instead, define a *fixed tau grid*
        # using rank of cutpoints (e.g. (1..K)/(K+1)) and let qvals = cutpoints.
        # Simpler: emit a bracket-backed dist on edges = [-inf-equivalent,
        # cuts, +inf-equivalent], probs = diff(0, proba_at_cuts, 1).
        #
        # Emit a bracket dist over [outer_edges[0], cuts..., outer_edges[1]].
        # The two outer bins absorb tail mass under TailRule.clip. The outer
        # edges are explicit constructor args (no invented pad).
        lo, hi = self.outer_edges
        edges = np.concatenate([[lo], cuts, [hi]])
        # CDF at the inner edges = classifier proba; at the outer edges = 0 and 1.
        cdf_at_edges = np.column_stack([
            np.zeros(N),
            proba,
            np.ones(N),
        ])
        probs = bracket_probs_from_cdf_at_edges(
            cdf_at_edges, source="CumulativeBinary.predict_dist",
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
class TailSpecialist(BaseEstimator):
    """Gaussian body (from upstream EMOS μ̂/σ̂) + LightGBM tail classifiers.

    depends_on a parametric-normal upstream (typically named 'emos') and a ladder
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
            verbose=-1,
        )
        # class_weight="balanced" only when the caller does NOT supply
        # sample_weight (don't silently multiply user weights
        # by sklearn's balanced inverse-frequency weights).
        if sample_weight is None:
            common["class_weight"] = "balanced"
        self.clf_lo_ = lgb.LGBMClassifier(**common)
        self.clf_hi_ = lgb.LGBMClassifier(**common)
        if sample_weight is not None:
            self.clf_lo_.fit(X, y_lo, sample_weight=sample_weight)
            self.clf_hi_.fit(X, y_hi, sample_weight=sample_weight)
        else:
            self.clf_lo_.fit(X, y_lo)
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
        # Sanity check: when the ladder is narrow, the upstream EMOS
        # may put substantial mass in body bins 0 and B-1 (the bins
        # the tail classifiers are about to replace). Discarding that
        # mass is the *point* of TailSpecialist — but if the classifier
        # disagrees with the upstream by more than `tail_disagreement_tol`,
        # the user is probably running on a ladder too narrow for this
        # trainer's design. Warn loudly so it's not silent.
        upstream_p_lo = body_probs[:, 0]
        upstream_p_hi = body_probs[:, -1]
        max_disagreement = float(np.maximum(
            np.abs(p_lo - upstream_p_lo), np.abs(p_hi - upstream_p_hi),
        ).max())
        if max_disagreement > 0.5:
            import warnings
            warnings.warn(
                f"TailSpecialist: classifier tail probabilities disagree "
                f"with upstream EMOS by up to {max_disagreement:.2f} on "
                f"the outer bins. The classifier outputs *replace* the "
                f"upstream's edge-bin mass — large disagreement on a "
                f"narrow ladder usually means the EMOS body is dominating "
                f"the tails. Consider widening the ladder.",
                UserWarning, stacklevel=2,
            )
        # Rescale inner bins (1..B-1, i.e. all but first and last) to
        # (1 - p_lo - p_hi). Refuse to silently fabricate a
        # uniform body when the upstream returns zero inner mass —
        # that means the upstream is degenerate and the caller needs
        # to know.
        inner = body_probs[:, 1:-1]
        inner_sum = inner.sum(axis=1, keepdims=True)
        if np.any(inner_sum <= 0):
            n_bad = int((inner_sum.ravel() <= 0).sum())
            raise ValueError(
                f"TailSpecialist.predict_dist: {n_bad}/{N} rows have zero "
                f"upstream body mass in bins [1..B-2]. Refusing to "
                f"redistribute uniformly."
            )
        body_total = np.maximum(0.0, 1.0 - p_lo - p_hi)[:, None]
        inner_scaled = inner * (body_total / inner_sum)
        probs = np.concatenate(
            [p_lo[:, None], inner_scaled, p_hi[:, None]], axis=1,
        )
        # Final renorm against numerical drift (clip+scale can leave
        # rounding errors at the 1e-15 level). Any row sum at or below
        # zero indicates a logic error, not drift — raise.
        row_sum = probs.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError(
                "TailSpecialist.predict_dist: row sum is non-positive after "
                "renormalisation — should be unreachable; investigate."
            )
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
class OnlineAggregator(BaseEstimator):
    """AdaHedge over forecast experts (columns of X).

    Walks rows in order, treats each column of X as an expert's point
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
        awake = ~np.isnan(X)                      # (N, K) bool
        # Weight matrix: final_w_ broadcast against awake mask.
        w_mat = self.final_w_[None, :] * awake    # (N, K) — zeroes on asleep
        x_mat = np.where(awake, X, 0.0)
        num = (w_mat * x_mat).sum(axis=1)         # (N,)
        denom = w_mat.sum(axis=1)                 # (N,)
        awake_counts = awake.sum(axis=1)          # (N,)
        ok = (awake_counts >= self.min_experts) & (denom > 0)
        mu = np.full(N, np.nan)
        mu[ok] = num[ok] / denom[ok]
        # Leftover NaNs are a real coverage hole — raise.
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
class RNNHourly(BaseEstimator):
    """Tiny GRU on a (24, C) hourly tensor → residual-corrected point forecast.

    GRU reads the 24-hour sequence, concatenates a station embedding (if station_ids
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
            # Raise on unknown station IDs instead of silently
            # clamping them onto station 0's embedding. Cold-start is a
            # real failure mode that needs caller-level handling (drop the
            # row, pick a fallback embedding policy explicitly, or extend
            # the training set), not a silent map-to-zero.
            unknown_mask = (sid < 0) | (sid >= self.n_stations_)
            if np.any(unknown_mask):
                bad = np.unique(sid[unknown_mask]).tolist()
                raise ValueError(
                    f"RNNHourly.predict: {int(unknown_mask.sum())} rows have "
                    f"station_ids outside the trained range "
                    f"[0, {self.n_stations_ - 1}]; unknown IDs={bad[:10]}"
                )
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


# ---------------------------------------------------------------------------
# DistAsFeatures — generic bridge: upstream dists → feature matrix → any trainer.
# ---------------------------------------------------------------------------


_DIST_FEATURE_TAUS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class DistAsFeatures(BaseEstimator):
    """Materialise K upstream distributions into a feature matrix and hand it
    to a downstream forecaster.

    Per-row features extracted from each upstream dist:

        - quantiles at ``feature_taus`` (default 5 quantiles)
        - mean (if ``include_mean=True``)
        - variance (if ``include_variance=True``)
        - CDF at ``tail_cutpoints`` (tail-mass features)

    Total per row: K · (|taus| + include_mean + include_variance + |cuts|).

    The downstream forecaster sees ONLY dist-derived features, not raw X.
    If you also want raw X, build a separate node — keeping this class
    single-purpose is intentional.

    The downstream's own ``depends_on`` is ignored; ``DistAsFeatures`` owns
    the dep contract via its ``deps`` argument.

    Requires each upstream backing to support ``ppf`` for the requested
    ``feature_taus``. v0.1 ppf coverage: parametric-normal, mixture-normal,
    quantile, bracket.
    """

    deps: tuple[str, ...]
    downstream: Any
    feature_taus: tuple[float, ...] = _DIST_FEATURE_TAUS
    tail_cutpoints: tuple[float, ...] = ()
    include_mean: bool = True
    include_variance: bool = True
    name: str = "DistAsFeatures"
    _n_features_: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.deps:
            raise ValueError("DistAsFeatures requires at least one upstream dep")
        self.depends_on = tuple(self.deps)

    def _featurize(self, deps_oof: dict[str, Any]) -> np.ndarray:
        taus = np.asarray(self.feature_taus, dtype=float)
        cuts = np.asarray(self.tail_cutpoints, dtype=float)
        cols: list[np.ndarray] = []
        for name in self.depends_on:
            d = deps_oof[name]
            cols.append(d.ppf(taus))                  # (N, len(taus))
            if self.include_mean:
                cols.append(d.mean()[:, None])
            if self.include_variance:
                cols.append(d.variance()[:, None])
            if cuts.size:
                cols.append(d.cdf(cuts))              # (N, len(cuts))
        return np.column_stack(cols)

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
                f"DistAsFeatures.fit needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        Z = self._featurize(deps_oof)
        # Forward sample_weight only if downstream accepts it; matches the
        # SklearnPoint convention.
        try:
            self.downstream.fit(Z, y, sample_weight=sample_weight, deps_oof=None)
        except TypeError:
            self.downstream.fit(Z, y)
        self._n_features_ = Z.shape[1]
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> PointForecast:
        Z = self._predict_features(deps_oof)
        return self.downstream.predict(Z, ids=ids, timestamps=timestamps)

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        Z = self._predict_features(deps_oof)
        return self.downstream.predict_dist(Z, ids=ids, timestamps=timestamps)

    def _predict_features(self, deps_oof: dict[str, Any] | None) -> np.ndarray:
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"DistAsFeatures.predict needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        Z = self._featurize(deps_oof)
        if Z.shape[1] != self._n_features_:
            raise RuntimeError(
                f"DistAsFeatures: train had {self._n_features_} features; "
                f"predict produced {Z.shape[1]}"
            )
        return Z


# ---------------------------------------------------------------------------
# LinearPoolDist — convex combination of upstream DistributionForecasts.
# ---------------------------------------------------------------------------


@dataclass
class LinearPoolDist(BaseEstimator):
    """Linear (mixture) opinion pool over K upstream dists:

        F(y | x) = Σ_k w_k · F_k(y | x),    w_k ≥ 0,  Σ w_k = 1

    Weights are GLOBAL (not per-row) and fit by minimising weighted-empirical
    CRPS on OOF. Per-component samples drawn from a fixed mid-rank τ grid
    via ppf — so each upstream backing must support ppf.

    Output backing: quantile, evaluated at a 99-point τ grid by inverting
    the weighted empirical CDF of stacked component samples. Tail policy:
    clip.

    For Gaussian-only upstream a closed-form mixture-CRPS exists (Grimit
    et al., 2006) — left as a v0.2 optimisation.
    """

    deps: tuple[str, ...]
    n_samples: int = 200
    name: str = "LinearPoolDist"
    weights_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if len(self.deps) < 2:
            raise ValueError(
                f"LinearPoolDist needs ≥2 upstream deps; got {self.deps}"
            )
        self.depends_on = tuple(self.deps)

    def _sample_grid(self) -> np.ndarray:
        # Mid-rank τ grid in (0, 1); excludes endpoints so parametric-normal
        # tails don't blow up to ±inf.
        return (np.arange(self.n_samples) + 0.5) / self.n_samples

    def _component_samples(self, deps_oof: dict[str, Any]) -> np.ndarray:
        """Return (K, N, n_samples) sample tensor from upstream ppfs."""
        taus = self._sample_grid()
        cols = [deps_oof[name].ppf(taus) for name in self.depends_on]
        return np.stack(cols, axis=0)

    @staticmethod
    def _weighted_crps(
        stacked: np.ndarray,                       # (N, M) — M = K·S
        sample_w: np.ndarray,                      # (M,) sums to 1
        y: np.ndarray,                             # (N,)
    ) -> np.ndarray:
        """Per-row weighted-empirical CRPS.

        CRPS = Σ_j w_j |x_j - y|  -  0.5 · Σ_{j,k} w_j w_k |x_j - x_k|

        Vectorised pairwise term via sorted-sample identity:
            0.5 · Σ_{j,k} w_j w_k |x_j - x_k|
              = Σ_j w_j (x_j · cum_w_j  -  cum_wx_j)
        where cum_w / cum_wx are cumulative sums over x-sorted samples.
        """
        N, M = stacked.shape
        term1 = (sample_w[None, :] * np.abs(stacked - y[:, None])).sum(axis=1)
        order = np.argsort(stacked, axis=1)
        s_sorted = np.take_along_axis(stacked, order, axis=1)
        w_sorted = sample_w[order]
        cum_w = np.cumsum(w_sorted, axis=1)
        cum_wx = np.cumsum(w_sorted * s_sorted, axis=1)
        pairwise = (w_sorted * (s_sorted * cum_w - cum_wx)).sum(axis=1)
        return term1 - pairwise

    def _objective(
        self,
        w: np.ndarray,
        comp_samples: np.ndarray,                  # (K, N, S)
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        K, N, S = comp_samples.shape
        stacked = comp_samples.transpose(1, 0, 2).reshape(N, K * S)
        sample_w = np.repeat(w, S) / S
        crps = self._weighted_crps(stacked, sample_w, y)
        if sample_weight is None:
            return float(crps.mean())
        sw = np.asarray(sample_weight, dtype=float)
        return float((sw * crps).sum() / sw.sum())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        from scipy.optimize import minimize

        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"LinearPoolDist.fit needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        y = np.asarray(y, dtype=float)
        comp_samples = self._component_samples(deps_oof)
        K = comp_samples.shape[0]
        w0 = np.full(K, 1.0 / K)
        bounds = [(0.0, 1.0)] * K
        constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        res = minimize(
            self._objective, w0,
            args=(comp_samples, y, sample_weight),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 200},
        )
        if not res.success:
            raise RuntimeError(
                f"LinearPoolDist: weight optimisation failed: {res.message}"
            )
        w_fit = np.clip(res.x, 0.0, 1.0)
        self.weights_ = w_fit / w_fit.sum()
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        from bracketlearn.tail import TailPolicy, TailRule

        if self.weights_ is None:
            raise RuntimeError("LinearPoolDist.predict_dist called before fit")
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"LinearPoolDist.predict_dist needs deps_oof for {self.depends_on}"
            )
        comp_samples = self._component_samples(deps_oof)        # (K, N, S)
        K, N, S = comp_samples.shape
        stacked = comp_samples.transpose(1, 0, 2).reshape(N, K * S)
        sample_w = np.repeat(self.weights_, S) / S

        taus_out = np.linspace(0.01, 0.99, 99)
        qvals = np.empty((N, taus_out.size))
        for i in range(N):
            order = np.argsort(stacked[i])
            s_sorted = stacked[i][order]
            w_sorted = sample_w[order]
            cum = np.cumsum(w_sorted)
            qvals[i] = np.interp(taus_out, cum, s_sorted)
        qvals = np.maximum.accumulate(qvals, axis=1)            # isotonic-repair

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
        return DistributionForecast.from_quantiles(
            taus=taus_out, qvals=qvals,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# CDFBoostBracket — B LightGBM heads on upstream-CDF features → bracket dist.
# ---------------------------------------------------------------------------


@dataclass
class CDFBoostBracket(BaseEstimator):
    """B LightGBM binary classifiers over upstream-CDF features.

    Construction:
        edges: (B+1,)  — bracket ladder. B = len(edges) - 1 bins.
        deps:  K upstream DistForecaster names.

    Feature matrix per row (passed to all B heads): the CDF of each upstream
    dist evaluated at every ladder edge → shape (K · (B+1),). Optionally
    concat raw X with ``include_raw_X=True`` (off by default — keeps the
    "dist features only" framing clean).

    Training: for each bin b, classifier_b predicts
        y_b = 1[edges[b] ≤ y < edges[b+1]]
    Outputs (N, B) probabilities, row-renormalised → bracket-backed dist.

    Why this rather than linear stacking on upstream µ:
      - sees the full CDF shape, not a point summary
      - tree splits can model conditional "trust schedules" across regimes
      - output is bracket-backed: natural fit for laddered contract pricing

    Compare with:
      - LinearPoolDist:    convex combination, global weights, full-dist mixture
      - DistAsFeatures + NGBoostNormal:  Gaussian output, dist-summary features
      - CumulativeBinary:  single classifier with cutpoint augmentation
    """

    deps: tuple[str, ...]
    edges: np.ndarray
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    include_raw_X: bool = False
    name: str = "CDFBoostBracket"
    clfs_: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.deps:
            raise ValueError("CDFBoostBracket requires at least one upstream dep")
        edges = np.asarray(self.edges, dtype=float)
        if edges.ndim != 1 or edges.size < 3:
            raise ValueError(
                f"edges must be 1-D with ≥3 entries (≥2 bins); got shape {edges.shape}"
            )
        if np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing")
        self.edges = edges
        self.depends_on = tuple(self.deps)

    def _featurize(
        self,
        X: np.ndarray | None,
        deps_oof: dict[str, Any],
    ) -> np.ndarray:
        cols = [deps_oof[name].cdf(self.edges) for name in self.depends_on]
        Z = np.column_stack(cols)
        if self.include_raw_X:
            if X is None:
                raise ValueError(
                    "CDFBoostBracket.include_raw_X=True but X was None"
                )
            X = np.asarray(X, dtype=float)
            Z = np.column_stack([X, Z])
        return Z

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"CDFBoostBracket.fit needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        y = np.asarray(y, dtype=float)
        Z = self._featurize(X, deps_oof)
        B = self.edges.size - 1
        # Bin assignment: index of bin each y falls into. y outside [edges[0],
        # edges[-1]] is clipped to the nearest bin (if you want
        # to forbid out-of-range y, do it at the caller).
        bin_idx = np.searchsorted(self.edges, y, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, B - 1)

        common = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            objective="binary",
            verbose=-1,
        )
        self.clfs_ = []
        for b in range(B):
            y_b = (bin_idx == b).astype(int)
            if y_b.sum() < 2:
                # Degenerate bin: no positives. Store sentinel = base rate.
                base_rate = float(y_b.mean())
                self.clfs_.append(("const", base_rate))
                continue
            clf = lgb.LGBMClassifier(**common, class_weight="balanced")
            if sample_weight is not None:
                clf.fit(Z, y_b, sample_weight=sample_weight)
            else:
                clf.fit(Z, y_b)
            self.clfs_.append(("model", clf))
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        if not self.clfs_:
            raise RuntimeError("CDFBoostBracket.predict_dist called before fit")
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"CDFBoostBracket.predict_dist needs deps_oof for {self.depends_on}"
            )
        Z = self._featurize(X, deps_oof)
        N = Z.shape[0]
        B = len(self.clfs_)
        probs = np.empty((N, B))
        for b, (kind, model) in enumerate(self.clfs_):
            if kind == "const":
                probs[:, b] = model
            else:
                probs[:, b] = model.predict_proba(Z)[:, 1]
        # Row-renorm (heads are independent, so sums won't be 1).
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise RuntimeError(
                "CDFBoostBracket: all-zero row in predict_proba "
                "(no head fired); check upstream dist coverage"
            )
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
            edges=self.edges, probs=probs,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )
