"""Native parametric-distribution forecasters.

EMOS, NGBoostNormal, MixtureNormals (parametric normal / mixture);
Stacking (parametric-normal meta-learner with depends_on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as _scipy_norm

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    DistributionForecast,
    ProvenanceMeta,
)
from bracketlearn.trainers._common import (
    _weighted_lstsq2,
)

# ---------------------------------------------------------------------------
# EMOS — Ensemble Model Output Statistics. Native parametric-normal DistForecaster.
# ---------------------------------------------------------------------------


@dataclass
class EMOS(BaseEstimator):
    """EMOS / NGR distributional regression for an ensemble forecast.

    Two fit algorithms are supported, selected by ``fit_method``:

    ``fit_method="ols"`` (default — bracketlearn's v0.1 method):
        μ̂(x) = a + b·ens_mean
        σ̂²(x) = c + d·ens_var      (linear-in-variance)
        Closed-form: OLS for (a, b); OLS on squared residuals for
        (c, d). Falls back to constant σ̂² (mean r²) if the linear
        variance fit emits non-positive variance anywhere in the
        training range. Fast (single lstsq call per side), no
        optimiser.

    ``fit_method="crps_nelder_mead"`` (matches the parent repo's
        ``prediction_market_weather/ml/trainers/emos.py`` snowflake
        exactly — Gneiting & Raftery 2005, Gneiting et al. 2005):
        μ̂(x) = a + b·ens_mean
        σ̂²(x) = exp(c) + exp(d)·ens_std²   (exp-link variance)
        Coefficients (a, b, c, d) minimise mean closed-form Gaussian
        CRPS via Nelder-Mead, initialised from OLS for (a, b) and
        half-split residual-variance for (c, d). Slower (a few seconds
        on 1k rows) but tightly fits the CRPS surface end-to-end.

    Two input forms via ``input_form``:

    ``input_form="members"`` (default):
        X holds the per-row ensemble *members* (one column per member).
        ens_mean/ens_var/ens_std are computed via ``X.mean(axis=1)`` /
        ``X.var(axis=1, ddof=0)`` / ``np.sqrt(var)``.

    ``input_form="aggregates"``:
        X already holds the pre-computed aggregates as two columns:
        ``X[:, 0] = ens_mean`` and ``X[:, 1] = ens_std``. Useful when
        the upstream pipeline already builds these (e.g. parent-repo
        weather feature matrix has ``src_<SIDE>_mean`` and
        ``src_<SIDE>_std`` columns).

    For ``fit_method="crps_nelder_mead"`` the closed-form Gaussian CRPS::

        CRPS(N(μ, σ²), y) = σ · [ z·(2·Φ(z) − 1) + 2·φ(z) − 1/√π ]
        where z = (y − μ) / σ

    is minimised over ``(a, b, c, d)``. Sample weights are not yet
    threaded through this fit method (raises if passed).
    """

    name: str = "EMOS"
    depends_on: tuple[str, ...] = ()
    fit_method: Literal["ols", "crps_nelder_mead"] = "ols"
    input_form: Literal["members", "aggregates"] = "members"
    a_: float | None = field(default=None, init=False)
    b_: float | None = field(default=None, init=False)
    c_: float | None = field(default=None, init=False)
    d_: float | None = field(default=None, init=False)

    # ---------- input adapter ----------

    def _row_aggregates(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ens_mean, ens_var, ens_std) per row from X under the
        configured input_form."""
        X = np.asarray(X, dtype=float)
        if self.input_form == "members":
            if X.ndim != 2:
                raise ValueError(
                    f"EMOS(input_form='members'): X must be 2-D; got shape {X.shape}"
                )
            ens_mean = X.mean(axis=1)
            ens_var = X.var(axis=1, ddof=0)
            ens_std = np.sqrt(ens_var)
        elif self.input_form == "aggregates":
            if X.ndim != 2 or X.shape[1] != 2:
                raise ValueError(
                    f"EMOS(input_form='aggregates'): X must be (N, 2) with "
                    f"X[:, 0]=ens_mean, X[:, 1]=ens_std; got shape {X.shape}"
                )
            ens_mean = X[:, 0]
            ens_std = X[:, 1]
            if np.any(ens_std <= 0):
                raise ValueError(
                    "EMOS(input_form='aggregates'): ens_std (X[:, 1]) must be "
                    "strictly positive."
                )
            ens_var = ens_std ** 2
        else:
            raise ValueError(
                f"EMOS.input_form must be 'members' or 'aggregates'; "
                f"got {self.input_form!r}"
            )
        return ens_mean, ens_var, ens_std

    # ---------- fit ----------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        y = np.asarray(y, dtype=float)
        ens_mean, ens_var, ens_std = self._row_aggregates(X)

        if self.fit_method == "ols":
            self._fit_ols(ens_mean, ens_var, y, sample_weight)
        elif self.fit_method == "crps_nelder_mead":
            if sample_weight is not None:
                raise NotImplementedError(
                    "EMOS(fit_method='crps_nelder_mead'): sample_weight is "
                    "not threaded through the Nelder-Mead fit. Use "
                    "fit_method='ols' if you need weighted fitting."
                )
            self._fit_crps_nelder_mead(ens_mean, ens_std, y)
        else:
            raise ValueError(
                f"EMOS.fit_method must be 'ols' or 'crps_nelder_mead'; "
                f"got {self.fit_method!r}"
            )
        return self

    def _fit_ols(
        self,
        ens_mean: np.ndarray,
        ens_var: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
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

    def _fit_crps_nelder_mead(
        self,
        ens_mean: np.ndarray,
        ens_std: np.ndarray,
        y: np.ndarray,
    ) -> None:
        # OLS init for (a, b).
        A_mu = np.column_stack([np.ones_like(ens_mean), ens_mean])
        beta, *_ = np.linalg.lstsq(A_mu, y, rcond=None)
        a0, b0 = float(beta[0]), float(beta[1])
        resid_var = float(np.var(y - (a0 + b0 * ens_mean)))
        # Split residual variance half-half between the constant term and
        # the spread coefficient (scaled by mean ens_std²).
        mean_spread_sq = float(np.mean(ens_std ** 2))
        c0 = math.log(max(resid_var / 2.0, 1e-6))
        d0 = math.log(max(resid_var / (2.0 * max(mean_spread_sq, 1e-6)), 1e-6))
        x0 = np.array([a0, b0, c0, d0], dtype=float)
        res = minimize(
            _crps_nelder_mead_loss, x0,
            args=(ens_mean, ens_std, y),
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 5000},
        )
        self.a_, self.b_, self.c_, self.d_ = (float(v) for v in res.x)

    # ---------- predict ----------

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        if self.a_ is None:
            raise RuntimeError("EMOS.predict_dist called before fit")
        ens_mean, ens_var, ens_std = self._row_aggregates(X)
        mu = self.a_ + self.b_ * ens_mean
        if self.fit_method == "crps_nelder_mead":
            # Exp-link variance — always strictly positive.
            var = math.exp(self.c_) + math.exp(self.d_) * (ens_std ** 2)
        else:
            # Linear-in-variance — guarded at fit, recheck on inference.
            var = self.c_ + self.d_ * ens_var
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
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_normal(
            mu, sigma, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# Standalone CRPS objective for EMOS(fit_method='crps_nelder_mead').
# ---------------------------------------------------------------------------


def _gaussian_crps_closed_form(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Closed-form Gaussian CRPS, vectorised. See EMOS docstring."""
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return sigma * (
        z * (2.0 * _scipy_norm.cdf(z) - 1.0)
        + 2.0 * _scipy_norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )


def _crps_nelder_mead_loss(
    params: np.ndarray,
    ens_mean: np.ndarray,
    ens_std: np.ndarray,
    y: np.ndarray,
) -> float:
    a, b, c, d = params
    mu = a + b * ens_mean
    var = math.exp(c) + math.exp(d) * (ens_std ** 2)
    sigma = np.sqrt(var)
    return float(np.mean(_gaussian_crps_closed_form(mu, sigma, y)))


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
        if (
            ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(ids), upstream_ids)
        ):
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
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
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

    For reproducible fits, set BOTH seeds:

    * ``random_seed`` → seeds NGBoost's minibatching / column-subsampling.
    * ``base_random_state`` → seeds the per-iteration cloned base learner
      (default ``DecisionTreeRegressor``). NGBoost's default Base has
      ``random_state=None`` so tree split tie-breaking draws from the OS
      RNG; without this, successive fits with the same ``random_seed``
      still produce different μ̂/σ̂.

    When ``base_random_state`` is set, this constructs a
    ``DecisionTreeRegressor`` matching ``ngboost.learners.
    default_tree_learner`` exactly except for the pinned seed and
    passes it as ``Base=``. When ``base_random_state`` is None,
    NGBoost's default unseeded Base is used (non-reproducible fits).
    """

    n_estimators: int = 400
    learning_rate: float = 0.01
    minibatch_frac: float = 0.5
    natural_gradient: bool = True
    sigma_floor: float = 0.5
    random_seed: int | None = None
    base_random_state: int | None = None
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
        ngb_kwargs: dict[str, Any] = dict(
            Dist=Normal,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            natural_gradient=self.natural_gradient,
            random_state=self.random_seed,
            verbose=False,
        )
        if self.base_random_state is not None:
            from sklearn.tree import DecisionTreeRegressor
            ngb_kwargs["Base"] = DecisionTreeRegressor(
                criterion="friedman_mse",
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_depth=3,
                splitter="best",
                random_state=self.base_random_state,
            )
        self.model_ = NGBRegressor(**ngb_kwargs)
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
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native", random_seed=self.random_seed)
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
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_mixture_normal(
            weights=weights, mus=mus, sigmas=sigmas,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


