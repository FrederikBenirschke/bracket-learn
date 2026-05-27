"""Native parametric-distribution forecasters.

EMOS, NGBoostNormal, MixtureNormals (parametric normal / mixture);
StackedParametric (parametric meta-learner with depends_on; legacy
name ``Stacking`` is preserved as a module-level alias for back-compat).
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

# Euler-Mascheroni constant. Used by StackedParametric(sigma_method=
# 'geometric_mean_upstream') to debias E[log Z²] under Gaussian residuals:
# for Z ~ N(0, 1), E[log Z²] = −γ_E − log 2.
_EULER_GAMMA = 0.5772156649015329

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
# StackedParametric — DistForecaster with depends_on. Parametric meta-learner
# over upstream μ (and optionally σ). Legacy name ``Stacking`` aliased below.
# ---------------------------------------------------------------------------


@dataclass
class StackedParametric(BaseEstimator):
    """Meta-learner over upstream forecasters' parametric outputs.

    Defaults reproduce v0.1 ``Stacking`` behaviour exactly: OLS over
    upstream μ with intercept (unconstrained), constant σ̂ from residual
    std, Gaussian output. The optional knobs below widen the surface;
    nothing changes for existing ``StackedParametric(deps=...)`` (or
    legacy-alias ``Stacking(deps=...)``) callers.

    ``weight_constraint``:
        * ``"unconstrained"`` (default) — OLS with intercept; μ-weights
          take any sign and any magnitude.
        * ``"convex"`` — Σ wₖ = 1, wₖ ≥ 0 via SLSQP (classic Breiman
          1996 stacking). Intercept stays free so it can absorb any
          common bias in the upstream μ scale.

    ``sigma_method``:
        * ``"constant"`` (default) — σ̂ = std(in-sample residuals);
          single scalar applied to every row.
        * ``"geometric_mean_upstream"`` — per-row dispersion modelled as
          σ̂(x) = exp(α + Σ wⱼ · log σⱼ(x)). Fit by OLS regressing the
          bias-corrected target ``0.5·(log(resid² + ε) + γ_E + log 2)``
          on per-upstream log σⱼ(x), where the additive constant
          ``γ_E + log 2`` debiases E[log Z²] for Gaussian residuals.
          Requires every upstream to expose a positive σ in
          ``params['sigma']`` (i.e. parametric Normal / Student-t).
          ε is a small floor on resid² so resid=0 rows do not produce
          −∞ targets.

    ``dist_family``:
        * ``"normal"`` (default) — N(μ̂, σ̂²).
        * ``"student_t"`` — t_ν(μ̂, scale) with ν = ``student_t_df``.
          The fitted σ̂ is interpreted as the standard deviation of
          residuals (matches the residual-fit semantics); it is
          converted to the t-distribution *scale* parameter via
          ``scale = σ̂ · sqrt((ν − 2) / ν)`` so the forecast variance
          equals σ̂² regardless of ν.

    Pipeline injects ``deps_oof: dict[name → DistributionForecast]`` at
    fit time; this StackedParametric reads ``.params['mu']`` (and ``['sigma']``
    when ``sigma_method='geometric_mean_upstream'``) from each.
    """

    deps: tuple[str, ...]
    name: str = "StackedParametric"
    weight_constraint: Literal["unconstrained", "convex"] = "unconstrained"
    sigma_method: Literal["constant", "geometric_mean_upstream"] = "constant"
    dist_family: Literal["normal", "student_t"] = "normal"
    student_t_df: float = 5.0
    weights_: np.ndarray | None = field(default=None, init=False)
    intercept_: float | None = field(default=None, init=False)
    sigma_: float | None = field(default=None, init=False)
    sigma_alpha_: float | None = field(default=None, init=False)
    sigma_log_weights_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.depends_on = tuple(self.deps)
        if self.weight_constraint not in ("unconstrained", "convex"):
            raise ValueError(
                f"StackedParametric.weight_constraint must be 'unconstrained' or "
                f"'convex'; got {self.weight_constraint!r}"
            )
        if self.sigma_method not in ("constant", "geometric_mean_upstream"):
            raise ValueError(
                f"StackedParametric.sigma_method must be 'constant' or "
                f"'geometric_mean_upstream'; got {self.sigma_method!r}"
            )
        if self.dist_family not in ("normal", "student_t"):
            raise ValueError(
                f"StackedParametric.dist_family must be 'normal' or 'student_t'; "
                f"got {self.dist_family!r}"
            )
        if self.dist_family == "student_t" and self.student_t_df <= 2.0:
            raise ValueError(
                f"StackedParametric(dist_family='student_t'): student_t_df must be > 2 "
                f"for finite variance; got {self.student_t_df}"
            )

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
                f"StackedParametric.fit needs deps_oof for {self.depends_on}; got {list(deps_oof or [])}"
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
                    f"StackedParametric expects parametric upstream; {name} is {d.backing}"
                )
            if d.params["mu"].shape[0] != y.shape[0]:
                raise ValueError(
                    f"StackedParametric.fit: deps_oof[{name!r}] has N={d.params['mu'].shape[0]} "
                    f"but y has N={y.shape[0]}"
                )
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"StackedParametric.fit: deps_oof[{name!r}].ids does not match the "
                    f"first upstream's ids — meta-learner rows would be misaligned"
                )
        if (
            ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(ids), upstream_ids)
        ):
            raise ValueError(
                "StackedParametric.fit: caller's ids do not match deps_oof ids — "
                "rows would be misaligned"
            )
        cols = [deps_oof[name].params["mu"] for name in self.depends_on]
        Z = np.column_stack(cols)  # (N, K)
        if self.weight_constraint == "unconstrained":
            self._fit_mu_unconstrained(Z, y, sample_weight)
        else:
            self._fit_mu_convex(Z, y, sample_weight)
        resid = y - (self.intercept_ + Z @ self.weights_)
        # Refuse degenerate residuals before either σ branch. v0.1 floored
        # σ̂≤0 to 1e-3, which produced near-deterministic forecasts and
        # masked upstream-μ-vs-y collinearity (data leak). The same
        # collinearity poisons the geometric σ fit (target → −∞), so we
        # guard once on the residual std and let both σ branches proceed.
        y_scale = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
        resid_std = float(np.std(resid, ddof=1))
        if resid_std <= max(1e-9 * max(y_scale, 1.0), 1e-12):
            raise ValueError(
                f"StackedParametric.fit: residual std is degenerate "
                f"(resid_std={resid_std:.3g}, y_scale={y_scale:.3g}); "
                f"meta-learner perfectly fits training y, which means either "
                f"upstream μ collinearity with y (data leak) or N is too small. "
                f"Refusing to substitute a 1e-3 floor."
            )
        if self.sigma_method == "constant":
            self.sigma_ = resid_std
        else:
            self._fit_sigma_geometric_upstream(
                resid, y_scale, sample_weight, deps_oof,
            )
        return self

    def _fit_mu_unconstrained(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
        # OLS with intercept (weighted if sample_weight given).
        A = np.column_stack([np.ones(Z.shape[0]), Z])
        if sample_weight is None:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            sw = np.sqrt(np.asarray(sample_weight, dtype=float))
            sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        self.intercept_ = float(sol[0])
        self.weights_ = sol[1:]

    def _fit_mu_convex(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
        K = Z.shape[1]
        if sample_weight is None:
            def loss(params: np.ndarray) -> float:
                resid = y - params[0] - Z @ params[1:]
                return float(np.mean(resid ** 2))
        else:
            sw = np.asarray(sample_weight, dtype=float)
            sw_sum = float(sw.sum())

            def loss(params: np.ndarray) -> float:
                resid = y - params[0] - Z @ params[1:]
                return float((sw * resid ** 2).sum() / sw_sum)

        x0 = np.concatenate(
            [[float(np.mean(y) - np.mean(Z))], np.full(K, 1.0 / K)],
        )
        bounds = [(None, None)] + [(0.0, 1.0)] * K
        constraints = {
            "type": "eq",
            "fun": lambda p: float(np.sum(p[1:]) - 1.0),
        }
        res = minimize(
            loss, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if not res.success:
            raise RuntimeError(
                f"StackedParametric(weight_constraint='convex'): SLSQP failed: "
                f"{res.message}"
            )
        self.intercept_ = float(res.x[0])
        self.weights_ = np.asarray(res.x[1:], dtype=float)

    def _fit_sigma_geometric_upstream(
        self,
        resid: np.ndarray,
        y_scale: float,
        sample_weight: np.ndarray | None,
        deps_oof: dict[str, Any],
    ) -> None:
        cols_sigma = self._collect_upstream_sigma(deps_oof, where="fit")
        Z_log = np.column_stack([np.log(c) for c in cols_sigma])
        # Floor on resid² protects exact-zero rows from −∞ targets while
        # staying well below typical residual variance.
        eps = (1e-6 * max(y_scale, 1.0)) ** 2
        target = 0.5 * (np.log(resid ** 2 + eps) + _EULER_GAMMA + math.log(2.0))
        A = np.column_stack([np.ones(Z_log.shape[0]), Z_log])
        if sample_weight is None:
            sol, *_ = np.linalg.lstsq(A, target, rcond=None)
        else:
            sw = np.sqrt(np.asarray(sample_weight, dtype=float))
            sol, *_ = np.linalg.lstsq(A * sw[:, None], target * sw, rcond=None)
        self.sigma_alpha_ = float(sol[0])
        self.sigma_log_weights_ = np.asarray(sol[1:], dtype=float)

    def _collect_upstream_sigma(
        self,
        deps_oof: dict[str, Any],
        *,
        where: str,
    ) -> list[np.ndarray]:
        cols: list[np.ndarray] = []
        for name in self.depends_on:
            d = deps_oof[name]
            if "sigma" not in d.params:
                raise ValueError(
                    f"StackedParametric({where}, sigma_method='geometric_mean_upstream'): "
                    f"upstream {name!r} has no σ in params "
                    f"(backing={d.backing!r}); either pick "
                    f"sigma_method='constant' or feed parametric upstreams"
                )
            s = np.asarray(d.params["sigma"], dtype=float)
            if np.any(s <= 0):
                n_bad = int(np.sum(s <= 0))
                raise ValueError(
                    f"StackedParametric({where}): upstream {name!r} has {n_bad} "
                    f"non-positive σ values; cannot take log for "
                    f"geometric_mean_upstream"
                )
            cols.append(s)
        return cols

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
            raise ValueError("StackedParametric.predict_dist needs deps_oof")
        # Row-alignment check: each upstream's ids must match the caller's ids
        # exactly (no silent misalignment).
        ids_arr = np.asarray(ids)
        for name in self.depends_on:
            d = deps_oof[name]
            if not np.array_equal(d.ids, ids_arr):
                raise ValueError(
                    f"StackedParametric.predict_dist: deps_oof[{name!r}].ids does not "
                    f"match caller ids — rows would be misaligned"
                )
        cols = [deps_oof[name].params["mu"] for name in self.depends_on]
        Z = np.column_stack(cols)
        mu = self.intercept_ + Z @ self.weights_
        if self.sigma_method == "constant":
            sigma_std = np.full_like(mu, self.sigma_)
        else:
            cols_sigma = self._collect_upstream_sigma(deps_oof, where="predict_dist")
            Z_log = np.column_stack([np.log(c) for c in cols_sigma])
            sigma_std = np.exp(self.sigma_alpha_ + Z_log @ self.sigma_log_weights_)
        ids_arr = np.asarray(ids)
        ts_arr = np.asarray(timestamps)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        if self.dist_family == "normal":
            return DistributionForecast.from_normal(
                mu, sigma_std, ids=ids_arr, timestamps=ts_arr, provenance=prov,
            )
        # student_t: σ̂ is residual std; convert to t-scale so variance == σ̂².
        nu = self.student_t_df
        scale = sigma_std * math.sqrt((nu - 2.0) / nu)
        df_arr = np.full_like(mu, nu)
        return DistributionForecast.from_student_t(
            mu, scale, df_arr, ids=ids_arr, timestamps=ts_arr, provenance=prov,
        )


# Legacy alias. Pre-rename callers used ``Stacking``; keep the name
# resolvable so external scripts / notebooks don't break. Internal
# bracketlearn code should use ``StackedParametric`` going forward.
Stacking = StackedParametric


# ---------------------------------------------------------------------------
# BMAStacking — Bayesian model averaging meta-learner. Mixture-of-Normals output.
# ---------------------------------------------------------------------------


@dataclass
class BMAStacking(BaseEstimator):
    """Bayesian model averaging meta-learner. DistForecaster with ``depends_on``.

    Replaces ``Stacking``'s OLS-of-μ with a posterior over the mixture
    weight vector ``w`` on the K-simplex, and emits a true
    ``MixtureNormalForecast`` instead of a Normal collapsed onto a
    constant residual σ̂.

    Model::

        y_i | w ~ Σ_k w_k · N(y_i; μ_{k,i}, σ_{k,i})
        w        ~ Dir(α_0, …, α_0)                  (symmetric concentration)

    where each upstream forecaster k contributes (μ_{k,i}, σ_{k,i}) per
    row i from its OOF ``DistributionForecast``. For non-Normal
    parametric upstreams (Student-t, MixtureNormal) we use the marginal
    moments — μ = ``dist.mean()``, σ = √``dist.variance()`` — i.e. the
    standard moment-matching BMA approximation.

    Fit (EM with Dirichlet prior):

    * E-step: γ_{ik} = w_k · L_{ik} / Σ_j w_j L_{ij}, where
      L_{ik} = N(y_i; μ_{k,i}, σ_{k,i}).
    * M-step: α_n_k = α_0 + Σ_i s_i · γ_{ik}, then
      w_k = α_n_k / Σ_j α_n_j (posterior mean).

    s_i = sample_weight_i (1 if unweighted). Iterates until
    ‖w_new − w‖∞ < ``tol`` or ``max_iter`` is reached — non-convergence
    raises (Rule #0.5; partial weights would silently misweight tails).

    Predict at new x*: the pipeline re-runs upstreams on the inference
    rows; each contributes (μ_{k,*}, σ_{k,*}). Output is
    ``MixtureNormalForecast`` with weights = posterior mean w broadcast
    to (N, K).

    Why this beats ``Stacking``:

    * Per-row output σ — the mixture's standard deviation grows wherever
      upstream μ̂'s disagree on that row. ``Stacking``'s σ̂ is one scalar
      from training residuals.
    * No σ̂ → 0 collapse (the v0.1 ``Stacking`` pathology). The mixture
      σ is bounded below by min_k σ_{k,i}.
    * Weights live on the simplex (no extrapolation pathology from
      unconstrained OLS coefficients).
    """

    deps: tuple[str, ...]
    alpha_prior: float = 1.0
    max_iter: int = 200
    tol: float = 1e-6
    name: str = "BMAStacking"
    weights_: np.ndarray | None = field(default=None, init=False)
    alpha_n_: np.ndarray | None = field(default=None, init=False)
    n_iter_: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.depends_on = tuple(self.deps)
        if self.alpha_prior <= 0:
            raise ValueError(
                f"BMAStacking: alpha_prior must be > 0 (got {self.alpha_prior}); "
                "improper Dirichlet priors not supported."
            )

    @staticmethod
    def _upstream_moments(dist: Any, N: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract per-row (μ, σ) from any parametric upstream via the
        DistributionForecast moment API. Reject non-parametric backings."""
        if dist.backing.value != "parametric":
            raise NotImplementedError(
                f"BMAStacking expects parametric upstream; "
                f"{name!r} is {dist.backing}"
            )
        mu = np.asarray(dist.mean(), dtype=float)
        var = np.asarray(dist.variance(), dtype=float)
        if mu.shape != (N,) or var.shape != (N,):
            raise ValueError(
                f"BMAStacking: upstream {name!r} returned mean/variance with "
                f"shape {mu.shape}/{var.shape}, expected ({N},)"
            )
        if np.any(var <= 0):
            raise ValueError(
                f"BMAStacking: upstream {name!r} has non-positive variance "
                "on some rows — likelihood would be undefined."
            )
        return mu, np.sqrt(var)

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
                f"BMAStacking.fit needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        y = np.asarray(y, dtype=float)
        N = y.shape[0]
        # Row-alignment guard. Same contract as Stacking — upstream ids
        # must agree with each other and with the caller's ids (if given).
        upstream_ids = None
        for name in self.depends_on:
            d = deps_oof[name]
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"BMAStacking.fit: deps_oof[{name!r}].ids does not match "
                    "the first upstream's ids — mixture rows would be misaligned"
                )
        if (
            ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(ids), upstream_ids)
        ):
            raise ValueError(
                "BMAStacking.fit: caller's ids do not match deps_oof ids — "
                "rows would be misaligned"
            )
        # Collect per-row moments.
        K = len(self.depends_on)
        mu = np.empty((N, K))
        sigma = np.empty((N, K))
        for j, name in enumerate(self.depends_on):
            mu[:, j], sigma[:, j] = self._upstream_moments(deps_oof[name], N, name)
        # Likelihood matrix L_ik = N(y_i; μ_{k,i}, σ_{k,i}).
        z = (y[:, None] - mu) / sigma
        log_L = -0.5 * z ** 2 - np.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        # Numerical-stability trick: subtract per-row max before exp.
        log_L_max = log_L.max(axis=1, keepdims=True)
        L = np.exp(log_L - log_L_max)  # (N, K), per-row max == 1
        s = (
            np.asarray(sample_weight, dtype=float)
            if sample_weight is not None
            else np.ones(N)
        )
        if s.shape != (N,) or np.any(s < 0):
            raise ValueError(
                "BMAStacking: sample_weight must be 1-D, same length as y, non-negative."
            )
        # EM loop.
        w = np.full(K, 1.0 / K)
        for it in range(self.max_iter):
            num = w[None, :] * L            # (N, K)
            denom = num.sum(axis=1, keepdims=True)
            if np.any(denom <= 0):
                raise ValueError(
                    "BMAStacking.fit: row likelihood is zero under all components — "
                    "upstream μ̂'s sit too far from y on some rows. Check upstream "
                    "fit or widen σ_floor on upstreams."
                )
            gamma = num / denom              # (N, K), rows sum to 1
            alpha_n = float(self.alpha_prior) + (s[:, None] * gamma).sum(axis=0)
            w_new = alpha_n / alpha_n.sum()
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                self.n_iter_ = it + 1
                break
            w = w_new
        else:
            raise RuntimeError(
                f"BMAStacking.fit: EM did not converge in {self.max_iter} "
                f"iterations (last Δw = {np.max(np.abs(w_new - w)):.3g}). "
                "Raise max_iter or check upstream quality."
            )
        self.weights_ = w
        self.alpha_n_ = alpha_n
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        if self.weights_ is None:
            raise RuntimeError("BMAStacking.predict_dist called before fit")
        if not deps_oof:
            raise ValueError("BMAStacking.predict_dist needs deps_oof")
        ids_arr = np.asarray(ids)
        N = ids_arr.shape[0]
        for name in self.depends_on:
            if not np.array_equal(deps_oof[name].ids, ids_arr):
                raise ValueError(
                    f"BMAStacking.predict_dist: deps_oof[{name!r}].ids does "
                    "not match caller ids — mixture rows would be misaligned"
                )
        K = len(self.depends_on)
        mu = np.empty((N, K))
        sigma = np.empty((N, K))
        for j, name in enumerate(self.depends_on):
            mu[:, j], sigma[:, j] = self._upstream_moments(deps_oof[name], N, name)
        weights = np.broadcast_to(self.weights_, (N, K)).copy()
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_mixture_normal(
            weights=weights, mus=mu, sigmas=sigma,
            ids=ids_arr, timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# BayesianRidge — conjugate Bayesian linear regression. Predictive Student-t.
# ---------------------------------------------------------------------------


@dataclass
class BayesianRidge(BaseEstimator):
    """Conjugate Bayesian linear regression. Predictive distribution per row
    is Student-t (μ_n, σ_n, ν_n) — σ_n grows with feature-space distance from
    training data, so dispersion is regime-conditional without a boosted σ.

    Prior (Normal-Inverse-Gamma):

        β | σ²  ~ N(0,  σ² · diag([1/λ₀, 1/λ, …, 1/λ]))
        σ²     ~ Inv-Gamma(a_0, b_0)

    where ``λ = prior_precision`` is the L2 shrinkage on slopes (default
    1.0) and ``λ₀ = prior_precision_intercept`` is a near-flat prior on
    the intercept column (default 1e-6). ``a_0`` and ``b_0`` parametrise
    the variance prior; defaults (1e-3, 1e-3) are weakly informative.

    Posterior (closed form, with intercept handled by augmenting X with a
    column of ones):

        V_n⁻¹ = V_0⁻¹ + Xᵀ W X
        m_n   = V_n · Xᵀ W y                    (since m_0 = 0)
        a_n   = a_0 + N_eff / 2
        b_n   = b_0 + 0.5 · (yᵀ W y − m_nᵀ Xᵀ W y)

    where W is the diagonal of sample_weight (identity if not supplied)
    and N_eff = N (or Σ w_i for weighted fits).

    Predictive at new x*:

        ν*    = 2 · a_n
        μ*    = x*ᵀ m_n
        σ*²   = (b_n / a_n) · (1 + x*ᵀ V_n x*)

    The (1 + x*ᵀ V_n x*) factor is the posterior-uncertainty inflation:
    rows whose features sit far from the training set get wider
    predictive intervals automatically — that's the regime-conditional σ
    that ``Stacking``'s constant residual σ̂ cannot give you.

    Features are standardised by default (subtract train mean, divide by
    train std) before fitting. The prior precision then applies on a
    consistent feature scale; the same train statistics are reused at
    predict time.
    """

    prior_precision: float = 1.0
    prior_precision_intercept: float = 1e-6
    a_0: float = 1e-3
    b_0: float = 1e-3
    standardize: bool = True
    name: str = "BayesianRidge"
    depends_on: tuple[str, ...] = ()
    # fitted state
    m_n_: np.ndarray | None = field(default=None, init=False)
    V_n_: np.ndarray | None = field(default=None, init=False)
    a_n_: float | None = field(default=None, init=False)
    b_n_: float | None = field(default=None, init=False)
    x_mean_: np.ndarray | None = field(default=None, init=False)
    x_scale_: np.ndarray | None = field(default=None, init=False)

    def _design(self, X: np.ndarray, *, fit_phase: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"BayesianRidge: X must be 2-D; got shape {X.shape}"
            )
        if self.standardize:
            if fit_phase:
                self.x_mean_ = X.mean(axis=0)
                self.x_scale_ = X.std(axis=0, ddof=0)
                if np.any(self.x_scale_ <= 0):
                    bad = np.where(self.x_scale_ <= 0)[0].tolist()
                    raise ValueError(
                        f"BayesianRidge.fit: zero-variance column(s) in X "
                        f"at indices {bad}. Drop them before fitting "
                        "(Rule #0.5 — no silent zero-scale)."
                    )
            else:
                if self.x_mean_ is None or self.x_scale_ is None:
                    raise RuntimeError(
                        "BayesianRidge._design: predict before fit "
                        "(no stored standardisation stats)."
                    )
            X = (X - self.x_mean_) / self.x_scale_
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        y = np.asarray(y, dtype=float)
        A = self._design(X, fit_phase=True)
        N, k = A.shape
        if N <= k:
            raise ValueError(
                f"BayesianRidge.fit: N={N} ≤ d+1={k}; posterior df "
                "would be too small for a finite-variance Student-t."
            )
        lam = float(self.prior_precision)
        lam0 = float(self.prior_precision_intercept)
        if lam <= 0 or lam0 <= 0:
            raise ValueError(
                "BayesianRidge: prior_precision and prior_precision_intercept "
                "must be strictly positive (use a small value like 1e-6 "
                "for a near-flat prior — improper priors not supported)."
            )
        V0_inv = np.diag(np.concatenate([[lam0], np.full(k - 1, lam)]))
        if sample_weight is None:
            AtA = A.T @ A
            Aty = A.T @ y
            yTy = float(y @ y)
            n_eff = float(N)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != y.shape or np.any(w < 0):
                raise ValueError(
                    "BayesianRidge: sample_weight must be 1-D, same length "
                    "as y, non-negative."
                )
            WA = A * w[:, None]
            AtA = WA.T @ A
            Aty = WA.T @ y
            yTy = float((w * y) @ y)
            n_eff = float(w.sum())
        Vn_inv = V0_inv + AtA
        try:
            L = np.linalg.cholesky(Vn_inv)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "BayesianRidge.fit: posterior precision matrix is not "
                "positive definite — Xᵀ W X is rank-deficient and the prior "
                "is too weak to regularise it. Raise prior_precision or "
                "drop collinear columns."
            ) from e
        m_n = np.linalg.solve(L.T, np.linalg.solve(L, Aty))
        a_n = float(self.a_0) + 0.5 * n_eff
        b_n = float(self.b_0) + 0.5 * (yTy - float(m_n @ Aty))
        if b_n <= 0:
            raise ValueError(
                f"BayesianRidge.fit: posterior IG scale b_n = {b_n:.3g} ≤ 0. "
                "Predictive variance would be non-positive — usually means "
                "the design perfectly explains y (data leak) or the prior "
                "is too tight."
            )
        Vn = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(k)))
        df = 2.0 * a_n
        if df <= 2.0:
            raise ValueError(
                f"BayesianRidge.fit: posterior df = 2·a_n = {df:.3g} ≤ 2. "
                "Student-t variance is undefined; raise a_0 or fit on more rows."
            )
        self.m_n_ = m_n
        self.V_n_ = Vn
        self.a_n_ = a_n
        self.b_n_ = b_n
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        if self.m_n_ is None:
            raise RuntimeError("BayesianRidge.predict_dist called before fit")
        A = self._design(X, fit_phase=False)
        mu = A @ self.m_n_
        quad = np.einsum("ni,ij,nj->n", A, self.V_n_, A)
        scale_sq = (self.b_n_ / self.a_n_) * (1.0 + quad)
        if np.any(scale_sq <= 0):
            n_bad = int(np.sum(scale_sq <= 0))
            raise ValueError(
                f"BayesianRidge.predict_dist: non-positive predictive variance "
                f"on {n_bad} rows — numerical issue with V_n. Refit with higher "
                "prior_precision."
            )
        sigma = np.sqrt(scale_sq)
        df = np.full(mu.shape, 2.0 * self.a_n_)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_student_t(
            mu, sigma, df,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
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


