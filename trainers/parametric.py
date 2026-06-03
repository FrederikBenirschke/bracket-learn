"""Native parametric-distribution forecasters.

EMOS, NGBoostNormal, MixtureNormals (parametric normal / mixture);
StackedParametric (parametric meta-learner over positional ``upstream=``;
legacy name ``Stacking`` is preserved as a module-level alias for back-compat).
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
    MixtureNormalForecast,
    NormalForecast,
    ProvenanceMeta,
    StudentTForecast,
)
from bracketlearn.trainers._common import (
    _weighted_lstsq2,
)
from bracketlearn.trainers._compose_util import resolve_upstream, upstream_label

# Tuple of subclasses that count as "parametric upstream" for the stacking
# trainers. Used in isinstance dispatch; replaces the v0.5.x Backing enum
# check (`d.backing.value == "parametric"`).
_PARAMETRIC_BACKINGS = (NormalForecast, StudentTForecast, MixtureNormalForecast)

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
            import warnings
            warnings.warn(
                f"EMOS(fit_method='ols'): linear-in-variance fit gave "
                f"non-positive variance (c={c_unc:.3g}, d={d_unc:.3g}); "
                f"fell back to constant σ²={c_fallback:.3g}. ens_var is "
                f"not informative about residual scale on this training "
                f"set — consider EMOS(fit_method='crps_nelder_mead') "
                f"(exp-link variance) or check ensemble spread-skill.",
                UserWarning, stacklevel=3,
            )
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
        # Tag provenance when the OLS variance fit collapsed to a constant
        # σ at fit time, so downstream consumers can distinguish a true
        # regime-conditional σ̂(x) from a single-scalar fallback.
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        if getattr(self, "sigma_fit_was_constant_", False):
            prov = ProvenanceMeta(
                **{**prov.__dict__,
                   "extras": {**prov.extras, "emos_sigma_fit": "constant_fallback"}},
            )
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
# HeteroscedasticNormal — distributional linear regression. Both the mean
# and the (log) scale of a Normal are linear functions of arbitrary feature
# columns. Generalises EMOS (which is the special case μ-features=[ens_mean],
# σ-features=[ens_std]) to any predictors — cloud, wind, dewpoint, spread, …
# ---------------------------------------------------------------------------


@dataclass
class HeteroscedasticNormal(BaseEstimator):
    r"""Distributional linear regression for a Normal: ``N(μ(x), σ(x)²)``.

    Both moments are linear functions of (selectable) feature columns, fit
    jointly by maximum likelihood::

        μ(x)      = β_μ0 + xμ · β_μ
        log σ(x)  = β_σ0 + xσ · β_σ           (log link → σ > 0 always)

    where ``xμ = X[:, mu_idx]`` and ``xσ = X[:, sigma_idx]`` select which
    columns of the shared design matrix ``X`` drive the mean vs the scale
    (the two sets may overlap — e.g. cloud cover in both). When ``mu_idx`` /
    ``sigma_idx`` are ``None`` every column drives that moment.

    This is the parametric, interpretable counterpart to ``NGBoostNormal``
    (which boosts μ̂/σ̂ non-linearly) and the feature-driven generalisation
    of ``EMOS`` (whose mean is hard-wired to ``ens_mean`` and whose scale is
    hard-wired to ``ens_std``). Setting ``mu_idx=(i_mean,)`` and
    ``sigma_idx=(i_logstd,)`` recovers EMOS's modelling philosophy — affine
    mean, spread-driven scale — but now cloud / wind / dewpoint can enter
    *either* moment as additional columns. The coefficients are readable:
    each ``β_σ`` is the multiplicative log-scale response to its feature.

    Fit details:

    * **NLL.** Minimises the Gaussian negative log-likelihood
      ``Σ wᵢ·(log σᵢ + ½·zᵢ²)``, ``zᵢ = (yᵢ − μᵢ)/σᵢ`` (the constant
      ``½log 2π`` is dropped). ``sample_weight`` reweights rows.
    * **Optimiser.** L-BFGS-B with the analytic gradient
      ``∂/∂β_μ = −Aμᵀ(w·(y−μ)/σ²)`` and ``∂/∂β_σ = Aσᵀ(w·(1−z²))``.
      Initialised from an OLS mean fit + a constant log-scale at the
      residual std. Non-convergence raises (Rule #0.5 — no silent
      return of the init).
    * **Standardisation.** Columns are standardised (mean/std stored from
      fit, reused at predict) so the optimiser is well-conditioned across
      features on different scales. Predictions are invariant to this.
    * **Ridge.** ``l2 > 0`` adds an L2 penalty on the non-intercept
      coefficients of *both* heads — the low-N overfit guard.
    * **σ floor.** ``sigma_floor`` clamps σ̂ at predict time only (the
      log link already keeps it positive; the floor bounds confidence).

    Per Rule #0.5: non-finite ``X``/``y`` raise rather than being imputed
    here — the caller decides how to handle missing features.
    """

    mu_idx: tuple[int, ...] | None = None
    sigma_idx: tuple[int, ...] | None = None
    l2: float = 0.0
    sigma_floor: float = 1e-3
    standardize: bool = True
    maxiter: int = 5000
    name: str = "HeteroscedasticNormal"
    beta_mu_: np.ndarray | None = field(default=None, init=False)
    beta_sigma_: np.ndarray | None = field(default=None, init=False)
    _x_mean_: np.ndarray | None = field(default=None, init=False)
    _x_std_: np.ndarray | None = field(default=None, init=False)
    _mu_idx_: tuple[int, ...] | None = field(default=None, init=False)
    _sigma_idx_: tuple[int, ...] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.l2 < 0:
            raise ValueError(f"HeteroscedasticNormal: l2 must be ≥ 0; got {self.l2}")
        if self.sigma_floor <= 0:
            raise ValueError(
                f"HeteroscedasticNormal: sigma_floor must be > 0; got {self.sigma_floor}"
            )

    def _resolve_idx(self, F: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        mu_idx = tuple(range(F)) if self.mu_idx is None else tuple(self.mu_idx)
        sig_idx = tuple(range(F)) if self.sigma_idx is None else tuple(self.sigma_idx)
        for name, idx in (("mu_idx", mu_idx), ("sigma_idx", sig_idx)):
            bad = [i for i in idx if not (0 <= i < F)]
            if bad:
                raise ValueError(
                    f"HeteroscedasticNormal: {name} {bad} out of range for "
                    f"X with {F} columns"
                )
            if not idx:
                raise ValueError(
                    f"HeteroscedasticNormal: {name} is empty — each moment "
                    f"needs at least its intercept's companion feature set "
                    f"(pass at least one column)"
                )
        return mu_idx, sig_idx

    def _designs(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = (X - self._x_mean_) / self._x_std_
        ones = np.ones((X.shape[0], 1))
        Amu = np.column_stack([ones, Xs[:, self._mu_idx_]])
        Asig = np.column_stack([ones, Xs[:, self._sigma_idx_]])
        return Amu, Asig

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"HeteroscedasticNormal.fit: X must be 2-D; got shape {X.shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "HeteroscedasticNormal.fit: X has non-finite entries — impute "
                "or drop upstream; this estimator does not guess (Rule #0.5)"
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("HeteroscedasticNormal.fit: y has non-finite entries")
        N, F = X.shape
        self._mu_idx_, self._sigma_idx_ = self._resolve_idx(F)

        if self.standardize:
            xm = X.mean(axis=0)
            xs = X.std(axis=0)
            xs = np.where(xs > 0, xs, 1.0)  # constant column → no scaling
        else:
            xm = np.zeros(F)
            xs = np.ones(F)
        self._x_mean_, self._x_std_ = xm, xs

        Amu, Asig = self._designs(X)
        kmu, ksig = Amu.shape[1], Asig.shape[1]
        w = (np.ones(N) if sample_weight is None
             else np.asarray(sample_weight, dtype=float))

        # Init: OLS mean, constant log-scale at residual std.
        beta_mu0, *_ = np.linalg.lstsq(Amu, y, rcond=None)
        resid = y - Amu @ beta_mu0
        log_s0 = math.log(max(float(np.std(resid)), 1e-3))
        beta_sig0 = np.zeros(ksig)
        beta_sig0[0] = log_s0
        p0 = np.concatenate([beta_mu0, beta_sig0])

        def nll_and_grad(p: np.ndarray) -> tuple[float, np.ndarray]:
            bmu = p[:kmu]
            bsig = p[kmu:]
            mu = Amu @ bmu
            eta = Asig @ bsig
            sigma = np.exp(eta)
            z = (y - mu) / sigma
            nll = float(np.sum(w * (eta + 0.5 * z ** 2)))
            gmu = -Amu.T @ (w * (y - mu) / sigma ** 2)
            gsig = Asig.T @ (w * (1.0 - z ** 2))
            if self.l2 > 0:
                nll += self.l2 * (float(bmu[1:] @ bmu[1:]) + float(bsig[1:] @ bsig[1:]))
                gmu[1:] += 2.0 * self.l2 * bmu[1:]
                gsig[1:] += 2.0 * self.l2 * bsig[1:]
            return nll, np.concatenate([gmu, gsig])

        res = minimize(
            nll_and_grad, p0, jac=True, method="L-BFGS-B",
            options={"maxiter": self.maxiter},
        )
        if not res.success:
            raise RuntimeError(
                f"HeteroscedasticNormal.fit: optimiser did not converge "
                f"({res.message}); refusing to return the unfit init (Rule #0.5)"
            )
        self.beta_mu_ = res.x[:kmu]
        self.beta_sigma_ = res.x[kmu:]
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        if self.beta_mu_ is None:
            raise RuntimeError("HeteroscedasticNormal.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "HeteroscedasticNormal.predict_dist: X has non-finite entries — "
                "impute or drop upstream (Rule #0.5)"
            )
        Amu, Asig = self._designs(X)
        mu = Amu @ self.beta_mu_
        sigma = np.maximum(np.exp(Asig @ self.beta_sigma_), self.sigma_floor)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_normal(
            mu, sigma, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# StackedParametric — DistForecaster meta-learner over upstream μ (and
# optionally σ), received positionally via ``upstream=[...]`` under a
# ``Stacker``. Legacy name ``Stacking`` aliased below.
# ---------------------------------------------------------------------------


@dataclass
class StackedParametric(BaseEstimator):
    """Meta-learner over upstream forecasters' parametric outputs.

    Defaults reproduce v0.1 ``Stacking`` behaviour exactly: OLS over
    upstream μ with intercept (unconstrained), constant σ̂ from residual
    std, Gaussian output. The optional knobs below widen the surface.

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

    Upstream forecasts arrive **positionally** via ``upstream=[dist, ...]``
    (the ``Stacker`` contract) — this reads ``.params['mu']`` (and
    ``['sigma']`` when ``sigma_method='geometric_mean_upstream'``) from each,
    in declared order.
    """

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
        upstream: list[Any] | None = None,
    ) -> Self:
        ups = resolve_upstream(upstream, where="StackedParametric.fit")
        y = np.asarray(y, dtype=float)
        # Stack upstream μ predictions row-aligned. We REQUIRE that each
        # upstream dist's .ids matches our (X, y) row order (no silent
        # misalignment). If the caller passes ids, we check
        # them; if not, we still require all upstream dists to agree on
        # their own ids vectors (else the meta-learner builds rows from
        # mis-zipped predictions).
        upstream_ids = None
        for i, d in enumerate(ups):
            label = upstream_label(i)
            if not isinstance(d, _PARAMETRIC_BACKINGS):
                raise NotImplementedError(
                    f"StackedParametric expects parametric upstream; "
                    f"{label} is {type(d).__name__}"
                )
            if d.params["mu"].shape[0] != y.shape[0]:
                raise ValueError(
                    f"StackedParametric.fit: upstream {label} has N={d.params['mu'].shape[0]} "
                    f"but y has N={y.shape[0]}"
                )
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"StackedParametric.fit: upstream {label}.ids does not match the "
                    f"first upstream's ids — meta-learner rows would be misaligned"
                )
        if (
            ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(ids), upstream_ids)
        ):
            raise ValueError(
                "StackedParametric.fit: caller's ids do not match upstream ids — "
                "rows would be misaligned"
            )
        cols = [d.params["mu"] for d in ups]
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
                resid, y_scale, sample_weight, ups,
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
        ups: list[Any],
    ) -> None:
        cols_sigma = self._collect_upstream_sigma(ups, where="fit")
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
        ups: list[Any],
        *,
        where: str,
    ) -> list[np.ndarray]:
        cols: list[np.ndarray] = []
        for i, d in enumerate(ups):
            label = upstream_label(i)
            if "sigma" not in d.params:
                raise ValueError(
                    f"StackedParametric({where}, sigma_method='geometric_mean_upstream'): "
                    f"upstream {label} has no σ in params "
                    f"(type={type(d).__name__}); either pick "
                    f"sigma_method='constant' or feed parametric upstreams"
                )
            s = np.asarray(d.params["sigma"], dtype=float)
            if np.any(s <= 0):
                n_bad = int(np.sum(s <= 0))
                raise ValueError(
                    f"StackedParametric({where}): upstream {label} has {n_bad} "
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
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        # At predict time, the driver must have re-run the upstream stages
        # on the current X; it passes their dist positionally.
        ups = resolve_upstream(upstream, where="StackedParametric.predict_dist")
        # Row-alignment check: each upstream's ids must match the caller's ids
        # exactly (no silent misalignment).
        ids_arr = np.asarray(ids)
        for i, d in enumerate(ups):
            if not np.array_equal(d.ids, ids_arr):
                raise ValueError(
                    f"StackedParametric.predict_dist: upstream "
                    f"{upstream_label(i)}.ids does not "
                    f"match caller ids — rows would be misaligned"
                )
        cols = [d.params["mu"] for d in ups]
        Z = np.column_stack(cols)
        mu = self.intercept_ + Z @ self.weights_
        if self.sigma_method == "constant":
            sigma_std = np.full_like(mu, self.sigma_)
        else:
            cols_sigma = self._collect_upstream_sigma(ups, where="predict_dist")
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
    """Bayesian model averaging meta-learner. DistForecaster over upstreams.

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

    alpha_prior: float = 1.0
    max_iter: int = 500
    tol: float = 1e-6
    name: str = "BMAStacking"
    weights_: np.ndarray | None = field(default=None, init=False)
    alpha_n_: np.ndarray | None = field(default=None, init=False)
    n_iter_: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.alpha_prior <= 0:
            raise ValueError(
                f"BMAStacking: alpha_prior must be > 0 (got {self.alpha_prior}); "
                "improper Dirichlet priors not supported."
            )

    @staticmethod
    def _upstream_moments(dist: Any, N: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract per-row (μ, σ) from any parametric upstream via the
        DistributionForecast moment API. Reject non-parametric backings."""
        if not isinstance(dist, _PARAMETRIC_BACKINGS):
            raise NotImplementedError(
                f"BMAStacking expects parametric upstream; "
                f"{name} is {type(dist).__name__}"
            )
        mu = np.asarray(dist.mean(), dtype=float)
        var = np.asarray(dist.variance(), dtype=float)
        if mu.shape != (N,) or var.shape != (N,):
            raise ValueError(
                f"BMAStacking: upstream {name} returned mean/variance with "
                f"shape {mu.shape}/{var.shape}, expected ({N},)"
            )
        if np.any(var <= 0):
            raise ValueError(
                f"BMAStacking: upstream {name} has non-positive variance "
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
        upstream: list[Any] | None = None,
    ) -> Self:
        ups = resolve_upstream(upstream, where="BMAStacking.fit")
        y = np.asarray(y, dtype=float)
        N = y.shape[0]
        # Row-alignment guard. Same contract as Stacking — upstream ids
        # must agree with each other and with the caller's ids (if given).
        upstream_ids = None
        for i, d in enumerate(ups):
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"BMAStacking.fit: upstream {upstream_label(i)}.ids "
                    "does not match the first upstream's ids — mixture rows would be misaligned"
                )
        if (
            ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(ids), upstream_ids)
        ):
            raise ValueError(
                "BMAStacking.fit: caller's ids do not match upstream ids — "
                "rows would be misaligned"
            )
        # Collect per-row moments.
        K = len(ups)
        mu = np.empty((N, K))
        sigma = np.empty((N, K))
        for j, d in enumerate(ups):
            mu[:, j], sigma[:, j] = self._upstream_moments(
                d, N, upstream_label(j),
            )
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
        # EM loop. Convergence on Δ weighted-log-likelihood (the EM objective)
        # rather than Δw — when upstreams are near-duplicates, w drifts
        # linearly toward the fixed point with no real change in the
        # objective, and tol-on-Δw spuriously fails. Δll is the proper
        # criterion (and Σ Δll = 0 implies w is stationary).
        w = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        delta_ll = np.inf
        # Constant offset from the per-row log_L_max subtraction (cancels in
        # γ but matters for the true log-likelihood reported below).
        offset = float((s * log_L_max[:, 0]).sum())
        for it in range(self.max_iter):
            num = w[None, :] * L            # (N, K)
            denom = num.sum(axis=1, keepdims=True)
            if np.any(denom <= 0):
                raise ValueError(
                    "BMAStacking.fit: row likelihood is zero under all components — "
                    "upstream μ̂'s sit too far from y on some rows. Check upstream "
                    "fit or widen σ_floor on upstreams."
                )
            # Marginal log-likelihood under the current mixture weights.
            ll = float((s * np.log(denom[:, 0])).sum()) + offset
            delta_ll = ll - prev_ll
            if it > 0 and abs(delta_ll) < self.tol * max(abs(ll), 1.0):
                self.n_iter_ = it + 1
                break
            gamma = num / denom              # (N, K), rows sum to 1
            alpha_n = float(self.alpha_prior) + (s[:, None] * gamma).sum(axis=0)
            w = alpha_n / alpha_n.sum()
            prev_ll = ll
        else:
            raise RuntimeError(
                f"BMAStacking.fit: EM did not converge in {self.max_iter} "
                f"iterations (last Δlog-lik = {delta_ll:.3g}). "
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
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        if self.weights_ is None:
            raise RuntimeError("BMAStacking.predict_dist called before fit")
        ups = resolve_upstream(upstream, where="BMAStacking.predict_dist")
        ids_arr = np.asarray(ids)
        N = ids_arr.shape[0]
        for i, d in enumerate(ups):
            if not np.array_equal(d.ids, ids_arr):
                raise ValueError(
                    f"BMAStacking.predict_dist: upstream "
                    f"{upstream_label(i)}.ids does "
                    "not match caller ids — mixture rows would be misaligned"
                )
        K = len(ups)
        mu = np.empty((N, K))
        sigma = np.empty((N, K))
        for j, d in enumerate(ups):
            mu[:, j], sigma[:, j] = self._upstream_moments(
                d, N, upstream_label(j),
            )
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
    ) -> Self:
        y = np.asarray(y, dtype=float)
        A = self._design(X, fit_phase=True)
        N, k = A.shape
        if k >= N:
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
# HierarchicalNormal — cross-site partial-pooling regression. Empirical Bayes
# on the shrinkage variance τ²; closed-form posterior per site.
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalNormal(BaseEstimator):
    """Hierarchical normal regression with site-level partial pooling.

    Cross-site (Form C) model. For each row i belonging to site s_i with
    K-dim feature vector x_i:

        y_i      = x_iᵀ β_{s_i} + ε_i,    ε_i ~ N(0, σ²)
        β_s | β₀ ~ N(β₀, τ² · I_K)        site coefficients ∈ R^K
        β₀        flat (improper)

    Each site has its own coefficient vector β_s. All sites' coefs are
    shrunk toward the global mean β₀ by an amount τ that the data
    itself estimates (empirical Bayes — Type-II marginal-likelihood
    maximisation over (log σ², log τ²); β₀ profiled out by GLS).

    Predictive at a new row in site s:

        μ̂   = xᵀ E[β_s | data]
        σ̂²  = σ² + xᵀ Cov(β_s | data) x

    For a row in a site not seen at fit time, predictive uses β₀ with
    the marginal prior τ² added to the posterior on β₀ (proper
    Bayesian predictive for a new group). Raises by default
    (``allow_unseen_sites=False``) — Rule #0.5.

    Inputs require a ``groups`` array of length N giving the site
    identifier per row (str or int). Features are standardised before
    fit (with stored stats reused at predict time).

    Pipeline integration: this trainer needs ``groups`` at both fit and
    predict time. ``WalkForward`` threads ``groups`` through to any node whose
    signature declares it; run standalone or as a ``Pipeline`` node under
    ``WalkForward(...).fit_predict(..., groups=...)``. Closed-form fit; no MCMC.

    Computational shortcut: per-site Σ_s = σ²·I + τ²·X_s X_sᵀ is a
    rank-K perturbation of σ²·I, so by Woodbury we only invert K×K
    matrices regardless of per-site n_s. Fit cost ≈ O(N·K² + S·K³·iters).
    """

    allow_unseen_sites: bool = False
    standardize: bool = True
    sigma2_init: float = 1.0
    tau2_init: float = 1.0
    max_iter: int = 500
    name: str = "HierarchicalNormal"
    # fitted state
    sigma2_: float | None = field(default=None, init=False)
    tau2_: float | None = field(default=None, init=False)
    beta_0_: np.ndarray | None = field(default=None, init=False)
    V_beta_0_: np.ndarray | None = field(default=None, init=False)
    site_m_: dict[Any, np.ndarray] | None = field(default=None, init=False)
    site_V_: dict[Any, np.ndarray] | None = field(default=None, init=False)
    x_mean_: np.ndarray | None = field(default=None, init=False)
    x_scale_: np.ndarray | None = field(default=None, init=False)
    n_iter_: int | None = field(default=None, init=False)

    def _design(self, X: np.ndarray, *, fit_phase: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"HierarchicalNormal: X must be 2-D; got shape {X.shape}"
            )
        if self.standardize:
            if fit_phase:
                self.x_mean_ = X.mean(axis=0)
                self.x_scale_ = X.std(axis=0, ddof=0)
                if np.any(self.x_scale_ <= 0):
                    bad = np.where(self.x_scale_ <= 0)[0].tolist()
                    raise ValueError(
                        f"HierarchicalNormal.fit: zero-variance column(s) "
                        f"at indices {bad}; drop them before fitting."
                    )
            else:
                if self.x_mean_ is None or self.x_scale_ is None:
                    raise RuntimeError(
                        "HierarchicalNormal._design: predict before fit."
                    )
            X = (X - self.x_mean_) / self.x_scale_
        return np.column_stack([np.ones(X.shape[0]), X])

    @staticmethod
    def _per_site_sufficient_stats(
        A_full: np.ndarray, y: np.ndarray, groups: np.ndarray,
    ) -> dict[Any, tuple[np.ndarray, np.ndarray, float, int]]:
        """Per site, precompute (X_sᵀX_s, X_sᵀy_s, y_sᵀy_s, n_s)."""
        out: dict[Any, tuple[np.ndarray, np.ndarray, float, int]] = {}
        for s in np.unique(groups):
            mask = groups == s
            X_s = A_full[mask]
            y_s = y[mask]
            out[s] = (
                X_s.T @ X_s,            # (K, K)
                X_s.T @ y_s,            # (K,)
                float(y_s @ y_s),       # scalar
                int(mask.sum()),
            )
        return out

    def _neg_log_marginal(
        self,
        log_psi: float,
        log_phi: float,
        stats: dict[Any, tuple[np.ndarray, np.ndarray, float, int]],
        K: int,
    ) -> float:
        """-log p(y | σ², τ²) after profiling out β₀ by GLS.

        Uses Woodbury so we only invert K×K matrices per site.
        """
        psi = math.exp(log_psi)
        phi = math.exp(log_phi)
        I_K = np.eye(K)
        Q = np.zeros((K, K))   # accumulator for GLS β̂_0 denominator
        g = np.zeros(K)        # numerator
        const_terms = 0.0
        quad_yy = 0.0
        for (A_s, c_s, d_s, n_s) in stats.values():
            # Woodbury core: H_s = (I_K/φ + A_s/ψ).
            H_s = I_K / phi + A_s / psi
            try:
                L_s = np.linalg.cholesky(H_s)
            except np.linalg.LinAlgError:
                return float("inf")
            # Σ_s^{-1} terms via Woodbury identity. Define
            #   M_s = A_s / ψ - (A_s / ψ²) · H_s^{-1} · A_s  (this is X_sᵀ Σ_s^{-1} X_s)
            #   u_s = c_s / ψ - (1/ψ²) · A_s · H_s^{-1} · c_s  (X_sᵀ Σ_s^{-1} y_s)
            #   q_s = d_s / ψ - (1/ψ²) · c_sᵀ · H_s^{-1} · c_s (y_sᵀ Σ_s^{-1} y_s)
            H_inv_A = np.linalg.solve(L_s.T, np.linalg.solve(L_s, A_s))
            H_inv_c = np.linalg.solve(L_s.T, np.linalg.solve(L_s, c_s))
            M_s = A_s / psi - (A_s @ H_inv_A) / (psi ** 2)
            u_s = c_s / psi - (A_s @ H_inv_c) / (psi ** 2)
            q_s = d_s / psi - float(c_s @ H_inv_c) / (psi ** 2)
            Q += M_s
            g += u_s
            quad_yy += q_s
            # log|Σ_s| = n_s log ψ + log|I_K + (φ/ψ) A_s| = n_s log ψ + log|φ H_s|
            #         = n_s log ψ + K log φ + log|H_s|
            log_det_H_s = 2.0 * float(np.log(np.diag(L_s)).sum())
            const_terms += n_s * log_psi + K * log_phi + log_det_H_s
        # GLS β̂_0.
        try:
            L_Q = np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            return float("inf")
        beta_0 = np.linalg.solve(L_Q.T, np.linalg.solve(L_Q, g))
        log_det_Q = 2.0 * float(np.log(np.diag(L_Q)).sum())
        # Marginal quadratic form: yᵀΣ^{-1}y - β̂_0ᵀ Q β̂_0 - log|Q| (Schur).
        quad = quad_yy - float(beta_0 @ g)
        N_total = sum(n_s for (_, _, _, n_s) in stats.values())
        nll = 0.5 * (
            N_total * math.log(2.0 * math.pi)
            + const_terms
            + quad
            + log_det_Q
        )
        return float(nll)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        if groups is None:
            raise ValueError(
                "HierarchicalNormal.fit: groups (site identifier per row) "
                "is required; this trainer has no fallback."
            )
        if sample_weight is not None:
            raise NotImplementedError(
                "HierarchicalNormal: sample_weight not yet threaded through "
                "the marginal-likelihood optimisation."
            )
        groups = np.asarray(groups)
        y = np.asarray(y, dtype=float)
        if groups.shape != y.shape:
            raise ValueError(
                f"groups shape {groups.shape} != y shape {y.shape}"
            )
        A = self._design(X, fit_phase=True)
        N, K = A.shape
        if y.shape[0] != N:
            raise ValueError(f"X has N={N} but y has N={y.shape[0]}")
        stats = self._per_site_sufficient_stats(A, y, groups)
        if any(n_s < 1 for (_, _, _, n_s) in stats.values()):
            raise ValueError("HierarchicalNormal: every site needs ≥1 row")
        if len(stats) < 2:
            raise ValueError(
                "HierarchicalNormal: needs ≥2 sites for pooling to be "
                "meaningful; got 1. Use BayesianRidge for single-site data."
            )
        # Optimise (log σ², log τ²) by Nelder-Mead on the negative marginal
        # log-likelihood. Bound-friendly via log parameterisation.
        x0 = np.array(
            [math.log(self.sigma2_init), math.log(self.tau2_init)],
            dtype=float,
        )
        res = minimize(
            lambda lp: self._neg_log_marginal(lp[0], lp[1], stats, K),
            x0,
            method="Nelder-Mead",
            options={
                "xatol": 1e-4, "fatol": 1e-5,
                "maxiter": self.max_iter, "adaptive": True,
            },
        )
        if not res.success and not np.isfinite(res.fun):
            raise RuntimeError(
                f"HierarchicalNormal.fit: marginal-likelihood optimisation "
                f"failed ({res.message}). Try different sigma2_init/tau2_init."
            )
        log_psi, log_phi = float(res.x[0]), float(res.x[1])
        self.sigma2_ = math.exp(log_psi)
        self.tau2_ = math.exp(log_phi)
        self.n_iter_ = int(res.nit)
        # Recompute β̂_0 and V_β0 at the fitted variance components.
        psi, phi = self.sigma2_, self.tau2_
        I_K = np.eye(K)
        Q = np.zeros((K, K))
        g = np.zeros(K)
        for (A_s, c_s, _, _) in stats.values():
            H_s = I_K / phi + A_s / psi
            L_s = np.linalg.cholesky(H_s)
            H_inv_A = np.linalg.solve(L_s.T, np.linalg.solve(L_s, A_s))
            H_inv_c = np.linalg.solve(L_s.T, np.linalg.solve(L_s, c_s))
            Q += A_s / psi - (A_s @ H_inv_A) / (psi ** 2)
            g += c_s / psi - (A_s @ H_inv_c) / (psi ** 2)
        L_Q = np.linalg.cholesky(Q)
        beta_0 = np.linalg.solve(L_Q.T, np.linalg.solve(L_Q, g))
        V_beta_0 = np.linalg.solve(L_Q.T, np.linalg.solve(L_Q, I_K))
        self.beta_0_ = beta_0
        self.V_beta_0_ = V_beta_0
        # Per-site posterior (m_s, V_s).
        site_m: dict[Any, np.ndarray] = {}
        site_V: dict[Any, np.ndarray] = {}
        for s, (A_s, c_s, _, _) in stats.items():
            V_s_inv = A_s / psi + I_K / phi
            L_V = np.linalg.cholesky(V_s_inv)
            rhs = c_s / psi + beta_0 / phi
            site_m[s] = np.linalg.solve(L_V.T, np.linalg.solve(L_V, rhs))
            site_V[s] = np.linalg.solve(L_V.T, np.linalg.solve(L_V, I_K))
        self.site_m_ = site_m
        self.site_V_ = site_V
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> DistributionForecast:
        if self.sigma2_ is None:
            raise RuntimeError("HierarchicalNormal.predict_dist called before fit")
        if groups is None:
            raise ValueError(
                "HierarchicalNormal.predict_dist: groups is required."
            )
        groups = np.asarray(groups)
        A = self._design(X, fit_phase=False)
        N = A.shape[0]
        if groups.shape != (N,):
            raise ValueError(
                f"groups shape {groups.shape} != ({N},)"
            )
        unseen = [s for s in np.unique(groups) if s not in self.site_m_]
        if unseen and not self.allow_unseen_sites:
            raise ValueError(
                f"HierarchicalNormal.predict_dist: sites {unseen} unseen at "
                "fit; pass allow_unseen_sites=True to use β₀ fallback "
                "(predictive σ will be larger to reflect missing site data)."
            )
        mu = np.empty(N)
        var = np.empty(N)
        psi, phi = self.sigma2_, self.tau2_
        # Unseen-site predictive: β_new ~ N(β̂_0, V_β0 + φ I).
        I_K = np.eye(A.shape[1]) if unseen else None
        for i in range(N):
            s = groups[i]
            x = A[i]
            if s in self.site_m_:
                m_s = self.site_m_[s]
                V_s = self.site_V_[s]
                mu[i] = float(x @ m_s)
                var[i] = psi + float(x @ V_s @ x)
            else:
                mu[i] = float(x @ self.beta_0_)
                V_new = self.V_beta_0_ + phi * I_K
                var[i] = psi + float(x @ V_new @ x)
        sigma = np.sqrt(var)
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
    model_: Any = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
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

    Treats X as already-curated vendor columns. NaN entries are honored
    as "vendor absent for this row":

    - **fit**: σ_v is computed per-column with NaN-skip semantics
      (``nanmean`` over (x_v − y)²); a column with zero finite entries
      raises.
    - **predict_dist**: each row gets weights ∝ (vendor present) and is
      renormalized to sum to 1; absent components carry zero weight and
      a placeholder μ/σ so downstream math stays finite. Rows with **all**
      vendors absent fall back to uniform weights with NaN μ — callers
      must handle those upstream.

    This mirrors the per-row vendor-presence semantics of the original
    snowflake ``mixture_normals`` trainer (dropping NaN vendors per row,
    never mean-imputing).
    """

    name: str = "MixtureNormals"
    sigma_floor: float = 0.5
    sigma_v_: np.ndarray | None = field(default=None, init=False)
    K_: int | None = field(default=None, init=False)
    vendor_trained_: np.ndarray | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"MixtureNormals expects 2-D X; got shape {X.shape}")
        K = X.shape[1]
        diffs = X - y[:, None]
        present = np.isfinite(diffs)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float)[:, None]
            num = np.where(present, w * diffs ** 2, 0.0).sum(axis=0)
            den = np.where(present, w, 0.0).sum(axis=0)
        else:
            num = np.where(present, diffs ** 2, 0.0).sum(axis=0)
            den = present.sum(axis=0).astype(float)
        vendor_trained = den > 0
        if not vendor_trained.any():
            raise RuntimeError(
                "MixtureNormals.fit: no vendor column has any finite training row"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_v = np.where(
                vendor_trained, np.sqrt(num / np.where(den > 0, den, 1.0)),
                np.inf,
            )
        # Floor only the trained vendors; untrained stay at +∞ so they
        # carry zero weight at predict time regardless of present-mask.
        sigma_v = np.where(
            vendor_trained, np.maximum(sigma_v, self.sigma_floor), np.inf,
        )
        self.sigma_v_ = sigma_v
        self.vendor_trained_ = vendor_trained
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
        # A component contributes only if (a) the vendor produced a value
        # for this row AND (b) σ_v was estimable on the train slice.
        vendor_trained = (
            self.vendor_trained_ if self.vendor_trained_ is not None
            else np.ones(self.K_, dtype=bool)
        )
        present = np.isfinite(X) & vendor_trained[None, :]
        n_present = present.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_w = np.where(
                n_present[:, None] > 0,
                present.astype(float) / np.maximum(n_present, 1)[:, None],
                # Row with no usable vendors: spread weight over trained
                # vendors with μ=0 placeholder. Mixture is degenerate but
                # finite; caller can detect via NaN realized-bracket prob.
                vendor_trained.astype(float)
                / max(int(vendor_trained.sum()), 1),
            )
        mus = np.where(present, X, 0.0)
        sigmas = np.broadcast_to(self.sigma_v_, (N, self.K_)).copy()
        # Replace any +∞ σ (untrained vendors) with sigma_floor as a
        # numerically safe placeholder; their weight is 0 so the component
        # contribution is zero regardless.
        sigmas = np.where(np.isfinite(sigmas), sigmas, self.sigma_floor)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_mixture_normal(
            weights=row_w, mus=mus, sigmas=sigmas,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


