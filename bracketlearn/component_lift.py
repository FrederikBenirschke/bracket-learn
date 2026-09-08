"""Turn raw point forecasts into per-component predictive distributions.

This is step 0 of Gneiting & Ranjan, "Combining Predictive Distributions"
(EJS 7:1747-1782, 2013), §4.2. It is the step that runs before any pooling
formula, and the step this repo currently skips.

Their procedure, verbatim from §4.2:

    Each ensemble member is a point forecast, which can be viewed as the
    most extreme form of an underdispersed density forecast. To address
    the underdispersion and obtain approximately neutrally dispersed
    components, we use the maximum likelihood method on the training data
    to fit, for each ensemble member i = 1, …, 8 individually, a Gaussian
    predictive density of the form  f_i = N(a_i + b_i x_ij, σ_i²).

Three parameters are fitted per member, the intercept a_i, the slope b_i,
and the member's own scale σ_i. On their data σ̂_i ranged 1.958–2.214, a
13% spread. Those differences survive into the pool, because every
combination formula (TLP, SLP, BLP, BMA) reads the component CDFs F_i.

Why this matters here, and how it differs from what the repo does
-----------------------------------------------------------------
``_meteo_features.add_skill_blend`` combines 10 vendors as points:

    dbf_v = x_v + bias_v                      # additive EWMA shift only
    w_v   ∝ 1 / max(recent_var_v, 0.05)²
    blend = Σ w_v·dbf_v / Σ w_v               # a single number

A single downstream EMOS then lifts that one number with one σ head. The
vendor-level combination therefore happens in temperature space, upstream
of any distribution existing, and three things are structurally
unreachable.

1. **Slope.** The de-bias is additive. A vendor whose forecast amplitude
   is miscalibrated, systematically over-reacting or under-reacting,
   cannot be corrected, only shifted.

2. **Per-vendor scale.** ``recent_var_v`` is a per-vendor variance
   estimate, already computed per station and EWMA'd. It is spent
   entirely on ranking vendors in the weight and then discarded, and it
   never becomes a σ. One blend-level σ from ``ens_std`` cannot express
   that HRRR is sharp and GEFS diffuse on a given row.

3. **Correlated vendors.** ``1/recent_var²`` is computed per vendor in
   isolation. There is no covariance term, so it cannot down-weight a
   redundant vendor. The paper's Table 10 zeroes ETA (w = 0.000) precisely
   because it shares an institutional origin with GFS. The same structure
   exists here, since nws_hourly and nbm are both NWS, and hrrr and
   gefs_p50 are both NCEP. A fitted pool over components can zero one, and
   a per-vendor precision weight cannot.

None of that makes precision weighting wrong. For unbiased, independent
estimators with known variances, w ∝ 1/σ² is the efficient linear
combination, and the registry records that it beat AdaHedge and Hedge here
with a stated mechanism, namely that vendors are correlated stochastic
estimators rather than adversarial experts. This module does not replace
it. It enables the comparison the repo cannot currently make.

Relation to the rest of bracketlearn
------------------------------------
``AffineNormal`` implements the ``Lifter`` protocol shape, mapping a point
to a distribution, but it fits many components at once and so takes arrays
rather than a single ``PointForecast``. Its output feeds
``bracketlearn.pool``. Once each vendor is an N(μ_v, σ_v), every formula
there applies, including SLP and BMA, which need per-component moments and
therefore cannot run on the PMF-only experts the repo currently pools.

The Pipeline branch this belongs to matters. The lifter path is the one
that is genuinely out-of-fold, fitting on ``[:half]``, predicting
``[half:]``, and then refitting. The calibrator path calls
``_core_predict_dist`` on rows the core model was already fitted on. A
component lift is a Lifter and so does not inherit that problem, but
callers still own the split, and ``fit`` must be handed rows the
components did not train on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from bracketlearn.base import BaseEstimator

__all__ = ["AffineNormal", "ComponentFit"]

BiasForm = Literal["affine", "shift", "none"]
ScaleForm = Literal["per_component", "shared", "conditional"]
FitMethod = Literal["mle", "ols_resid", "crps"]
DistForm = Literal["normal", "student_t"]

# Student-t degrees of freedom are searched on this grid rather than
# optimised continuously. The log-likelihood in nu is very flat above ~15,
# where t_30 and t_60 are visually indistinguishable from a normal, so a
# fitted real-valued nu would report spurious precision. The grid ends at 60
# and the fitter reports "normal-like" there rather than resolving further.
_NU_GRID = (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 60.0)

# Minimum finite (x, y) pairs before a component's parameters mean anything.
# Below this the slope is noise and σ is dominated by a handful of rows.
_MIN_COMPONENT_ROWS = 30
# Floor on any fitted scale, in the target's units, which are °F here. It
# prevents a component that happens to fit its training slice exactly from
# producing a near-degenerate density that then dominates every pool it
# enters.
_SIGMA_FLOOR = 1e-3


@dataclass
class ComponentFit:
    """Fitted (a, b, σ) for one component, plus what it was fitted on."""

    name: str
    intercept: float
    slope: float
    sigma: float
    n_rows: int
    coverage: float          # fraction of rows where this component reported
    converged: bool = True
    # Student-t only. ``sigma`` stays the scale parameter and is never the
    # standard deviation. For t_nu the standard deviation is
    # sigma*sqrt(nu/(nu-2)), which is larger, and it is undefined at nu<=2.
    # Anything comparing dispersion across dist families should call ``sd``
    # rather than reading ``sigma``.
    nu: float | None = None
    dist: DistForm = "normal"

    @property
    def sd(self) -> float:
        """Predictive standard deviation, comparable across dist families."""
        if self.dist == "normal":
            return self.sigma
        if self.nu is None:
            raise ValueError(f"{self.name}: student_t fit carries no nu")
        if self.nu <= 2.0:
            return float("inf")
        return self.sigma * math.sqrt(self.nu / (self.nu - 2.0))

    @property
    def is_heavy_tailed(self) -> bool:
        """nu low enough that the t is materially not a normal.

        At nu=15 the excess kurtosis is 6/(nu-4) = 0.55 and the 99th
        percentile differs from the normal's by ~4%. At the grid's top end,
        nu=60, the t is a normal for every practical purpose.
        """
        return self.dist == "student_t" and self.nu is not None and self.nu <= 15.0

    @property
    def is_amplitude_miscalibrated(self) -> bool:
        """Slope far from 1, the failure an additive de-bias cannot fix.

        A slope b < 1 means the component over-reacts, so its deviations from
        the mean are too large and get shrunk. A slope b > 1 means it
        under-reacts.
        """
        return abs(self.slope - 1.0) > 0.15


@dataclass(repr=False)
class AffineNormal(BaseEstimator):
    """Per-component Gaussian lift, ``f_i = N(a_i + b_i·x_i, σ_i²)``.

    Each component is fitted independently on the rows where that component
    reported. This is the practical departure from the paper, whose 8
    ensemble members always report, whereas vendor coverage here ranges from
    3% to 93% missing. A component is fitted on its own finite rows and
    carries its ``coverage``. Rows where it is silent yield NaN moments, and
    the caller or the pool drops it for that row. No imputed value ever
    stands in for a forecast (Rule #0.5).

    Parameters
    ----------
    bias
        ``"affine"`` fits intercept and slope, as in the paper. ``"shift"``
        fits the intercept alone with the slope pinned at 1. That is the form
        ``add_skill_blend`` currently uses, kept here so the two are
        comparable under one estimator. ``"none"`` passes the raw point
        through.
    scale
        ``"per_component"`` gives each component its own σ_i, as in the
        paper. ``"shared"`` fits one σ across all components, following
        Raftery et al. (2005) and BMA eq. (12). This is also the reason BMA
        and SLP coincide, since the ratio σ_shared/σ_i is SLP's spread
        adjustment c by another route. ``"conditional"`` regresses log σ on a
        supplied per-row covariate, so spread can vary by row rather than
        only by component.
    dist
        ``"normal"`` is the paper's step 0. ``"student_t"`` replaces the
        Gaussian with a scaled t_ν, fitting ν jointly with the scale on
        ``_NU_GRID``. The paper had no reason to reach for it, as its
        components were members of a single ensemble over 8
        near-exchangeable NWP runs. Vendor residuals here instead mix
        provider outages, station siting and occasional gross errors, which
        is the generating story that produces heavy tails. Under this option
        ``sigma`` is the scale rather than the standard deviation. Use
        ``ComponentFit.sd`` to compare dispersion across families.
    fit_method
        ``"mle"`` is the paper's. For a Gaussian with a fixed mean form it
        coincides with least squares plus the ML residual scale.
        ``"ols_resid"`` fits the mean by OLS and then takes the residual
        standard deviation with n-2 degrees of freedom, giving the same point
        estimate with an unbiased scale. ``"crps"`` minimises the closed-form
        Gaussian CRPS, which is less sensitive to a few large residuals than
        the log score.
    """

    bias: BiasForm = "affine"
    scale: ScaleForm = "per_component"
    dist: DistForm = "normal"
    fit_method: FitMethod = "mle"
    sigma_floor: float = _SIGMA_FLOOR
    min_rows: int = _MIN_COMPONENT_ROWS
    name: str = "AffineNormal"

    fits_: list[ComponentFit] | None = field(default=None, init=False)
    shared_sigma_: float | None = field(default=None, init=False)
    cond_coef_: tuple[float, float] | None = field(default=None, init=False)

    # ---------- fitting ----------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        names: list[str] | None = None,
        z: np.ndarray | None = None,
    ) -> AffineNormal:
        """Fit one Gaussian per component.

        ``X`` is ``(k, N)`` raw point forecasts, NaN where a component did
        not report. ``y`` is ``(N,)`` realized values. ``z`` is an optional
        ``(N,)`` covariate for ``scale="conditional"``.
        """
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        if Xa.ndim != 2:
            raise ValueError(f"AffineNormal.fit: X must be (k, N); got {Xa.shape}")
        k, n = Xa.shape
        if ya.shape[0] != n:
            raise ValueError(
                f"AffineNormal.fit: y has {ya.shape[0]} rows, X has N={n}"
            )
        if names is not None and len(names) != k:
            raise ValueError(
                f"AffineNormal.fit: {len(names)} names for k={k} components"
            )
        if self.scale == "conditional" and z is None:
            raise ValueError(
                "AffineNormal.fit: scale='conditional' needs a per-row "
                "covariate z (e.g. cross-vendor disagreement)."
            )

        fits: list[ComponentFit] = []
        all_resid: list[np.ndarray] = []
        for i in range(k):
            nm = names[i] if names else f"component_{i}"
            xi = Xa[i]
            ok = np.isfinite(xi) & np.isfinite(ya)
            n_ok = int(ok.sum())
            cov = n_ok / max(n, 1)
            if n_ok < self.min_rows:
                raise ValueError(
                    f"AffineNormal.fit: component {nm!r} has {n_ok} finite "
                    f"rows (<{self.min_rows}); its (a, b, σ) would be noise. "
                    f"Drop the component or lower min_rows deliberately."
                )
            a, b = self._fit_mean(xi[ok], ya[ok])
            resid = ya[ok] - (a + b * xi[ok])
            all_resid.append(resid)
            sig, nu = self._fit_scale_and_shape(resid)
            fits.append(
                ComponentFit(
                    name=nm, intercept=a, slope=b, sigma=sig,
                    n_rows=n_ok, coverage=cov, nu=nu, dist=self.dist,
                )
            )

        if self.scale == "shared":
            pooled = np.concatenate(all_resid)
            s, nu = self._fit_scale_and_shape(pooled)
            self.shared_sigma_ = s
            fits = [
                ComponentFit(
                    name=f.name, intercept=f.intercept, slope=f.slope,
                    sigma=s, n_rows=f.n_rows, coverage=f.coverage,
                    nu=nu, dist=self.dist,
                )
                for f in fits
            ]
        elif self.scale == "conditional":
            # Guarded at the top of fit, where scale='conditional'
            # requires z.
            assert z is not None
            self.cond_coef_ = self._fit_conditional_scale(Xa, ya, fits, z)

        self.fits_ = fits
        return self

    def _fit_mean(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Intercept and slope under the configured bias form."""
        if self.bias == "none":
            return 0.0, 1.0
        if self.bias == "shift":
            # Slope pinned at 1, so the intercept is the mean signed error.
            # This is add_skill_blend's form, expressed in the same object.
            return float(np.mean(y - x)), 1.0
        # The affine form is least squares, which for Gaussian errors is
        # also the maximum likelihood estimate of the mean parameters,
        # whatever the scale treatment.
        var = float(np.var(x))
        if var <= 0:
            # A constant component carries no slope information, so a shift
            # is used rather than a division by zero.
            return float(np.mean(y - x)), 1.0
        b = float(np.cov(x, y, bias=True)[0, 1] / var)
        a = float(np.mean(y) - b * np.mean(x))
        return a, b

    def _fit_scale_and_shape(
        self, resid: np.ndarray,
    ) -> tuple[float, float | None]:
        """Scale, and for Student-t the degrees of freedom alongside it.

        Dispatch is on ``self.dist`` so that callers never branch. Returns
        ``(sigma, None)`` for a normal and ``(scale, nu)`` for a t.
        """
        if self.dist == "normal":
            return self._fit_scale(resid), None
        return self._fit_scale_student_t(resid)

    def _fit_scale_student_t(
        self, resid: np.ndarray,
    ) -> tuple[float, float]:
        """Joint (scale, ν) maximum likelihood fit for centred residuals
        under a scaled t_ν.

        ν is profiled over ``_NU_GRID`` rather than optimised. For each
        candidate ν the conditional scale is fitted by EM, and the ν with the
        best log-likelihood wins. The likelihood in ν is flat at the top of
        the grid, so a continuous optimiser would return a precise-looking
        number the data does not support.

        The EM step is the standard Gaussian-scale-mixture one. Since t_ν is
        N(0, σ²/w) with w ~ Gamma(ν/2, ν/2),

            E[w_i | r_i] = (ν + 1) / (ν + r_i²/σ²)
            σ² ← mean(E[w_i] · r_i²)

        which is monotone in the likelihood and needs no derivatives. The
        update downweights large residuals by design. A maximum likelihood
        normal σ is dragged up by the tail, and that inflated σ is one
        candidate explanation for the overdispersion this module measures.
        """
        from scipy.stats import t as student_t

        r2 = np.asarray(resid, dtype=float) ** 2
        n = r2.size
        if n < 2:
            raise ValueError("AffineNormal: need ≥2 residuals to fit a scale")

        best: tuple[float, float, float] | None = None   # (loglik, sigma, nu)
        s2_start = max(float(np.mean(r2)), self.sigma_floor ** 2)
        for nu in _NU_GRID:
            s2 = s2_start
            for _ in range(200):
                w = (nu + 1.0) / (nu + r2 / s2)
                s2_new = float(np.mean(w * r2))
                s2_new = max(s2_new, self.sigma_floor ** 2)
                if abs(s2_new - s2) <= 1e-12 * max(s2, 1.0):
                    s2 = s2_new
                    break
                s2 = s2_new
            sig = math.sqrt(s2)
            ll = float(np.sum(student_t.logpdf(resid, df=nu, scale=sig)))
            if not math.isfinite(ll):
                continue
            if best is None or ll > best[0]:
                best = (ll, sig, nu)
        if best is None:
            raise ValueError(
                "AffineNormal: no Student-t (scale, ν) on the grid produced a "
                "finite log-likelihood: the residuals are degenerate."
            )
        return max(best[1], self.sigma_floor), best[2]

    def _fit_scale(self, resid: np.ndarray) -> float:
        if resid.size < 2:
            raise ValueError("AffineNormal: need ≥2 residuals to fit a scale")
        if self.fit_method == "ols_resid":
            dof = max(resid.size - 2, 1)
            s = float(np.sqrt(np.sum(resid**2) / dof))
        elif self.fit_method == "crps":
            s = self._fit_scale_crps(resid)
        else:  # mle
            s = float(np.sqrt(np.mean(resid**2)))
        if not math.isfinite(s) or s <= 0:
            raise ValueError(
                "AffineNormal: fitted σ is non-positive: the component fits "
                "its training rows exactly, which means the inputs are "
                "in-sample."
            )
        return max(s, self.sigma_floor)

    def _fit_scale_crps(self, resid: np.ndarray) -> float:
        """σ minimising the mean Gaussian CRPS of the centred residuals.

        CRPS(N(0,σ); r) = σ·[ z(2Φ(z)−1) + 2φ(z) − 1/√π ],  z = r/σ.
        It is less sensitive to a few large residuals than the log score, and
        so is offered alongside maximum likelihood rather than as a variant
        of it.
        """
        from scipy.optimize import minimize_scalar
        from scipy.stats import norm

        def mean_crps(log_s: float) -> float:
            s = math.exp(log_s)
            z = resid / s
            return float(
                np.mean(
                    s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                         - 1.0 / math.sqrt(math.pi))
                )
            )

        s0 = math.log(max(float(np.sqrt(np.mean(resid**2))), self.sigma_floor))
        # Bounded rather than bracketed. A bracket triple has to satisfy
        # f(xb) < f(xa) and f(xb) < f(xc), which fails whenever the maximum
        # likelihood start already sits at or beyond the CRPS optimum. That
        # is the common case, since CRPS wants a smaller sigma than maximum
        # likelihood on heavy tails.
        res = minimize_scalar(
            mean_crps, bounds=(s0 - 3.0, s0 + 3.0), method="bounded",
        )
        return float(math.exp(res.x))

    def _fit_conditional_scale(
        self, X: np.ndarray, y: np.ndarray,
        fits: list[ComponentFit], z: np.ndarray,
    ) -> tuple[float, float]:
        """Regress log|resid| on a per-row covariate, log σ = c0 + c1·z."""
        zs, ls = [], []
        za = np.asarray(z, dtype=float)
        for i, f in enumerate(fits):
            ok = np.isfinite(X[i]) & np.isfinite(y) & np.isfinite(za)
            r = y[ok] - (f.intercept + f.slope * X[i][ok])
            keep = np.abs(r) > 0
            # E[log|N(0,σ)|] = log σ − (γ + log 2)/2. Subtracting that
            # constant makes the intercept an unbiased estimate of log σ.
            ls.append(np.log(np.abs(r[keep])) + 0.5 * (np.euler_gamma + math.log(2)))
            zs.append(za[ok][keep])
        zz, ll = np.concatenate(zs), np.concatenate(ls)
        var = float(np.var(zz))
        if var <= 0:
            return float(np.mean(ll)), 0.0
        c1 = float(np.cov(zz, ll, bias=True)[0, 1] / var)
        c0 = float(np.mean(ll) - c1 * np.mean(zz))
        return c0, c1

    # ---------- prediction ----------

    def moments(
        self, X: np.ndarray, *, z: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(k, N)`` μ and σ per component. NaN where a component is silent.

        The NaN is deliberate. A silent vendor must drop out of whatever
        pools these moments rather than be imputed to a blend mean. The
        pool's own sleeping-component handling then renormalises the
        surviving weights.
        """
        if self.fits_ is None:
            raise RuntimeError("AffineNormal.moments called before fit")
        Xa = np.asarray(X, dtype=float)
        if Xa.shape[0] != len(self.fits_):
            raise ValueError(
                f"AffineNormal.moments: X has k={Xa.shape[0]} but the "
                f"estimator was fitted with k={len(self.fits_)}"
            )
        mu = np.full(Xa.shape, np.nan)
        sig = np.full(Xa.shape, np.nan)
        for i, f in enumerate(self.fits_):
            ok = np.isfinite(Xa[i])
            mu[i, ok] = f.intercept + f.slope * Xa[i][ok]
            if self.scale == "conditional":
                if z is None:
                    raise ValueError(
                        "AffineNormal.moments: scale='conditional' needs z"
                    )
                assert self.cond_coef_ is not None  # set by fit for this scale
                c0, c1 = self.cond_coef_
                za = np.asarray(z, dtype=float)
                s = np.exp(c0 + c1 * za[ok])
                sig[i, ok] = np.maximum(s, self.sigma_floor)
            else:
                sig[i, ok] = f.sigma
        return mu, sig

    def cdf(
        self, thresholds: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    ) -> np.ndarray:
        """P(Y <= threshold) under the fitted family, elementwise.

        This exists so that callers computing a PIT or bracket probability
        never hard-code ``norm.cdf``. Switching ``dist`` to ``"student_t"``
        and forgetting one call site mixes families, and the result surfaces
        as a dispersion number rather than as an error.
        """
        z = (np.asarray(thresholds, dtype=float) - mu) / sigma
        if self.dist == "normal":
            from scipy.stats import norm
            return norm.cdf(z)
        from scipy.stats import t as student_t
        nus = {f.nu for f in (self.fits_ or []) if f.nu is not None}
        if len(nus) != 1:
            raise ValueError(
                "AffineNormal.cdf: components carry different ν "
                f"({sorted(nus)}); call per component instead."
            )
        return student_t.cdf(z, df=nus.pop())

    # ---------- reporting ----------

    def report(self) -> str:
        """One line per component, the paper's Table 9 for this fit."""
        if self.fits_ is None:
            raise RuntimeError("AffineNormal.report called before fit")
        t = self.dist == "student_t"
        head = (
            f"{'component':22s} {'a':>8s} {'b':>7s} {'scale':>7s} "
            f"{'nu':>6s} {'sd':>7s} {'n':>6s} {'cov':>6s}  flag"
            if t else
            f"{'component':22s} {'a':>8s} {'b':>7s} {'sigma':>7s} "
            f"{'n':>6s} {'cov':>6s}  flag"
        )
        out = [head]
        for f in self.fits_:
            flags = []
            if f.is_amplitude_miscalibrated:
                flags.append("amplitude")
            if f.is_heavy_tailed:
                flags.append("heavy-tail")
            flag = " ".join(flags)
            if t:
                out.append(
                    f"{f.name:22s} {f.intercept:8.3f} {f.slope:7.3f} "
                    f"{f.sigma:7.3f} {f.nu:6.1f} {f.sd:7.3f} "
                    f"{f.n_rows:6d} {f.coverage:6.2f}  {flag}"
                )
            else:
                out.append(
                    f"{f.name:22s} {f.intercept:8.3f} {f.slope:7.3f} "
                    f"{f.sigma:7.3f} {f.n_rows:6d} {f.coverage:6.2f}  {flag}"
                )
        # Standard deviations are compared rather than scales. For a t the
        # scale understates dispersion by sqrt(nu/(nu-2)), so a scale-based
        # spread ratio is not comparable to the normal fit's.
        sds = [f.sd for f in self.fits_]
        out.append(
            f"sd spread: {min(sds):.3f}–{max(sds):.3f} "
            f"(ratio {max(sds)/max(min(sds), 1e-9):.2f}x)"
        )
        return "\n".join(out)
