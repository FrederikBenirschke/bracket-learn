"""Combination formulas for predictive distributions (Gneiting & Ranjan 2013).

Four aggregation families behind one interface, plus a ``k=1`` calibration
mode. Reference: Tilmann Gneiting and Roopesh Ranjan, "Combining Predictive
Distributions", Electronic Journal of Statistics 7:1747-1782 (2013),
arXiv:1106.1638.

Why this module exists
----------------------
Theorem 3.1 of that paper concerns a linear pool ``F = Σ wᵢFᵢ`` with
strictly positive weights. Its PIT is ``Z = Σ wᵢZᵢ``, and therefore

    var(Z) = ΣΣ wᵢwⱼ cov(Zᵢ, Zⱼ) ≤ max var(Zᵢ),

with strict inequality unless the component PITs are perfectly correlated.
A linear pool is thus at least as dispersed as its least-dispersed
component. If the components are neutrally dispersed and regular, the pool
is strictly *over*dispersed. By Theorem 3.3 no weight vector repairs this.
The linear pool is not *flexibly dispersive*, meaning its reachable set of
var(PIT) does not include the neutral value 1/12.

The defect belongs to the functional form rather than to the fit. A
well-fitted pool of well-fitted components can therefore be calibrated
worse than any component alone. In the paper's Seattle-Tacoma
daily-maximum-temperature study (§4.2, Table 11) the linear pool measured
var(PIT) 0.057 against ~0.070 for every individual member, on the same
test days.

The remedies, in ascending parameter count:

    TLP   Σ wᵢFᵢ(y)                          the baseline, nested at c=1, α=β=1
    SLP   Σ wᵢFᵢ⁰((y − μᵢ)/c)                one spread parameter (eq. 6)
    BLP   B_{α,β}(Σ wᵢFᵢ(y))                 beta-warped pool (eq. 8)
    GLP   h⁻¹(Σ wᵢ h(Fᵢ(y)))                 link-function pool (eq. 5)
    BMA   Σ wᵢ N(y; μᵢ, σ²)                  joint (w, σ) mixture fit (eq. 12)

BMA and SLP are the same object reached by different estimators, and the
paper's own numbers say so. On Seattle-Tacoma, SLP fits ĉ = 0.768 while
BMA's common σ̂ = 1.566 against member σᵢ ∈ [1.958, 2.214], a ratio of
0.707–0.800 that brackets ĉ. The fitted densities lie on top of each other
(§4.2) and the log scores differ by 0.002. The two differ only in where the
shrink is estimated. SLP fits c with the components held fixed, in two
stages. BMA estimates weights and a common σ jointly in one mixture model.
Having both behind one interface makes ĉ and σ̂/σᵢ directly comparable on
the same rows, as in the paper's Table 10.

BLP is *exchangeably flexibly dispersive* (Thm 3.9). As (α, β) range over
the positive quadrant it attains any var(PIT) in (0, 1/4). SLP is not
(Thm 3.7), but it is adequate whenever the components are neutrally
dispersed or underdispersed, which is the common case. GLP with weights
summing to at most 1 is incoherent (Thm 3.4). The probit link with weights
above 1 is coherent for Bernoulli (Example 3.5), the extremization case.
``weight_sum_max`` is therefore a parameter rather than the constant 1.

Locality, and BLP on bracket ladders
------------------------------------
BLP is *local*. ``G(y)`` depends on the components only through the values
``F₁(y), …, F_k(y)`` at that same ``y``. On a bracket ladder the cumulative
mass at each edge is exactly those values, so BLP is an exact pointwise map
on edge CDFs. It needs no resampling, no interpolation, and no assumption
about within-bracket shape.

SLP is not local. It needs each component's median ``μᵢ`` and evaluates a
recentred, rescaled CDF at points that are generally not edges. For
parametric components, with μ and σ available, that is exact. For
bracket-only components it requires interpolating within brackets, which
imposes a within-bracket shape the data never specified. This module
therefore implements SLP exactly for parametric input and refuses
bracket-only input unless ``allow_interpolated_slp=True`` is passed. Under
that flag the approximation is recorded in provenance rather than applied
without a record (Rule #0.5).

Splits are the caller's job
---------------------------
Nothing here knows about folds. ``fit`` consumes the rows it is handed and
fits on all of them. The caller is responsible for handing it out-of-sample
component predictions. This matches the contract the existing combiners in
``bracketlearn.trainers.combiners`` already assert in their docstrings.

The contract is enforced rather than assumed, because the failure is
directional and leaves no trace in the fitted diagnostics. Weights and
(α, β) are dispersion parameters fitted against how often the realized
value lands in the tails. Suppose the component predictions came from
models that had already seen these ``y``. Their residuals are then too
small, the pooled distribution looks too wide relative to them, and the fit
picks parameters that sharpen. Applied to genuine forecasts, that is
systematic overconfidence, giving narrow intervals and a U-shaped PIT.
``fit`` therefore computes a held-out dispersion check when given
``holdout=``, and ``leak_warning_`` records the verdict. See
``PoolFit.leak_warning_``.

The repo's own history on this is instructive. In
``prediction_market_weather.ml.trainers``, ``bl_emos_iso`` is the correct
pattern, with a per-fold refit inside ``run_trainer`` and disjoint fit and
apply rows. The deleted snowflake ``emos_calibrated.py`` was the incorrect
one, using a single time split with an isotonic fit on the first 60% of
days applied to all of them. ``Pipeline``'s calibrator branch still has the
incorrect shape. It calls ``_core_predict_dist`` on rows the core model was
fitted on, so do not route a pool through it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "PoolFit",
    "PoolFormula",
    "beta_warp",
    "fit_bma",
    "fit_pool",
    "neutral_pit_variance",
    "pit_variance",
    "pool_cdf",
    "spread_adjusted_cdfs",
]

# The variance of a Uniform(0, 1) PIT. By Definition 2.7(c) a forecast is
# overdispersed below this value, underdispersed above it, and neutrally
# dispersed at it. The attainable range is [0, 1/4].
NEUTRAL_PIT_VAR = 1.0 / 12.0

# Numerical floor. Probabilities are clipped away from {0, 1} before any
# log or link evaluation. The beta density diverges at the endpoints, and
# the log and probit links are undefined there.
_EPS = 1e-9

# Below this many rows the log-score surface is dominated by noise and the
# fitted (α, β) are not meaningful. This matches PITCalibrate's floor in
# lift.py.
_MIN_FIT_ROWS = 30

PoolFormula = Literal["tlp", "slp", "blp", "glp", "bma"]

_LINKS: dict[str, tuple[Any, Any]] = {}


def _init_links() -> None:
    """Link functions for the generalized linear pool, eq. (5) and Table 4.

    Each entry is ``(h, h_inv)`` on the open unit interval. Type A, the
    identity link, is the traditional linear pool. It is handled by the
    ``tlp`` branch.
    """
    from scipy.stats import norm

    _LINKS.update(
        {
            # Type C, h(x) = log x, the geometric pool.
            "log": (np.log, np.exp),
            # Type B, h(x) = 1/x, the harmonic pool. Range (1, ∞).
            "harmonic": (lambda x: 1.0 / x, lambda z: 1.0 / z),
            # Type D, h(x) = Φ⁻¹(x), the probit pool. It is the one link
            # that admits a coherent formula when weights exceed 1
            # (Example 3.5).
            "probit": (norm.ppf, norm.cdf),
        }
    )


def neutral_pit_variance() -> float:
    """var(PIT) of a probabilistically calibrated forecast, namely 1/12."""
    return NEUTRAL_PIT_VAR


def pit_variance(u: np.ndarray) -> float:
    """Sample variance of PIT values.

    Compare against :func:`neutral_pit_variance`. Below 1/12 the forecast is
    overdispersed, with intervals too wide and a hump-shaped PIT histogram.
    Above 1/12 it is underdispersed, with intervals too narrow and a
    U-shaped histogram.
    """
    u = np.asarray(u, dtype=float)
    finite = np.isfinite(u)
    if finite.sum() < 2:
        raise ValueError(
            f"pit_variance: need ≥2 finite PIT values; got {int(finite.sum())}."
        )
    return float(np.var(u[finite]))


def beta_warp(p: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """``B_{α,β}(p)``, the beta CDF applied pointwise to pooled CDF values.

    This is the whole of BLP's nonlinearity (eq. 8). At ``α = β = 1`` it is
    the identity and recovers the traditional linear pool.
    """
    from scipy.stats import beta as beta_dist

    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.asarray(beta_dist.cdf(p, alpha, beta), dtype=float)


def pool_cdf(
    component_cdfs: np.ndarray,
    *,
    weights: np.ndarray,
    formula: PoolFormula = "tlp",
    alpha: float = 1.0,
    beta: float = 1.0,
    link: str = "log",
) -> np.ndarray:
    """Combine component CDF values into one pooled CDF value.

    ``component_cdfs`` has shape ``(k, ...)``, with the component axis first
    and any trailing shape of rows and edges. The pooled values are returned
    with the component axis removed.

    All four formulas agree at ``α = β = 1`` under the identity link, where
    they reduce to the traditional linear pool.
    """
    F = np.asarray(component_cdfs, dtype=float)
    w = np.asarray(weights, dtype=float)
    if F.ndim < 1 or F.shape[0] != w.shape[0]:
        raise ValueError(
            f"pool_cdf: component axis mismatch: component_cdfs has "
            f"k={F.shape[0] if F.ndim else 0}, weights has k={w.shape[0]}."
        )
    w_shaped = w.reshape((-1,) + (1,) * (F.ndim - 1))

    if formula in ("tlp", "slp", "bma"):
        # All three pool linearly at this point. They differ only in how the
        # component CDFs handed in were built.
        #   tlp: components as given.
        #   slp: components already rescaled by c (spread_adjusted_cdfs).
        #   bma: components already rebuilt at the fitted common sigma.
        # bma is kept here rather than raising because fit_pool accepts it.
        # A caller that fits with formula='bma' and then predicts through
        # pool_cdf would otherwise hit an unknown-formula error on the
        # predict path alone, which is how it failed first.
        return np.sum(w_shaped * F, axis=0)

    if formula == "blp":
        return beta_warp(np.sum(w_shaped * F, axis=0), alpha, beta)

    if formula == "glp":
        if not _LINKS:
            _init_links()
        if link not in _LINKS:
            raise ValueError(
                f"pool_cdf: unknown link {link!r}; "
                f"available: {sorted(_LINKS)}."
            )
        h, h_inv = _LINKS[link]
        Fc = np.clip(F, _EPS, 1.0 - _EPS)
        pooled = h_inv(np.sum(w_shaped * h(Fc), axis=0))
        # The link's range need not be [0, 1] once weights are free to
        # exceed 1 (Example 3.5), so the inverse can land outside it. The
        # value is clipped rather than raised on, since this is the
        # documented extremization regime.
        return np.clip(np.asarray(pooled, dtype=float), 0.0, 1.0)

    raise ValueError(
        f"pool_cdf: unknown formula {formula!r}; "
        f"expected one of 'tlp', 'slp', 'blp', 'glp'."
    )


def spread_adjusted_cdfs(
    mu: np.ndarray,
    sigma: np.ndarray,
    edges: np.ndarray,
    *,
    spread_c: float,
    dist: str = "normal",
    nu: float | None = None,
) -> np.ndarray:
    """Component CDFs at bracket edges under SLP's spread adjustment, eq. (6).

    Under ``Gc(y) = Σ wᵢ Fᵢ⁰((y − μᵢ)/c)`` each component is recentred on its
    own median and rescaled by ``c`` *before* the linear pool. For Gaussian
    components that is exactly ``N(μᵢ, (c·σᵢ)²)``, so this path is exact and
    needs no within-bracket interpolation.

    A value ``c < 1`` sharpens, which suits neutrally dispersed or
    overdispersed components. A value ``c > 1`` widens, which suits
    underdispersed components. At ``c = 1`` the traditional linear pool is
    recovered. Berrocal et al. (2007) report estimates from 0.65 to 1.03,
    and the paper's Seattle-Tacoma fit is 0.768.

    Parameters
    ----------
    mu, sigma
        ``(k, N)`` per-component, per-row location and scale.
    edges
        ``(N, B+1)`` bracket edges, or ``(B+1,)`` shared across rows.
    dist, nu
        Component family. ``"normal"`` is the paper's. ``"student_t"`` with
        degrees of freedom ``nu`` supports components lifted by
        ``AffineNormal(dist="student_t")``, where ``sigma`` is the scale and
        the standard deviation is ``sigma*sqrt(nu/(nu-2))``. Passing t
        components through the Gaussian path raises no error and produces
        plausible bracket probabilities with the wrong tails. The family
        therefore travels with the moments.

    Returns
    -------
    np.ndarray
        ``(k, N, B+1)`` cumulative mass at edges, ready for
        :func:`pool_cdf`.
    """
    from scipy.stats import norm
    from scipy.stats import t as student_t

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.shape != sigma.shape or mu.ndim != 2:
        raise ValueError(
            f"spread_adjusted_cdfs: mu and sigma must both be (k, N); got "
            f"{mu.shape} and {sigma.shape}."
        )
    if spread_c <= 0:
        raise ValueError(
            f"spread_adjusted_cdfs: spread_c must be > 0; got {spread_c}."
        )
    if np.any(sigma <= 0):
        raise ValueError(
            "spread_adjusted_cdfs: sigma must be strictly positive; a "
            "degenerate component has no CDF to rescale."
        )
    e = np.asarray(edges, dtype=float)
    if e.ndim == 1:
        e = np.broadcast_to(e[None, :], (mu.shape[1], e.shape[0]))
    if e.shape[0] != mu.shape[1]:
        raise ValueError(
            f"spread_adjusted_cdfs: edges has N={e.shape[0]} but mu has "
            f"N={mu.shape[1]}."
        )
    if dist not in ("normal", "student_t"):
        raise ValueError(f"spread_adjusted_cdfs: unknown dist {dist!r}")
    if dist == "student_t":
        if nu is None:
            raise ValueError(
                "spread_adjusted_cdfs: dist='student_t' needs nu: without it "
                "the components would be silently pooled as Gaussians."
            )
        if nu <= 1.0:
            raise ValueError(f"spread_adjusted_cdfs: need nu > 1; got {nu}")
    scaled = sigma * spread_c
    _cdf = (norm.cdf if dist == "normal"
            else (lambda z: student_t.cdf(z, df=nu)))
    out = _cdf(
        (e[None, :, :] - mu[:, :, None]) / scaled[:, :, None]
    )
    # Pin the outer edges so every component carries full mass over the
    # ladder. Otherwise the tails leak and the pooled row no longer sums to
    # 1 after differencing.
    out[:, :, 0] = 0.0
    out[:, :, -1] = 1.0
    return np.asarray(out, dtype=float)


def fit_bma(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    *,
    alpha_prior: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float, int]:
    """Bayesian model averaging, a joint EM fit of weights and a common σ.

    The EM in :class:`bracketlearn.trainers.combiners.BMAStacking` is wrapped
    rather than reimplemented. That implementation already handles
    log-sum-exp stability and the Dirichlet prior, and it raises on
    non-convergence (Rule #0.5).

    One difference from that class is the reason this function exists.
    Raftery et al. (2005), and the paper's eq. (12), fit a common spread
    parameter σ shared across components rather than keeping each
    component's native σᵢ. That common σ is what makes BMA comparable to
    SLP's ``c``, since the ratio σ̂/σᵢ is SLP's spread adjustment by another
    name. ``BMAStacking`` uses a per-component σ and cannot produce that
    number.

    Returns ``(weights, sigma_common, n_iter)``.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)
    if mu.ndim != 2 or mu.shape != sigma.shape:
        raise ValueError(
            f"fit_bma: mu and sigma must both be (k, N); got {mu.shape} "
            f"and {sigma.shape}."
        )
    k, n = mu.shape
    if y.shape[0] != n:
        raise ValueError(
            f"fit_bma: y has {y.shape[0]} rows but mu has N={n}."
        )
    if np.any(sigma <= 0):
        raise ValueError("fit_bma: sigma must be strictly positive.")

    # Initialise the common σ at the mean component scale, then alternate
    # two steps. EM gives w for a fixed σ, and a closed-form
    # weighted-residual update gives σ for a fixed w.
    w = np.full(k, 1.0 / k)
    sig = float(np.mean(sigma))
    prev_ll = -np.inf
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        z = (y[None, :] - mu) / sig
        log_L = -0.5 * z**2 - math.log(sig) - 0.5 * math.log(2.0 * math.pi)
        log_L_max = log_L.max(axis=0, keepdims=True)
        L = np.exp(log_L - log_L_max)
        num = w[:, None] * L
        denom = num.sum(axis=0, keepdims=True)
        if np.any(denom <= 0):
            raise ValueError(
                "fit_bma: row likelihood is zero under all components: the "
                "component means sit too far from y. Check upstream fit."
            )
        ll = float(np.sum(np.log(denom[0]) + log_L_max[0]))
        if it > 0 and abs(ll - prev_ll) < tol * max(abs(ll), 1.0):
            break
        gamma = num / denom                       # (k, N) responsibilities
        alpha_n = float(alpha_prior) + gamma.sum(axis=1)
        w = alpha_n / alpha_n.sum()
        # M-step for the shared scale, the responsibility-weighted RMS
        # residual.
        sq = (y[None, :] - mu) ** 2
        sig = float(np.sqrt(np.sum(gamma * sq) / max(np.sum(gamma), _EPS)))
        if not math.isfinite(sig) or sig <= 0:
            raise ValueError(
                "fit_bma: common sigma collapsed to a non-positive value: "
                "component means fit y exactly, which means the inputs are "
                "in-sample."
            )
        prev_ll = ll
    else:
        raise RuntimeError(
            f"fit_bma: EM did not converge in {max_iter} iterations. Raise "
            f"max_iter or check component quality."
        )
    return w, sig, n_iter


@dataclass
class PoolFit:
    """Fitted parameters and the diagnostics needed to judge them.

    Every field a verdict could rest on is carried here rather than printed.
    A caller can then report the whole package, comprising the fit, its
    dispersion, its sample size, and whether the fit rows were trustworthy.
    """

    formula: PoolFormula
    weights: np.ndarray
    alpha: float = 1.0
    beta: float = 1.0
    spread_c: float = 1.0
    link: str = "log"

    n_fit_rows: int = 0
    n_excluded_open_tail: int = 0
    mean_log_score: float = float("nan")
    pit_var_fit: float = float("nan")
    pit_var_holdout: float = float("nan")
    leak_warning_: str | None = None
    converged: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def dispersion_verdict(self) -> str:
        """Where the fitted pool sits relative to neutral dispersion."""
        v = self.pit_var_holdout
        if not math.isfinite(v):
            v = self.pit_var_fit
        if not math.isfinite(v):
            return "unknown"
        if v < NEUTRAL_PIT_VAR * 0.90:
            return "overdispersed"
        if v > NEUTRAL_PIT_VAR * 1.10:
            return "underdispersed"
        return "neutral"


def _check_leakage(
    pit_var_fit: float,
    pit_var_holdout: float,
    *,
    tol: float = 0.25,
) -> str | None:
    """Compare fit-row against holdout dispersion.

    The in-sample failure mode is directional and specific. Component
    predictions produced by models that already saw these ``y`` have
    residuals that are too small, so their PIT clusters toward the middle
    and var(PIT) on the fit rows reads lower than it truly is. The fit then
    compensates by sharpening. Holdout dispersion materially higher than fit
    dispersion is the signature of this.

    Returns a message, or None when the two agree.
    """
    if not (math.isfinite(pit_var_fit) and math.isfinite(pit_var_holdout)):
        return None
    if pit_var_fit <= 0:
        return (
            "fit-row var(PIT) is 0: the component distributions place all "
            "mass at the realized values. This is the signature of fitting "
            "on in-sample component predictions."
        )
    rel = (pit_var_holdout - pit_var_fit) / pit_var_fit
    if rel > tol:
        return (
            f"holdout var(PIT) {pit_var_holdout:.4f} exceeds fit-row "
            f"var(PIT) {pit_var_fit:.4f} by {rel:.0%} (tol {tol:.0%}). The "
            f"component predictions on the fit rows are probably in-sample; "
            f"fitted parameters will be biased toward overconfidence."
        )
    return None


def _mean_log_score(
    pooled_pdf: np.ndarray,
    *,
    source: str,
) -> float:
    """Mean log score, eq. (10). Positively oriented, so higher is better."""
    pdf = np.asarray(pooled_pdf, dtype=float)
    if np.any(pdf < 0):
        raise ValueError(f"{source}: negative density in log-score input.")
    return float(np.mean(np.log(np.maximum(pdf, _EPS))))


def _bracket_pdf_at_realized(
    edge_cdfs: np.ndarray,
    realized_idx: np.ndarray,
) -> np.ndarray:
    """Mass the pooled CDF assigns to each row's realized bracket.

    ``edge_cdfs`` is the (N, B+1) cumulative mass at bracket edges. The
    density of the realized bracket is the difference across it. On a
    bracket ladder the log score is the log of this mass, which is the
    discrete analogue of eq. (10).
    """
    n = edge_cdfs.shape[0]
    rows = np.arange(n)
    hi = edge_cdfs[rows, realized_idx + 1]
    lo = edge_cdfs[rows, realized_idx]
    return hi - lo


def fit_pool(
    component_cdfs: np.ndarray,
    realized_idx: np.ndarray,
    *,
    formula: PoolFormula = "blp",
    link: str = "log",
    weight_sum_max: float = 1.0,
    fixed_weights: np.ndarray | None = None,
    holdout: tuple[np.ndarray, np.ndarray] | None = None,
    min_rows: int = _MIN_FIT_ROWS,
    moments: tuple[np.ndarray, np.ndarray] | None = None,
    edges: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> PoolFit:
    """Fit pool parameters by maximizing the mean log score.

    Parameters
    ----------
    component_cdfs
        ``(k, N, B+1)`` cumulative mass at bracket edges, per component and
        per row, with the component axis first. Each row's last edge value
        must be 1.0, the full mass. Rows whose realized value falls in an
        open tail have no well-defined density and must be excluded by the
        caller. See ``n_excluded_open_tail``.
    realized_idx
        ``(N,)`` index of the bracket containing each row's realized value.
    formula
        Which family to fit. ``"tlp"`` fits weights alone, ``"slp"`` adds the
        spread parameter ``c``, ``"blp"`` adds ``(α, β)``, and ``"glp"`` fits
        weights under the chosen link.
    weight_sum_max
        Upper bound on ``Σwᵢ``. The default 1.0 is the convex case. Values
        above 1 open the extremization regime, the only setting in which a
        generalized linear pool is coherent (Example 3.5). Ignored when
        ``fixed_weights`` is given.
    fixed_weights
        Hold weights fixed and fit only the shape parameters. This is the
        ``k=1`` calibration mode when ``k == 1``.
    holdout
        ``(component_cdfs, realized_idx)`` for rows not used in the fit. Used
        only for the dispersion check that populates ``leak_warning_``.
    moments, edges, y
        Required for ``formula="slp"`` and ``formula="bma"``, which are
        defined in continuous space and cannot be evaluated from edge CDFs
        alone. ``moments`` is ``(mu, sigma)``, each ``(k, N)``. ``edges`` is
        the bracket grid and ``y`` the realized values. Supplying them for
        the other formulas raises rather than being ignored.

    SLP on bracket-only components
    ------------------------------
    SLP is not local. It evaluates each component's CDF at rescaled points
    that are generally not bracket edges. With ``moments`` it is exact. Given
    only ``component_cdfs``, computing it would require interpolating within
    brackets, which imposes a within-bracket shape the data never specified.
    This function raises instead (Rule #0.5).
    """
    from scipy.optimize import minimize

    if formula in ("slp", "bma"):
        if moments is None or edges is None or y is None:
            raise ValueError(
                f"fit_pool: formula={formula!r} is defined in continuous "
                f"space and requires moments=(mu, sigma), edges= and y=. "
                f"Computing it from edge CDFs alone would require "
                f"interpolating within brackets, imposing a within-bracket "
                f"shape the data does not specify."
            )
    elif moments is not None or edges is not None or y is not None:
        raise ValueError(
            f"fit_pool: moments/edges/y are only used by formula='slp' and "
            f"'bma'; got formula={formula!r}."
        )

    F = np.asarray(component_cdfs, dtype=float)
    if F.ndim != 3:
        raise ValueError(
            f"fit_pool: component_cdfs must be (k, N, B+1); got shape {F.shape}."
        )
    k, n_rows, _ = F.shape
    r_idx = np.asarray(realized_idx, dtype=int)
    if r_idx.shape[0] != n_rows:
        raise ValueError(
            f"fit_pool: realized_idx has {r_idx.shape[0]} rows but "
            f"component_cdfs has {n_rows}."
        )
    if n_rows < min_rows:
        raise ValueError(
            f"fit_pool: need ≥{min_rows} rows to fit; got {n_rows}. Below "
            f"this the log-score surface is noise-dominated and the fitted "
            f"parameters are not meaningful."
        )

    # BMA has its own estimator, EM, rather than the shared Nelder-Mead
    # objective. Weights and the common sigma are fitted jointly in one
    # mixture model rather than by maximizing a pooled log score over a
    # parameter vector.
    if formula == "bma":
        # Guaranteed by the formula check above. It is restated so the
        # narrowing is visible to a reader, and to mypy, at the point of use.
        assert moments is not None and edges is not None
        mu_a, sigma_a = moments
        w_hat, sigma_common, n_it = fit_bma(mu_a, sigma_a, np.asarray(y, float))
        # Rebuild the component CDFs at the fitted common sigma so the
        # reported dispersion and log score are those of the BMA mixture.
        sig_shared = np.full_like(np.asarray(sigma_a, dtype=float), sigma_common)
        F_bma = spread_adjusted_cdfs(mu_a, sig_shared, edges, spread_c=1.0)
        edge_cdfs = pool_cdf(F_bma, weights=w_hat, formula="tlp")
        pdf_fit = _bracket_pdf_at_realized(edge_cdfs, r_idx)
        rows = np.arange(n_rows)
        pit_fit = edge_cdfs[rows, r_idx] + 0.5 * pdf_fit
        pit_var_fit = pit_variance(pit_fit)
        # sigma_common / sigma_i is BMA's implied spread adjustment, the
        # quantity directly comparable to SLP's fitted c (paper §4.2).
        ratio = sigma_common / np.asarray(sigma_a, dtype=float)
        return PoolFit(
            formula="bma",
            weights=w_hat,
            spread_c=float(np.mean(ratio)),
            n_fit_rows=n_rows,
            mean_log_score=_mean_log_score(
                np.maximum(pdf_fit, 0.0), source="fit_pool[bma]"
            ),
            pit_var_fit=pit_var_fit,
            converged=True,
            notes={
                "sigma_common": sigma_common,
                "implied_spread_ratio_min": float(ratio.min()),
                "implied_spread_ratio_max": float(ratio.max()),
                "n_iter": n_it,
                "n_components": k,
            },
        )

    if fixed_weights is not None:
        w0 = np.asarray(fixed_weights, dtype=float)
        if w0.shape[0] != k:
            raise ValueError(
                f"fit_pool: fixed_weights has k={w0.shape[0]}, "
                f"component_cdfs has k={k}."
            )
        fit_weights = False
    else:
        w0 = np.full(k, 1.0 / k)
        fit_weights = True

    # Shape parameters, per formula. These are parametrized in log space so
    # the optimizer works unconstrained on the positive quadrant.
    if formula == "blp":
        shape0 = [0.0, 0.0]          # log α, log β  → α = β = 1 (the TLP nest)
    elif formula == "slp":
        shape0 = [0.0]               # log c         → c = 1     (the TLP nest)
    else:
        shape0 = []

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        if fit_weights:
            w_raw = np.abs(theta[:k])
            s = w_raw.sum()
            # Project onto the simplex scaled by weight_sum_max.
            w = w_raw / s * weight_sum_max if s > 0 else np.full(k, weight_sum_max / k)
            rest = theta[k:]
        else:
            w = w0
            rest = theta
        alpha = beta = spread = 1.0
        if formula == "blp":
            alpha, beta = float(np.exp(rest[0])), float(np.exp(rest[1]))
        elif formula == "slp":
            spread = float(np.exp(rest[0]))
        return w, alpha, beta, spread

    def components_at(spread: float) -> np.ndarray:
        """Component CDFs for the current parameters.

        Only SLP moves them. Its spread adjustment is applied to each
        component *before* pooling (eq. 6), so the component CDFs must be
        recomputed at every candidate c. Every other formula pools the fixed
        input CDFs.
        """
        if formula != "slp":
            return F
        assert moments is not None and edges is not None
        mu_a, sigma_a = moments
        return spread_adjusted_cdfs(mu_a, sigma_a, edges, spread_c=spread)

    def objective(theta: np.ndarray) -> float:
        w, alpha, beta, spread = unpack(theta)
        edge_cdfs = pool_cdf(
            components_at(spread), weights=w, formula=formula,
            alpha=alpha, beta=beta, link=link,
        )
        pdf = _bracket_pdf_at_realized(edge_cdfs, r_idx)
        return -_mean_log_score(np.maximum(pdf, 0.0), source="fit_pool")

    theta0 = np.concatenate(
        [w0 if fit_weights else np.array([]), np.array(shape0, dtype=float)]
    )
    if theta0.size == 0:
        raise ValueError(
            f"fit_pool: formula {formula!r} with fixed weights has no free "
            f"parameters to fit."
        )

    res = minimize(objective, theta0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9})
    w_hat, alpha_hat, beta_hat, spread_hat = unpack(res.x)

    edge_cdfs = pool_cdf(
        components_at(spread_hat), weights=w_hat, formula=formula,
        alpha=alpha_hat, beta=beta_hat, link=link,
    )
    pdf_fit = _bracket_pdf_at_realized(edge_cdfs, r_idx)
    # PIT at the realized value, interpolated within the realized bracket
    # and never randomized. A randomized PIT smears the outcome uniformly
    # across its bracket and shifts the mean. That shift was measured in this
    # repo at +0.093..+0.111 on every model over 85,250 forecasts
    # (2026-09-02). See
    # scripts/research/interpolated_pit_model_vs_market.py.
    rows = np.arange(n_rows)
    pit_fit = edge_cdfs[rows, r_idx] + 0.5 * pdf_fit

    pit_var_hold = float("nan")
    if holdout is not None:
        F_h, r_h = holdout
        F_h = np.asarray(F_h, dtype=float)
        r_h = np.asarray(r_h, dtype=int)
        edge_h = pool_cdf(
            F_h, weights=w_hat, formula=formula,
            alpha=alpha_hat, beta=beta_hat, link=link,
        )
        pdf_h = _bracket_pdf_at_realized(edge_h, r_h)
        rows_h = np.arange(F_h.shape[1])
        pit_var_hold = pit_variance(edge_h[rows_h, r_h] + 0.5 * pdf_h)

    pit_var_fit = pit_variance(pit_fit)
    return PoolFit(
        formula=formula,
        weights=w_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        spread_c=spread_hat,
        link=link,
        n_fit_rows=n_rows,
        mean_log_score=-float(res.fun),
        pit_var_fit=pit_var_fit,
        pit_var_holdout=pit_var_hold,
        leak_warning_=_check_leakage(pit_var_fit, pit_var_hold),
        converged=bool(res.success),
        notes={"optimizer_message": str(res.message), "n_components": k},
    )
