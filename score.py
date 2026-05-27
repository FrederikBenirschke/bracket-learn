"""Scoring: distribution-level + contract-level.

v0.1 supplies the essentials for the e2e demo:
- dist.crps_gaussian       — CRPS for Gaussian parametric backing.
- dist.log_score_gaussian  — predictive log-likelihood.
- dist.pit                 — Probability Integral Transform values for diag.
- contract.log_loss_bracket — categorical log-loss over a bracket ladder.
- contract.brier_bracket    — multi-class Brier on a bracket ladder.

Free functions delegate via ``isinstance`` to the matching dist subclass. New
backings should add their math as methods on the subclass; free functions
remain for downstream callers that pass a dist as first arg.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats

from bracketlearn.forecast import (
    BracketForecast,
    ContractForecast,
    DistributionForecast,
    MixtureNormalForecast,
    NormalForecast,
    QuantileForecast,
)

# ---------------------------------------------------------------------------
# distribution-level
# ---------------------------------------------------------------------------


def _check_normal(dist: DistributionForecast) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(dist, NormalForecast):
        raise NotImplementedError(
            f"score expects NormalForecast; got {type(dist).__name__}"
        )
    return dist.mu, dist.sigma


def crps_gaussian(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """CRPS for a Gaussian forecast against realized y. Returns (N,).

    Closed form: σ · [ z·(2·Φ(z) − 1) + 2·φ(z) − 1/√π ], z = (y − μ)/σ.
    """
    mu, sigma = _check_normal(dist)
    y = np.asarray(y, dtype=float)
    z = (y - mu) / sigma
    return sigma * (z * (2 * _stats.norm.cdf(z) - 1) + 2 * _stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def log_score_gaussian(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Negative log-likelihood per row (smaller = better)."""
    mu, sigma = _check_normal(dist)
    y = np.asarray(y, dtype=float)
    return -_stats.norm.logpdf(y, loc=mu, scale=sigma)


def log_score_mixture_normal(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Negative log-likelihood for a mixture-of-Normals."""
    if not isinstance(dist, MixtureNormalForecast):
        raise NotImplementedError(
            f"log_score_mixture_normal expects MixtureNormalForecast; got "
            f"{type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    w = dist.weights
    mus = dist.mus
    sigmas = dist.sigmas
    pdfs = _stats.norm.pdf(y[:, None], loc=mus, scale=sigmas)
    px = (w * pdfs).sum(axis=1)
    px = np.maximum(px, 1e-300)
    return -np.log(px)


def pit(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """PIT values F_i(y_i). Uniform-distributed if forecast is calibrated."""
    return dist.cdf_at(np.asarray(y, dtype=float))


def crps_quantile(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Pinball-loss approximation of CRPS for quantile-backed dists.

    CRPS = 2 · ∫_0^1 pinball_τ(y, q_τ) dτ. We use the trapezoidal rule on
    the (taus, qvals) grid. Exact under linear-interpolated CDF.
    """
    if not isinstance(dist, QuantileForecast):
        raise NotImplementedError(
            f"crps_quantile expects QuantileForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    taus = dist.taus
    qvals = dist.qvals                       # (N, Q)
    diff = y[:, None] - qvals
    pinball = np.where(diff >= 0, taus[None, :] * diff, (taus[None, :] - 1.0) * diff)
    dt = np.diff(taus)
    avg = 0.5 * (pinball[:, :-1] + pinball[:, 1:])
    return 2.0 * (avg * dt[None, :]).sum(axis=1)


def log_score_quantile(
    dist: DistributionForecast, y: np.ndarray,
) -> np.ndarray:
    """Negative log-density per row from a quantile-backed dist.

    Treats the CDF as piecewise-linear between the stored quantiles, so
    the density is piecewise-constant: between τ_i and τ_{i+1}, the
    density at any y in [q_i, q_{i+1}] is (τ_{i+1} - τ_i) / (q_{i+1} - q_i).

    Below q_0 and above q_{Q-1}, density falls back to a tail-rule
    estimate: we extend the local density of the outermost bin (a
    cautious choice that mirrors ``tail_policy="clip"``).
    """
    if not isinstance(dist, QuantileForecast):
        raise NotImplementedError(
            f"log_score_quantile expects QuantileForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    taus = dist.taus
    qvals = dist.qvals
    N, Q = qvals.shape
    dq = np.diff(qvals, axis=1)
    dt = np.diff(taus)
    safe_dq = np.where(dq > 1e-12, dq, 1e-12)
    density_bins = dt[None, :] / safe_dq

    out = np.empty(N, dtype=float)
    for r in range(N):
        q_r = qvals[r]
        y_r = y[r]
        if y_r <= q_r[0]:
            out[r] = density_bins[r, 0]
        elif y_r >= q_r[-1]:
            out[r] = density_bins[r, -1]
        else:
            i = int(np.searchsorted(q_r, y_r, side="right") - 1)
            i = min(max(i, 0), Q - 2)
            out[r] = density_bins[r, i]
    out = np.maximum(out, 1e-300)
    return -np.log(out)


def crps_mixture_normal(
    dist: DistributionForecast,
    y: np.ndarray,
    *,
    n_samples: int = 2000,
    random_state: int | None = 0,
) -> np.ndarray:
    """Monte-Carlo CRPS for a mixture-of-normals parametric backing.

    Uses the energy form:

        CRPS(F, y) = E|X - y| - 0.5 · E|X - X'|

    where X, X' are i.i.d. draws from the predictive mixture.
    """
    if not isinstance(dist, MixtureNormalForecast):
        raise NotImplementedError(
            f"crps_mixture_normal expects MixtureNormalForecast; got "
            f"{type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    weights = dist.weights
    mus = dist.mus
    sigmas = dist.sigmas
    N, K = weights.shape
    rng = np.random.default_rng(random_state)
    cumw = np.cumsum(weights, axis=1)
    u = rng.random((N, n_samples))
    comp = (u[:, :, None] >= cumw[:, None, :]).sum(axis=2)
    comp = np.clip(comp, 0, K - 1)
    rows = np.arange(N)[:, None]
    mu_s = mus[rows, comp]
    sig_s = sigmas[rows, comp]
    z = rng.standard_normal((N, n_samples))
    x = mu_s + sig_s * z
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    x_prime = np.roll(x, 1, axis=1)
    term2 = 0.5 * np.abs(x - x_prime).mean(axis=1)
    return term1 - term2


def to_point(
    dist: DistributionForecast,
    *,
    how: str = "mean",
) -> np.ndarray:
    """Collapse any ``DistributionForecast`` to a 1-D point forecast.

    Thin wrapper over ``dist.to_point(how=how)`` — each subclass implements
    the math. Kept as a free function for callers that pass a dist as first
    positional arg.
    """
    return dist.to_point(how=how)


def log_score_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Negative log-density per row for bracket-backed dist (uniform-in-bin).

    Per-row edges supported: each row uses its own bracket grid via
    ``BracketForecast.realized_bin`` (NaN-padded tails are ignored).
    """
    if not isinstance(dist, BracketForecast):
        raise NotImplementedError(
            f"log_score_bracket expects BracketForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    edges = dist.edges
    probs = dist.probs
    widths = np.diff(edges, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(widths > 0, probs / widths, 0.0)
    bin_idx = dist.realized_bin(y)
    px = density[np.arange(density.shape[0]), bin_idx]
    px = np.maximum(px, 1e-300)
    return -np.log(px)


def crps_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """CRPS for a bracket-backed distribution.

    Computes ∫(F(z) - 1[z ≥ y])² dz under uniform-within-bin density.
    Per-row edges supported (NaN-padded tail columns are skipped).
    """
    if not isinstance(dist, BracketForecast):
        raise NotImplementedError(
            f"crps_bracket expects BracketForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    edges = dist.edges
    probs_clean = np.nan_to_num(dist.probs, nan=0.0)
    N, B_max = probs_clean.shape
    cum = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(probs_clean, axis=1)], axis=1
    )
    B_per_row = (~np.isnan(dist.probs)).sum(axis=1).astype(int)
    out = np.zeros(N)
    for k in range(B_max):
        active = B_per_row > k
        if not active.any():
            continue
        lo = edges[:, k]
        hi = edges[:, k + 1]
        width = hi - lo
        a = cum[:, k]
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.where(width > 0, probs_clean[:, k] / width, 0.0)
        case_above = active & (y >= hi)
        case_below = active & (y <= lo)
        case_inside = active & ~case_above & ~case_below

        if case_above.any():
            m = case_above
            w = width[m]
            integ = a[m] ** 2 * w + a[m] * b[m] * w ** 2 + b[m] ** 2 * w ** 3 / 3.0
            out[m] += integ

        if case_below.any():
            m = case_below
            w = width[m]
            integ = (a[m] - 1) ** 2 * w + (a[m] - 1) * b[m] * w ** 2 + b[m] ** 2 * w ** 3 / 3.0
            out[m] += integ

        if case_inside.any():
            m = case_inside
            w = width[m]
            t = y[m] - lo[m]
            wL = t
            aL = a[m]
            bL = b[m]
            left = aL ** 2 * wL + aL * bL * wL ** 2 + bL ** 2 * wL ** 3 / 3.0
            full = (aL - 1) ** 2 * w + (aL - 1) * bL * w ** 2 + bL ** 2 * w ** 3 / 3.0
            head = (aL - 1) ** 2 * wL + (aL - 1) * bL * wL ** 2 + bL ** 2 * wL ** 3 / 3.0
            right = full - head
            out[m] += left + right
    return out


# ---------------------------------------------------------------------------
# contract-level (bracket ladder)
# ---------------------------------------------------------------------------


def log_loss_bracket(
    contracts: ContractForecast,
    edges: np.ndarray,
    y: np.ndarray,
    *,
    entity_order: np.ndarray | None = None,
) -> float:
    """Categorical log-loss over a bracket ladder.

    contracts is a long-form ContractForecast with one row per (entity, bin).
    edges is the (B+1,) ladder.
    y is the realized value per entity.
    """
    edges = np.asarray(edges, dtype=float)
    y = np.asarray(y, dtype=float)
    B = edges.shape[0] - 1
    fair = contracts.fair_price
    if fair.size % B != 0:
        raise ValueError(f"fair_price size {fair.size} not divisible by B={B}")
    N = fair.size // B
    if y.shape[0] != N:
        raise ValueError(f"y has {y.shape[0]} entities; contracts have {N}")
    probs = fair.reshape(N, B)
    bin_idx = np.searchsorted(edges, y, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, B - 1)
    p_realized = probs[np.arange(N), bin_idx]
    p_realized = np.clip(p_realized, 1e-12, 1.0)
    return float(-np.log(p_realized).mean())


def brier_bracket(
    contracts: ContractForecast,
    edges: np.ndarray,
    y: np.ndarray,
) -> float:
    """Multi-class Brier: Σ_b (p_b - 1[y in bin b])²."""
    edges = np.asarray(edges, dtype=float)
    y = np.asarray(y, dtype=float)
    B = edges.shape[0] - 1
    fair = contracts.fair_price
    N = fair.size // B
    probs = fair.reshape(N, B)
    onehot = np.zeros_like(probs)
    bin_idx = np.searchsorted(edges, y, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, B - 1)
    onehot[np.arange(N), bin_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())
