"""Scoring: distribution-level + contract-level.

v0.1 supplies the essentials for the e2e demo:
- dist.crps_gaussian       — CRPS for Gaussian parametric backing.
- dist.log_score_gaussian  — predictive log-likelihood.
- dist.pit                 — Probability Integral Transform values for diag.
- contract.log_loss_bracket — categorical log-loss over a bracket ladder.
- contract.brier_bracket    — multi-class Brier on a bracket ladder.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats

from bracketlearn.forecast import (
    Backing,
    ContractForecast,
    DistributionForecast,
    ParametricFamily,
)


# ---------------------------------------------------------------------------
# distribution-level
# ---------------------------------------------------------------------------


def _check_normal(dist: DistributionForecast) -> tuple[np.ndarray, np.ndarray]:
    if dist.backing != Backing.PARAMETRIC or dist.family != ParametricFamily.NORMAL:
        raise NotImplementedError(
            f"score expects parametric normal; got {dist.backing}/{dist.family}"
        )
    return dist.params["mu"], dist.params["sigma"]


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
    if dist.backing != Backing.PARAMETRIC or dist.family != ParametricFamily.MIXTURE_NORMAL:
        raise NotImplementedError(
            f"log_score_mixture_normal expects parametric mixture_normal; got "
            f"{dist.backing}/{dist.family}"
        )
    y = np.asarray(y, dtype=float)
    w = dist.params["weights"]
    mus = dist.params["mus"]
    sigmas = dist.params["sigmas"]
    # pdf at y per component, then weighted sum.
    pdfs = _stats.norm.pdf(y[:, None], loc=mus, scale=sigmas)
    px = (w * pdfs).sum(axis=1)
    px = np.maximum(px, 1e-300)
    return -np.log(px)


def pit(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """PIT values F(y). Uniform if forecast is calibrated.

    Works on any backing whose dist.cdf accepts a 1-D array of length N
    aligned with the rows — we extract the diagonal of dist.cdf(y).
    """
    y = np.asarray(y, dtype=float)
    if dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.NORMAL:
        mu = dist.params["mu"]
        sigma = dist.params["sigma"]
        return _stats.norm.cdf(y, loc=mu, scale=sigma)
    # Generic path: dist.cdf(y_array) returns (N, len(y_array)); take diagonal.
    return np.diag(dist.cdf(y))


def crps_quantile(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Pinball-loss approximation of CRPS for quantile-backed dists.

    CRPS = 2 · ∫_0^1 pinball_τ(y, q_τ) dτ. We use the trapezoidal rule on
    the (taus, qvals) grid. Exact under linear-interpolated CDF.
    """
    if dist.backing != Backing.QUANTILE:
        raise NotImplementedError(f"crps_quantile expects quantile backing; got {dist.backing}")
    y = np.asarray(y, dtype=float)
    taus = dist.taus
    qvals = dist.qvals                       # (N, Q)
    # Pinball loss per (row, τ).
    diff = y[:, None] - qvals
    pinball = np.where(diff >= 0, taus[None, :] * diff, (taus[None, :] - 1.0) * diff)
    # Trapezoidal integral over τ.
    dt = np.diff(taus)
    avg = 0.5 * (pinball[:, :-1] + pinball[:, 1:])
    return 2.0 * (avg * dt[None, :]).sum(axis=1)


def log_score_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Negative log-density per row for bracket-backed dist (uniform-in-bin)."""
    if dist.backing != Backing.BRACKET:
        raise NotImplementedError(f"log_score_bracket expects bracket backing; got {dist.backing}")
    y = np.asarray(y, dtype=float)
    edges = dist.edges
    probs = dist.probs
    widths = np.diff(edges)
    density = probs / widths[None, :]
    B = probs.shape[1]
    bin_idx = np.searchsorted(edges, y, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, B - 1)
    px = density[np.arange(probs.shape[0]), bin_idx]
    px = np.maximum(px, 1e-300)
    return -np.log(px)


def crps_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """CRPS for a bracket-backed distribution.

    Computes ∫(F(z) - 1[z ≥ y])² dz under uniform-within-bin density.
    """
    if dist.backing != Backing.BRACKET:
        raise NotImplementedError(f"crps_bracket expects bracket backing; got {dist.backing}")
    y = np.asarray(y, dtype=float)
    edges = dist.edges
    probs = dist.probs                       # (N, B)
    B = probs.shape[1]
    cum = np.concatenate(
        [np.zeros((probs.shape[0], 1)), np.cumsum(probs, axis=1)], axis=1
    )                                        # (N, B+1) — F at edges
    out = np.zeros(probs.shape[0])
    for k in range(B):
        lo, hi = edges[k], edges[k + 1]
        width = hi - lo
        # F linear in [lo, hi]: F(z) = cum[:,k] + (z-lo)/width * probs[:,k].
        # 1[z≥y]: 0 for z<y, 1 for z≥y. Split by where y sits relative to bin.
        a = cum[:, k]                        # F at lo
        b = probs[:, k] / width if width > 0 else np.zeros_like(probs[:, k])
        # Cases: y >= hi → integrand = F²; y <= lo → integrand = (F-1)²;
        # lo < y < hi → split.
        # Compute Σ over rows separately:
        case_above = y >= hi
        case_below = y <= lo
        case_inside = ~case_above & ~case_below

        # 1) y >= hi: integrate F(z)² over [lo, hi].
        if case_above.any():
            # F(z) = a + b(z-lo). ∫_0^w (a+bt)² dt = a²w + abw² + b²w³/3.
            mask = case_above
            integ = a[mask] ** 2 * width + a[mask] * b[mask] * width ** 2 + b[mask] ** 2 * width ** 3 / 3.0
            out[mask] += integ

        # 2) y <= lo: integrate (F(z)-1)² over [lo, hi].
        if case_below.any():
            mask = case_below
            # (F-1)² = ((a-1)+bt)². ∫_0^w = (a-1)²w + (a-1)bw² + b²w³/3
            integ = (a[mask] - 1) ** 2 * width + (a[mask] - 1) * b[mask] * width ** 2 + b[mask] ** 2 * width ** 3 / 3.0
            out[mask] += integ

        # 3) lo < y < hi: split at y. Left of y (z < y): integrand = F².
        # Right of y (z >= y): integrand = (F - 1)².
        if case_inside.any():
            mask = case_inside
            t = y[mask] - lo                 # length on left side
            wL = t
            wR = width - t
            aL = a[mask]
            bL = b[mask]
            # ∫_0^{wL} (aL + bL u)² du
            left = aL ** 2 * wL + aL * bL * wL ** 2 + bL ** 2 * wL ** 3 / 3.0
            # ∫_{wL}^{w} (F - 1)² dz, where F at z = aL + bL(z-lo). Substitute u = z - lo:
            # ∫_{wL}^{w} (aL - 1 + bL u)² du. Compute as integral from 0 to w minus 0 to wL.
            full = (aL - 1) ** 2 * width + (aL - 1) * bL * width ** 2 + bL ** 2 * width ** 3 / 3.0
            head = (aL - 1) ** 2 * wL + (aL - 1) * bL * wL ** 2 + bL ** 2 * wL ** 3 / 3.0
            right = full - head
            out[mask] += left + right
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
    # Group rows by entity_id; assume each entity has exactly B rows in
    # contract_ids order [0..B-1].
    fair = contracts.fair_price
    if fair.size % B != 0:
        raise ValueError(f"fair_price size {fair.size} not divisible by B={B}")
    N = fair.size // B
    if y.shape[0] != N:
        raise ValueError(f"y has {y.shape[0]} entities; contracts have {N}")
    probs = fair.reshape(N, B)
    # Determine realized bin per entity.
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
