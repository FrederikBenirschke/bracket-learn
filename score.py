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


def pit(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """PIT values F(y). Uniform if forecast is calibrated."""
    mu, sigma = _check_normal(dist)
    y = np.asarray(y, dtype=float)
    return _stats.norm.cdf(y, loc=mu, scale=sigma)


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
