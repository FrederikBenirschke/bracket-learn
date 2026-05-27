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
    """PIT values F_i(y_i). Uniform-distributed if forecast is calibrated.

    Per-row CDF lookup via ``dist.cdf_at(y)``. Costs O(N), not O(N²) —
    earlier versions built ``dist.cdf(y)`` as an (N, N) matrix and took
    the diagonal (800 MB at N=10k).
    """
    return dist.cdf_at(np.asarray(y, dtype=float))


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


def log_score_quantile(
    dist: DistributionForecast, y: np.ndarray,
) -> np.ndarray:
    """Negative log-density per row from a quantile-backed dist.

    Treats the CDF as piecewise-linear between the stored quantiles, so
    the density is piecewise-constant: between τ_i and τ_{i+1}, the
    density at any y in [q_i, q_{i+1}] is (τ_{i+1} - τ_i) / (q_{i+1} - q_i).

    Below q_0 and above q_{Q-1}, density falls back to a tail-rule
    estimate: we extend the local density of the outermost bin (a
    cautious choice that mirrors ``tail_policy="clip"``). A `gpd`
    or `gaussian_match` tail would give heavier tails, but those rules
    aren't in v0.2 — see the README "Not yet" list.
    """
    if dist.backing != Backing.QUANTILE:
        raise NotImplementedError(
            f"log_score_quantile expects quantile backing; got {dist.backing}"
        )
    y = np.asarray(y, dtype=float)
    taus = dist.taus                       # (Q,)
    qvals = dist.qvals                     # (N, Q)
    N, Q = qvals.shape
    # Per-bin slope: (τ_{i+1} - τ_i) / (q_{i+1} - q_i). Guard zero width.
    dq = np.diff(qvals, axis=1)            # (N, Q-1)
    dt = np.diff(taus)                     # (Q-1,)
    safe_dq = np.where(dq > 1e-12, dq, 1e-12)
    density_bins = dt[None, :] / safe_dq   # (N, Q-1)

    # Locate y in qvals row-by-row (vectorised via argmax of a comparison).
    # For each row, find i such that qvals[i] <= y < qvals[i+1]; outside
    # the support, use the nearest interior bin's density (clip rule).
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

    where X, X' are i.i.d. draws from the predictive mixture. Sampling
    is per-row vectorised; ``n_samples`` controls both the MAE term
    and the self-distance term (X' = roll(X, 1) for a cheap antithetic
    estimate of the second expectation).

    A closed form exists for mixtures of normals but is O(K²) per row
    with quadratic numerical headaches; for the moderate K bracketlearn
    actually ships (<10 vendors), MC at n=2000 lands within ~1 % of the
    true value at ms-per-row cost.
    """
    if dist.backing != Backing.PARAMETRIC or dist.family != ParametricFamily.MIXTURE_NORMAL:
        raise NotImplementedError(
            f"crps_mixture_normal expects mixture_normal backing; got "
            f"{dist.backing}/{dist.family}"
        )
    y = np.asarray(y, dtype=float)
    weights = dist.params["weights"]        # (N, K)
    mus = dist.params["mus"]                # (N, K)
    sigmas = dist.params["sigmas"]          # (N, K)
    N, K = weights.shape
    rng = np.random.default_rng(random_state)
    # Sample mixture indices: (N, n_samples).
    cumw = np.cumsum(weights, axis=1)
    u = rng.random((N, n_samples))
    comp = (u[:, :, None] >= cumw[:, None, :]).sum(axis=2)
    comp = np.clip(comp, 0, K - 1)
    rows = np.arange(N)[:, None]
    mu_s = mus[rows, comp]
    sig_s = sigmas[rows, comp]
    # Draw the samples.
    z = rng.standard_normal((N, n_samples))
    x = mu_s + sig_s * z                    # (N, n_samples)
    # E|X - y|.
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    # 0.5 · E|X - X'|. Antithetic via roll for variance reduction.
    x_prime = np.roll(x, 1, axis=1)
    term2 = 0.5 * np.abs(x - x_prime).mean(axis=1)
    return term1 - term2


def to_point(
    dist: DistributionForecast,
    *,
    how: str = "mean",
) -> np.ndarray:
    """Collapse any ``DistributionForecast`` to a 1-D point forecast.

    Args:
        dist: any backing.
        how: one of ``"mean"``, ``"median"``, ``"mode"``.

    The point forecast is what classical-ML reviewers want to see —
    feed the output to ``sklearn.metrics.mean_squared_error`` or
    ``mean_absolute_error`` and compare against an ``Ridge`` or
    ``LGBMRegressor`` benchmark.

    Mode definitions:
        - parametric normal / mixture: μ (matches mean for single normal;
          most-likely component for mixture).
        - bracket: midpoint of the highest-probability bin.
        - quantile: the stored quantile whose density is highest, taken
          at the bin midpoint (rough — quantile dists don't carry a true
          mode without further interpolation).
    """
    if how not in ("mean", "median", "mode"):
        raise ValueError(f"how={how!r} not in 'mean'/'median'/'mode'")
    if dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.NORMAL:
        mu = dist.params["mu"]
        if how == "mode":
            return np.asarray(mu, dtype=float)
        if how == "mean" or how == "median":
            return np.asarray(mu, dtype=float)
    if dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.MIXTURE_NORMAL:
        weights = dist.params["weights"]
        mus = dist.params["mus"]
        if how == "mean":
            return (weights * mus).sum(axis=1)
        if how == "mode":
            best = np.argmax(weights, axis=1)
            return mus[np.arange(mus.shape[0]), best]
        # median for mixture: invert the CDF row-by-row.
        return _quantile_at(dist, 0.5)
    if dist.backing == Backing.BRACKET:
        edges = dist.edges                              # (N, B_max+1) per-row
        probs = dist.probs                              # (N, B_max)
        mids = 0.5 * (edges[:, :-1] + edges[:, 1:])     # (N, B_max), NaN where padded
        if how == "mean":
            return np.nansum(probs * mids, axis=1)
        if how == "mode":
            # NaN-tolerant argmax: use np.nan_to_num so padded bins (NaN
            # probs) aren't picked.
            p_clean = np.nan_to_num(probs, nan=-np.inf)
            top = np.argmax(p_clean, axis=1)
            rows = np.arange(probs.shape[0])
            return mids[rows, top]
        return _quantile_at(dist, 0.5)
    if dist.backing == Backing.QUANTILE:
        taus = dist.taus
        qvals = dist.qvals
        if how == "median":
            j = int(np.argmin(np.abs(taus - 0.5)))
            return qvals[:, j]
        if how == "mean":
            # Trapezoidal estimate using the stored (τ, q) grid.
            # E[Y] = ∫ q dτ ≈ trapz(qvals, taus) over the [τ_0, τ_{Q-1}]
            # interval. Extends to [0, 1] via the clip tail rule.
            dt = np.diff(taus)
            avg = 0.5 * (qvals[:, :-1] + qvals[:, 1:])
            inner = (avg * dt[None, :]).sum(axis=1)
            # Mass below τ_0 and above τ_{Q-1} held at the boundary q values.
            lower = qvals[:, 0] * taus[0]
            upper = qvals[:, -1] * (1.0 - taus[-1])
            return lower + inner + upper
        # mode: highest-density bin → midpoint of (q_i, q_{i+1}).
        dq = np.diff(qvals, axis=1)
        dt = np.diff(taus)
        density = dt[None, :] / np.where(dq > 1e-12, dq, 1e-12)
        top = np.argmax(density, axis=1)
        rows = np.arange(qvals.shape[0])
        return 0.5 * (qvals[rows, top] + qvals[rows, top + 1])
    raise NotImplementedError(f"to_point: backing {dist.backing} not handled")


def _quantile_at(dist: DistributionForecast, q: float) -> np.ndarray:
    """Numerical CDF inversion at level ``q``. Used for medians of
    distributions without closed-form medians (mixture, bracket)."""
    from scipy.optimize import brentq

    n = dist.params["mu"].shape[0] if dist.backing == Backing.PARAMETRIC else dist.probs.shape[0]
    out = np.empty(n, dtype=float)
    # Build a per-row CDF callable using the dist's vectorised cdf().
    for i in range(n):
        # cdf() expects a 1-D array of points; bind i with a default arg so
        # the closure doesn't re-capture the loop variable.
        def f(x, _i=i):
            return float(dist.cdf(np.array([x]))[_i, 0]) - q
        # Probe brackets — start with the dist's own quantile-like landmarks.
        if dist.backing == Backing.BRACKET:
            # Per-row: use this row's finite edge prefix to bracket brentq.
            e_row = dist.edges[i]
            finite = e_row[~np.isnan(e_row)]
            lo, hi = float(finite[0]), float(finite[-1])
        elif dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.MIXTURE_NORMAL:
            mus = dist.params["mus"][i]
            sigmas = dist.params["sigmas"][i]
            lo = float((mus - 6 * sigmas).min())
            hi = float((mus + 6 * sigmas).max())
        else:
            lo, hi = -1e6, 1e6
        out[i] = brentq(f, lo, hi, xtol=1e-6)
    return out


def log_score_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Negative log-density per row for bracket-backed dist (uniform-in-bin).

    Per-row edges supported: each row uses its own bracket grid via
    ``BracketForecast.realized_bin`` (NaN-padded tails are ignored).
    """
    if dist.backing != Backing.BRACKET:
        raise NotImplementedError(f"log_score_bracket expects bracket backing; got {dist.backing}")
    y = np.asarray(y, dtype=float)
    edges = dist.edges                          # (N, B+1) per-row
    probs = dist.probs                          # (N, B)   per-row
    widths = np.diff(edges, axis=1)             # (N, B), NaN where padded
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(widths > 0, probs / widths, 0.0)
    bin_idx = dist.realized_bin(y)              # (N,) per-row valid bin
    px = density[np.arange(density.shape[0]), bin_idx]
    px = np.maximum(px, 1e-300)
    return -np.log(px)


def crps_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """CRPS for a bracket-backed distribution.

    Computes ∫(F(z) - 1[z ≥ y])² dz under uniform-within-bin density.
    Per-row edges supported (NaN-padded tail columns are skipped).
    """
    if dist.backing != Backing.BRACKET:
        raise NotImplementedError(f"crps_bracket expects bracket backing; got {dist.backing}")
    y = np.asarray(y, dtype=float)
    edges = dist.edges                       # (N, B_max+1)
    probs_clean = np.nan_to_num(dist.probs, nan=0.0)  # (N, B_max)
    N, B_max = probs_clean.shape
    cum = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(probs_clean, axis=1)], axis=1
    )                                        # (N, B_max+1) — F at edges
    # Per-row valid bin count B_i.
    B_per_row = (~np.isnan(dist.probs)).sum(axis=1).astype(int)
    out = np.zeros(N)
    for k in range(B_max):
        # Active rows for bin k: rows where B_i > k.
        active = B_per_row > k
        if not active.any():
            continue
        lo = edges[:, k]                     # (N,)
        hi = edges[:, k + 1]                 # (N,)
        width = hi - lo                      # (N,)
        a = cum[:, k]                        # (N,)
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
