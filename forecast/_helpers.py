"""Shared array / numeric helpers used across DistributionForecast subclasses.

Pure functions — no class state, no behaviour changes from the v0.2
inlined versions. Lives below the subclass modules in the dependency
graph: ``_quantile_via_brentq`` does isinstance checks against
``BracketForecast`` and ``MixtureNormalForecast`` via local imports to
keep this module circular-free.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# bracket-probs normalisation
# ---------------------------------------------------------------------------


def normalize_bracket_probs(
    raw: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    """Normalise raw per-bracket weights into a valid distribution. See v0.2 docstring."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim not in (1, 2):
        raise ValueError(
            f"{source}: normalize_bracket_probs expects 1-D or 2-D "
            f"input; got shape {raw.shape}."
        )
    if np.any(raw < 0):
        raise ValueError(
            f"{source}: normalize_bracket_probs received negative "
            f"weights. Refusing to clip silently — upstream produced "
            f"invalid data."
        )
    if raw.ndim == 1:
        s = float(raw.sum())
        if s <= 0:
            raise ValueError(
                f"{source}: normalize_bracket_probs got total weight "
                f"{s:.6g} ≤ 0 across K={raw.shape[0]} brackets. Refusing "
                f"to fabricate a uniform distribution."
            )
        return raw / s
    row_sum = raw.sum(axis=1, keepdims=True)
    if np.any(row_sum.ravel() <= 0):
        bad = np.where(row_sum.ravel() <= 0)[0]
        raise ValueError(
            f"{source}: normalize_bracket_probs got {bad.size} row(s) "
            f"with total weight ≤ 0. First offending row indices: "
            f"{bad[:5].tolist()}."
        )
    return raw / row_sum


def bracket_probs_from_cdf_at_edges(
    cdf_at_edges: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    probs = np.diff(cdf_at_edges, axis=1)
    probs = np.clip(probs, 0.0, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    if np.any(row_sum.ravel() <= 0):
        n_bad = int((row_sum.ravel() <= 0).sum())
        raise ValueError(
            f"{source}: {n_bad}/{probs.shape[0]} rows produced zero total "
            f"bracket mass. Refusing to substitute a uniform distribution."
        )
    return probs / row_sum


# ---------------------------------------------------------------------------
# Tail-policy helper.
# ---------------------------------------------------------------------------


def _resolve_tail_kinds(tail_policy) -> tuple[str, str]:
    if tail_policy is None:
        raise NotImplementedError("quantile-backed cdf requires a TailPolicy")
    return tail_policy.left.kind, tail_policy.right.kind


# ---------------------------------------------------------------------------
# Per-row edges normalisation (1-D shared / 2-D dense / ragged sequence
# → dense NaN-padded (N, B_max+1)).
# ---------------------------------------------------------------------------


def _to_dense_2d(edges_per_row, *, n_rows: int) -> np.ndarray:
    """Normalise heterogeneous edge inputs to a dense (N, B_max+1) array
    with NaN padding for ragged rows.

    Accepts: 1-D shared (B+1,), 2-D dense (N, B+1), or a length-N
    sequence of 1-D arrays.
    """
    if isinstance(edges_per_row, np.ndarray):
        if edges_per_row.ndim == 1:
            if edges_per_row.shape[0] < 2:
                raise ValueError(
                    f"shared edges must have ≥2 entries; got {edges_per_row.shape}"
                )
            return np.broadcast_to(
                edges_per_row[None, :].astype(float),
                (n_rows, edges_per_row.shape[0]),
            ).copy()
        if edges_per_row.ndim == 2:
            if edges_per_row.shape[0] != n_rows:
                raise ValueError(
                    f"edges_per_row N={edges_per_row.shape[0]} != dist N={n_rows}"
                )
            return edges_per_row.astype(float)
        raise ValueError(f"edges_per_row ndarray must be 1-D or 2-D; got shape {edges_per_row.shape}")
    # Sequence path.
    rows = list(edges_per_row)
    if len(rows) != n_rows:
        raise ValueError(
            f"edges_per_row has length {len(rows)}; dist has {n_rows} rows"
        )
    rows_arr = [np.asarray(r, dtype=float) for r in rows]
    if any(r.ndim != 1 or r.shape[0] < 2 for r in rows_arr):
        bad = [i for i, r in enumerate(rows_arr) if r.ndim != 1 or r.shape[0] < 2]
        raise ValueError(
            f"edges_per_row entries must be 1-D with ≥2 entries; bad row(s): {bad[:5]}"
        )
    B_max1 = max(r.shape[0] for r in rows_arr)
    out = np.full((n_rows, B_max1), np.nan, dtype=float)
    for i, r in enumerate(rows_arr):
        out[i, : r.shape[0]] = r
    return out


def _clip_tiny_negatives(probs: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Clip small numerical-noise negative entries in a probs array to 0.
    Larger negatives raise — they indicate a real upstream bug rather
    than rounding."""
    if np.any(probs < -atol):
        worst = float(np.nanmin(probs))
        raise ValueError(
            f"integrate: probs contain negative entries beyond tolerance "
            f"(min={worst:.6g}); upstream cdf_at_grid is non-monotone."
        )
    return np.where(probs < 0, 0.0, probs)


# ---------------------------------------------------------------------------
# CDF inversion via brentq — used by mixture / bracket for median.
# Local imports avoid a circular at module load.
# ---------------------------------------------------------------------------


def _quantile_via_brentq(dist, q: float) -> np.ndarray:
    """Numerical CDF inversion at level ``q`` per row.

    Used for medians of dists without closed-form medians (mixture, bracket).
    Bracket bounds are picked from the dist's own structure when possible.
    """
    from scipy.optimize import brentq

    from bracketlearn.forecast.bracket import BracketForecast
    from bracketlearn.forecast.parametric import MixtureNormalForecast

    n = dist.ids.shape[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        def f(x, _i=i):
            return float(dist.cdf(np.array([x]))[_i, 0]) - q
        if isinstance(dist, BracketForecast):
            e_row = dist.edges[i]
            finite = e_row[~np.isnan(e_row)]
            lo, hi = float(finite[0]), float(finite[-1])
        elif isinstance(dist, MixtureNormalForecast):
            mus = dist.mus[i]
            sigmas = dist.sigmas[i]
            lo = float((mus - 6 * sigmas).min())
            hi = float((mus + 6 * sigmas).max())
        else:
            lo, hi = -1e6, 1e6
        out[i] = brentq(f, lo, hi, xtol=1e-6)
    return out
