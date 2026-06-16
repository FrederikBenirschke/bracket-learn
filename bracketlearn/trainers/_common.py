"""Shared utilities for bracketlearn trainer subpackage."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np


def _estimator_accepts_sample_weight(estimator: Any) -> bool:
    """Inspect fit signature for a sample_weight parameter.

    Replaces the v0.1 bare ``except TypeError`` pattern (any TypeError
    raised *inside* fit silently dropped the weights). Now we either pass
    weights or skip them by explicit signature check.
    """
    try:
        sig = inspect.signature(estimator.fit)
    except (ValueError, TypeError):
        return False
    return "sample_weight" in sig.parameters


def _weighted_lstsq2(A: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> tuple[float, float]:
    """Weighted least squares for 2-column design matrices. Returns (a, b)."""
    if w is None:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    else:
        sw = np.sqrt(np.asarray(w, dtype=float))
        sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return float(sol[0]), float(sol[1])


def _validate_brackets_by_id(
    brackets_by_id: dict[Any, np.ndarray],
    *,
    owner: str,
) -> None:
    """Validate an id → 1-D edge-array dict (≥2 bins, strictly increasing).

    Used by every trainer that takes per-row bracket edges. ``owner`` is the
    class name surfaced in error messages.
    """
    if not isinstance(brackets_by_id, dict) or not brackets_by_id:
        raise ValueError(
            f"{owner} needs a non-empty brackets_by_id dict "
            "(id → 1-D edge array)"
        )
    for k, e in brackets_by_id.items():
        e_arr = np.asarray(e, dtype=float)
        if e_arr.ndim != 1 or e_arr.size < 3:
            raise ValueError(
                f"brackets_by_id[{k!r}]: ladder must have ≥2 bins "
                f"(≥3 edges); got shape {e_arr.shape}"
            )
        if np.any(np.diff(e_arr) <= 0):
            raise ValueError(
                f"brackets_by_id[{k!r}]: edges must be strictly increasing"
            )


def _augment_with_bracket_bounds(
    X: np.ndarray,
    ids: np.ndarray,
    brackets_by_id: dict[Any, np.ndarray],
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build per-bracket augmented design matrix [X_i, lo_b, hi_b].

    Returns ``(X_aug, offsets, per_row_edges)`` where
    ``offsets[i] : offsets[i+1]`` covers row i's B_i augmented rows, and
    ``per_row_edges[i]`` is the row's edge array (length B_i + 1).

    Used by ``bracketlearn.transformers.BracketExpander``. ``owner``
    surfaces in the missing-id KeyError.
    """
    N = X.shape[0]
    per_row_edges: list[np.ndarray] = []
    missing: list[Any] = []
    for k in ids:
        try:
            per_row_edges.append(
                np.asarray(brackets_by_id[k], dtype=float),
            )
        except KeyError:
            missing.append(k)
    if missing:
        raise KeyError(
            f"{owner}: brackets_by_id missing {len(missing)} "
            f"id(s); first: {missing[:3]}"
        )
    B_per_row = np.array([e.size - 1 for e in per_row_edges], dtype=int)
    offsets = np.concatenate([[0], np.cumsum(B_per_row)])
    M = int(offsets[-1])
    n_feat = X.shape[1]
    X_aug = np.empty((M, n_feat + 2), dtype=float)
    for i in range(N):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        X_aug[sl, :n_feat] = X[i]
        e_i = per_row_edges[i]
        X_aug[sl, n_feat] = e_i[:-1]      # lo
        X_aug[sl, n_feat + 1] = e_i[1:]   # hi
    return X_aug, offsets, per_row_edges
