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
