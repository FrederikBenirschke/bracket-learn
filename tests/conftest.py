"""Shared pytest fixtures.

Centralises the boilerplate ProvenanceMeta + ids + timestamps used by
constructor tests. Keeping it here means individual tests stay focused on
the invariants under test rather than wiring.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from bracketlearn.forecast import ProvenanceMeta


@pytest.fixture
def prov() -> ProvenanceMeta:
    return ProvenanceMeta(
        forecaster_name="test",
        forecaster_version="0.1",
        fit_window=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
        fold_idx=None,
        calibration_set_hash=None,
        random_seed=0,
        code_sha="test",
        feature_matrix_hash="test",
        created_at=datetime(2024, 1, 1),
    )


@pytest.fixture
def ids_ts():
    """Return a factory: ids_ts(N) → (ids, timestamps) of length N."""
    def _make(n: int):
        return np.arange(n), np.arange(n, dtype=float)
    return _make


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
