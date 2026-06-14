"""Reference-relative value metrics (Edge-Alignment family) in ``score``.

Covers the algebraic identities, the headline claim (accuracy and value can
disagree), the bracket-ladder wrappers, and the loud-error contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.forecast import ContractForecast
from bracketlearn.forecast.contract import ContractSpec
from bracketlearn.score import (
    brier_bracket,
    edge_alignment,
    edge_alignment_bracket,
    edge_alignment_corr,
    shared_bias_slope,
    value_report,
    value_report_bracket,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _toy(seed: int = 7, n: int = 40_000):
    """Two latents; the reference (market) sees only the dominant one. q_acc
    knows that dominant latent (accurate, but its edge is in priced territory);
    q_orth knows only the orthogonal latent (less accurate, edge un-priced)."""
    rng = np.random.default_rng(seed)
    s1 = rng.normal(0, 1.5, n)
    s2 = rng.normal(0, 1.5, n)
    pi = sigmoid(1.1 * s1 + 0.7 * s2)
    r = (rng.uniform(size=n) < pi).astype(float)
    m = np.clip(sigmoid(0.9 * s1), 1e-4, 1 - 1e-4)
    q_acc = np.clip(sigmoid(1.1 * s1 + rng.normal(0, 0.08, n)), 1e-4, 1 - 1e-4)
    q_orth = np.clip(sigmoid(0.7 * s2 + rng.normal(0, 0.08, n)), 1e-4, 1 - 1e-4)
    return q_acc, q_orth, m, r


def test_ea_equals_A_minus_B():
    q, _, m, r = _toy()
    rep = value_report(q, m, r)
    assert rep["EA"] == pytest.approx(rep["A_reference_mse"] - rep["B_non_orthogonality"])
    assert rep["EA"] == pytest.approx(edge_alignment(q, m, r))


def test_ea_definition_matches_mean_product():
    q, _, m, r = _toy()
    assert edge_alignment(q, m, r) == pytest.approx(np.mean((q - m) * (r - m)))


def test_no_edge_no_value():
    """q == m everywhere: zero edge, zero EA, shared-bias slope 1."""
    _, _, m, r = _toy()
    assert edge_alignment(m, m, r) == pytest.approx(0.0, abs=1e-12)
    assert shared_bias_slope(m, m, r) == pytest.approx(1.0)


def test_accuracy_and_value_can_disagree():
    """The headline: q_acc is MORE accurate (lower Brier) yet LESS valuable
    (lower EA) than q_orth, because q_orth's edge is orthogonal to the reference."""
    q_acc, q_orth, m, r = _toy()

    def brier(q):
        return float(np.mean((q - r) ** 2))

    assert brier(q_acc) < brier(q_orth)              # q_acc more accurate
    assert edge_alignment(q_acc, m, r) < edge_alignment(q_orth, m, r)  # q_orth more valuable
    # the orthogonal forecast keeps its edge correlated with the market's error
    assert edge_alignment_corr(q_orth, m, r) > edge_alignment_corr(q_acc, m, r)


def test_shared_bias_slope_detects_aligned_error():
    """A forecast that shares the reference's bias has a higher shared-bias slope
    than one whose error is orthogonal to it."""
    q_acc, q_orth, m, r = _toy()
    assert shared_bias_slope(q_acc, m, r) > shared_bias_slope(q_orth, m, r)


def _ladder(probs: np.ndarray, prov) -> ContractForecast:
    """Build a long-form ContractForecast from an (N, B) prob matrix. Only
    ``fair_price`` is read by the value metrics; the rest is valid wiring."""
    N, B = probs.shape
    return ContractForecast(
        contract_ids=np.tile(np.arange(B), N),
        entity_ids=np.repeat(np.arange(N), B),
        timestamps=np.repeat(np.arange(N, dtype=float), B),
        fair_price=probs.ravel(),
        group_id=np.repeat(np.arange(N), B),
        contract_spec=ContractSpec(kind="bracket"),
        provenance=prov,
    )


def test_bracket_wrapper_matches_flat_form(prov):
    rng = np.random.default_rng(0)
    N, B = 500, 4
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    qp = rng.dirichlet(np.ones(B), size=N)
    mp = rng.dirichlet(np.ones(B), size=N)
    y = rng.uniform(0, 4, size=N)
    # reproduce the flat (q, m, r) the wrapper builds
    onehot = np.zeros((N, B))
    bin_idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, B - 1)
    onehot[np.arange(N), bin_idx] = 1.0
    flat = edge_alignment(qp.ravel(), mp.ravel(), onehot.ravel())
    wrap = edge_alignment_bracket(_ladder(qp, prov), _ladder(mp, prov), edges, y)
    assert wrap == pytest.approx(flat)
    # array reference accepted too
    assert edge_alignment_bracket(_ladder(qp, prov), mp.ravel(), edges, y) == pytest.approx(flat)
    rep = value_report_bracket(_ladder(qp, prov), _ladder(mp, prov), edges, y)
    assert rep["EA"] == pytest.approx(flat)
    # sanity: brier_bracket still works on the same ladder
    assert brier_bracket(_ladder(qp, prov), edges, y) >= 0.0


def test_loud_errors():
    _, _, m, r = _toy(n=100)
    with pytest.raises(ValueError):
        edge_alignment(np.ones(3), np.ones(4), np.ones(4))   # shape mismatch
    with pytest.raises(ValueError):
        edge_alignment(np.array([]), np.array([]), np.array([]))  # empty
    with pytest.raises(ValueError):
        edge_alignment_corr(m, m, r)                          # zero-variance edge
