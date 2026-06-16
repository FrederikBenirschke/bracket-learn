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
    edge_alignment_costed,
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


def test_costed_zero_fee_is_sign_strategy():
    """fee=0, tau=0: every contract trades, payoff = sign(edge)·(r−m)."""
    q, _, m, r = _toy()
    c = edge_alignment_costed(q, m, r, fee=0.0, tau=0.0)
    expected = float(np.mean(np.sign(q - m) * (r - m)))
    assert c["mean_pnl"] == pytest.approx(expected)
    assert c["trade_frac"] == pytest.approx(1.0)


def test_costed_value_decreases_with_fee_at_fixed_gate():
    """At a FIXED trade gate the same trades each pay more fee, so value falls.
    (With tau tied to fee it is NOT monotone: a higher gate also drops losing
    trades — only the oracle E[(|δ|−fee)₊] is monotone in fee.)"""
    q, _, m, r = _toy()
    vals = [edge_alignment_costed(q, m, r, fee=f, tau=0.0)["mean_pnl"]
            for f in (0.0, 0.01, 0.02, 0.05)]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))


def test_costed_higher_tau_trades_less():
    q, _, m, r = _toy()
    lo = edge_alignment_costed(q, m, r, fee=0.0, tau=0.02)["trade_frac"]
    hi = edge_alignment_costed(q, m, r, fee=0.0, tau=0.10)["trade_frac"]
    assert hi < lo


def test_costed_prefers_orthogonal_forecast():
    """The orthogonal forecast (real edge) still wins once fees are charged."""
    q_acc, q_orth, m, r = _toy()
    v_acc = edge_alignment_costed(q_acc, m, r, fee=0.02)["mean_pnl"]
    v_orth = edge_alignment_costed(q_orth, m, r, fee=0.02)["mean_pnl"]
    assert v_orth > v_acc


def test_costed_loud_errors():
    q, _, m, r = _toy(n=100)
    with pytest.raises(ValueError):
        edge_alignment_costed(q, m, r, fee=-0.01)
    with pytest.raises(ValueError):
        edge_alignment_costed(q, m, r, fee=0.02, tau=-0.01)


def test_loud_errors():
    _, _, m, r = _toy(n=100)
    with pytest.raises(ValueError):
        edge_alignment(np.ones(3), np.ones(4), np.ones(4))   # shape mismatch
    with pytest.raises(ValueError):
        edge_alignment(np.array([]), np.array([]), np.array([]))  # empty
    with pytest.raises(ValueError):
        edge_alignment_corr(m, m, r)                          # zero-variance edge


def test_nonfinite_inputs_raise():
    """A NaN price (e.g. an unquoted bracket) must raise, not silently score NaN."""
    q, _, m, r = _toy(n=100)
    m_nan = m.copy()
    m_nan[0] = np.nan
    with pytest.raises(ValueError):
        edge_alignment(q, m_nan, r)
    with pytest.raises(ValueError):
        value_report(q, m_nan, r)
