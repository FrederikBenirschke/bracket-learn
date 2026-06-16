"""Value-tilted bracket trainers (``bracketlearn.value``).

Covers the blended objective math, the fit/predict contract for both engines,
reference alignment + loud errors, and the headline behaviour: tilting ``lam``
up raises out-of-sample EA on synthetic data with exploitable mispricing.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn import WalkForward
from bracketlearn.score import edge_alignment
from bracketlearn.value import (
    BlendedBracketGBM,
    BlendedBracketNet,
    blended_grad_hess,
    blended_loss,
    ea_scale_for_reference,
)

torch = pytest.importorskip("torch", reason="torch trainer test needs torch")
lgb = pytest.importorskip("lightgbm", reason="gbm trainer test needs lightgbm")


# ----------------------------------------------------------------- objective
def test_grad_hess_reduce_to_ce_at_lam0():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    r = (rng.uniform(size=200) < 0.4).astype(float)
    m = rng.uniform(0.1, 0.9, 200)
    g, h = blended_grad_hess(z, r, m, lam=0.0)
    q = 1 / (1 + np.exp(-z))
    assert np.allclose(g, q - r)                 # pure CE gradient
    assert np.all(h > 0)                         # PD Hessian


def test_blended_loss_tilts_with_lam():
    rng = np.random.default_rng(1)
    q = rng.uniform(0.05, 0.95, 500)
    r = (rng.uniform(size=500) < q).astype(float)
    m = np.clip(q + rng.normal(0, 0.1, 500), 0.01, 0.99)
    ea = float(np.mean((q - m) * (r - m)))
    # L = CE - lam*EA, so dL/dlam = -EA
    assert blended_loss(q, r, m, 1.0) == pytest.approx(blended_loss(q, r, m, 0.0) - ea)


# ----------------------------------------------------------------- synthetic
def _synth(seed=0, E=2500, K=5):
    """Events with a dominant shared latent (priced by the market) and an
    orthogonal latent (un-priced) — so there is real mispricing to capture."""
    rng = np.random.default_rng(seed)
    Fsh, Forth = 3, 2
    X = rng.normal(size=(E, Fsh + Forth))
    A = rng.normal(size=(Fsh + Forth, K)) * 0.9
    tl = X @ A
    pi = np.exp(tl - tl.max(1, keepdims=True))
    pi /= pi.sum(1, keepdims=True)
    realized = np.array([rng.choice(K, p=pi[i]) for i in range(E)])
    Xm = X.copy()
    Xm[:, Fsh:] = 0.0                            # market sees only shared latent
    ml = 0.7 * (Xm @ A)
    m = np.exp(ml - ml.max(1, keepdims=True))
    m /= m.sum(1, keepdims=True)
    # bracket grid: K integer brackets [0,1),...,[K-1,K); y = realized index + 0.5
    edges = np.arange(K + 1, dtype=float)
    ids = np.arange(E)
    brackets_by_id = {i: edges for i in ids}
    reference_by_id = {i: m[i] for i in ids}
    y = realized + 0.5
    ts = np.arange(E, dtype=float)
    return X, y, ids, ts, brackets_by_id, reference_by_id, m, realized, K


def _oos_ea(dist, m_te, realized_te, K):
    q = np.nan_to_num(dist.probs)[:, :K]
    q = q / q.sum(1, keepdims=True)
    onehot = np.zeros_like(q)
    onehot[np.arange(len(realized_te)), realized_te] = 1.0
    return edge_alignment(q.ravel(), m_te.ravel(), onehot.ravel()) * 100


@pytest.mark.parametrize("Trainer", [BlendedBracketGBM, BlendedBracketNet])
def test_fit_predict_contract_and_shape(Trainer):
    X, y, ids, ts, bbi, rbi, m, realized, K = _synth(E=800)
    kw = dict(epochs=120) if Trainer is BlendedBracketNet else {}
    model = Trainer(lam=1.0, **kw)
    model.fit(X, y, ids=ids, brackets_by_id=bbi, reference_by_id=rbi)
    dist = model.predict_dist(X, ids=ids, timestamps=ts, brackets_by_id=bbi)
    probs = np.nan_to_num(dist.probs)[:, :K]
    assert probs.shape == (len(ids), K)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6)   # per-row renormalized


@pytest.mark.parametrize("Trainer", [BlendedBracketGBM, BlendedBracketNet])
def test_tilt_raises_oos_ea(Trainer):
    X, y, ids, ts, bbi, rbi, m, realized, K = _synth(E=2500)
    tr = ids < 1500
    te = ~tr
    kw = dict(epochs=400) if Trainer is BlendedBracketNet else {}

    def run(lam):
        # Hyperparam-only construction; grids/references passed at call time.
        # fit on train ids, predict on test ids — the trainer subsets by ids.
        model = Trainer(lam=lam, **kw)
        model.fit(X[tr], y[tr], ids=ids[tr], brackets_by_id=bbi, reference_by_id=rbi)
        dist = model.predict_dist(X[te], ids=ids[te], timestamps=ts[te],
                                  brackets_by_id=bbi)
        return _oos_ea(dist, m[te], realized[te], K)

    ea_lo = run(0.0)
    ea_hi = run(8.0)
    assert ea_hi > ea_lo                          # tilting up captures more value


# ----------------------------------------------------------------- errors
def test_reference_misalignment_raises():
    X, y, ids, ts, bbi, rbi, *_ = _synth(E=50)
    bad = {i: rbi[i][:-1] for i in ids}           # wrong length per row
    with pytest.raises(ValueError):
        BlendedBracketGBM().fit(X, y, ids=ids, brackets_by_id=bbi, reference_by_id=bad)


def test_nonfinite_reference_raises():
    """A NaN reference price must raise at fit, not silently make NaN gradients."""
    X, y, ids, ts, bbi, rbi, *_ = _synth(E=50)
    bad = {i: rbi[i].copy() for i in ids}
    bad[ids[0]][0] = np.nan
    with pytest.raises(ValueError):
        BlendedBracketGBM().fit(X, y, ids=ids, brackets_by_id=bbi, reference_by_id=bad)


def test_negative_lam_raises():
    X, y, ids, ts, bbi, rbi, *_ = _synth(E=50)
    with pytest.raises(ValueError):
        BlendedBracketGBM(lam=-1.0)               # rejected at construction


def test_predict_before_fit_raises():
    X, y, ids, ts, bbi, rbi, *_ = _synth(E=50)
    with pytest.raises(RuntimeError):
        BlendedBracketGBM().predict_dist(X, ids=ids, timestamps=ts, brackets_by_id=bbi)


@pytest.mark.parametrize("Trainer", [BlendedBracketGBM, BlendedBracketNet])
def test_missing_ids_raises_not_silently_misaligns(Trainer):
    """Forgetting ids= must raise, not auto-fill arange(N): a fabricated id would
    silently pair each row with the wrong grid / reference."""
    X, y, ids, ts, bbi, rbi, *_ = _synth(E=60)
    model = Trainer()
    with pytest.raises(TypeError, match="ids"):
        model.fit(X, y, brackets_by_id=bbi, reference_by_id=rbi)   # NO ids=


# --------------------------------------------------------- value_report_dist (#1)
def test_value_report_dist_matches_manual_and_both_y_forms():
    from bracketlearn.value import edge_alignment_dist, value_report_dist
    X, y, ids, ts, bbi, rbi, m, realized, K = _synth(E=400)
    model = BlendedBracketGBM(lam=2.0, min_child_samples=20, n_estimators=60)
    model.fit(X, y, ids=ids, brackets_by_id=bbi, reference_by_id=rbi)
    dist = model.predict_dist(X, ids=ids, timestamps=ts, brackets_by_id=bbi)
    # one-call helper == the manual flatten used elsewhere in this file
    assert edge_alignment_dist(dist, rbi, y) * 100 == pytest.approx(
        _oos_ea(dist, m, realized, K), rel=1e-9)
    # array y (row order) == dict y (by id)
    y_by_id = {int(i): float(y[j]) for j, i in enumerate(ids)}
    assert edge_alignment_dist(dist, rbi, y_by_id) == pytest.approx(
        edge_alignment_dist(dist, rbi, y))
    # full report; fee merges costed_* keys
    rep = value_report_dist(dist, rbi, y, fee=0.0175)
    assert rep["EA"] == pytest.approx(edge_alignment_dist(dist, rbi, y))
    assert {"costed_mean_pnl", "costed_trade_frac"} <= rep.keys()


def test_value_report_dist_ragged(prov):
    """Rows with different bracket counts (NaN-padded) flatten correctly."""
    from bracketlearn.forecast import DistributionForecast
    from bracketlearn.value import edge_alignment, value_report_dist
    edges = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, np.nan]])
    probs = np.array([[0.2, 0.3, 0.5], [0.6, 0.4, np.nan]])
    ids = np.array([10, 11])
    dist = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=np.array([0.0, 1.0]),
        provenance=prov)
    ref = {10: np.array([0.3, 0.3, 0.4]), 11: np.array([0.5, 0.5])}
    y = np.array([1.5, 0.5])                       # row0 → bracket1, row1 → bracket0
    q = np.array([0.2, 0.3, 0.5, 0.6, 0.4])
    mm = np.array([0.3, 0.3, 0.4, 0.5, 0.5])
    rr = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    rep = value_report_dist(dist, ref, y)
    assert rep["EA"] == pytest.approx(edge_alignment(q, mm, rr))


def test_value_report_dist_wrong_reference_length_raises(prov):
    from bracketlearn.forecast import DistributionForecast
    from bracketlearn.value import value_report_dist
    dist = DistributionForecast.from_brackets(
        edges=np.array([[0.0, 1.0, 2.0, 3.0]]), probs=np.array([[0.2, 0.3, 0.5]]),
        ids=np.array([10]), timestamps=np.array([0.0]), provenance=prov)
    with pytest.raises(ValueError):
        value_report_dist(dist, {10: np.array([0.5, 0.5])}, np.array([1.5]))


def test_negative_lam_raises_at_construction():
    """lam < 0 is rejected at construction (loud-early), not deep inside fit."""
    with pytest.raises(ValueError):
        BlendedBracketGBM(lam=-1.0)


# ----------------------------------------------------------- integration
def test_walkforward_integration_forwards_grids():
    """The trainer drops into WalkForward unchanged: construction is
    hyperparameters only, and WalkForward forwards brackets_by_id /
    reference_by_id verbatim to each deep-copied fold's fit/predict (the trainer
    subsets by the fold's ids)."""
    X, y, ids, ts, bbi, rbi, m, realized, K = _synth(E=600)
    model = BlendedBracketGBM(lam=2.0, min_child_samples=20, n_estimators=60)
    result = WalkForward(cv="kfold", n_folds=3).fit_predict(
        model, X, y, ids=ids, timestamps=ts, brackets_by_id=bbi, reference_by_id=rbi)
    fc = result.forecasts[model.name]
    probs = np.nan_to_num(fc.probs)[:, :K]
    assert probs.shape == (len(ids), K)               # OOF covers every row
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6)  # per-row renormalized


# --------------------------------------------------------- ea_scale (lam parity)
def test_ea_scale_derives_from_reference_curvature():
    """The data-derived ea_scale is exactly 1 / mean(m(1-m)) over the expanded
    reference contracts — read off the prices, not a magic constant."""
    _, _, _, _, _, rbi, *_ = _synth(E=200)
    m_exp = np.concatenate([rbi[i] for i in sorted(rbi)])
    assert ea_scale_for_reference(m_exp) == pytest.approx(1.0 / np.mean(m_exp * (1 - m_exp)))


def test_ea_scale_degenerate_reference_raises():
    with pytest.raises(ValueError):
        ea_scale_for_reference(np.zeros(10))          # m(1-m)=0 everywhere


def test_net_stores_derived_ea_scale_when_unset():
    """ea_scale=None (default) → fit derives and records ea_scale_ from the
    fit-set references; an explicit ea_scale is honored verbatim."""
    X, y, ids, ts, bbi, rbi, *_ = _synth(E=300)
    auto = BlendedBracketNet(epochs=5)
    auto.fit(X, y, ids=ids, brackets_by_id=bbi, reference_by_id=rbi)
    m_exp = np.concatenate([rbi[i] for i in ids])
    assert auto.ea_scale_ == pytest.approx(ea_scale_for_reference(m_exp))
    pinned = BlendedBracketNet(ea_scale=3.0, epochs=5)
    pinned.fit(X, y, ids=ids, brackets_by_id=bbi, reference_by_id=rbi)
    assert pinned.ea_scale_ == 3.0
