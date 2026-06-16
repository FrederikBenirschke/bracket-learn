"""The blended value objective ``L = CE − λ·EA`` shared by the value trainers.

Per binary (row, bracket) contract with model probability ``q = σ(z)``, realized
hit ``r ∈ {0,1}`` and reference price ``m``::

    CE = −[ r·log q + (1−r)·log(1−q) ]          (calibration / accuracy)
    EA =  (q − m)(r − m)                          (value vs the reference)
    L  =  CE − λ·EA

``λ ≥ 0`` is the tilt: ``λ = 0`` is a pure calibration objective; larger ``λ``
tilts toward value (capturing the reference's mispricing). **Parameterized as
``CE − λ·EA``, not ``α·CE + (1−α)·EA``**, so the CE term always supplies full,
correctly-scaled curvature — otherwise a gradient-boosted model just underfits
as the tilt grows (the EA term alone is linear and curvature-free).

Select ``λ`` by *costed* value (``score.edge_alignment_costed``), not by EA: EA
is fee-free and rises monotonically with the tilt, so it always over-tilts. See
``docs/guides/value_with_fees.md``.

Gradient / Hessian w.r.t. the raw score ``z`` (for a LightGBM custom objective)::

    ∂L/∂z   = (q − r) − λ·(r − m)·q(1−q)
    ∂²L/∂z² = q(1−q) − λ·(r − m)·q(1−q)(1−2q)      (exact)
            ≈ q(1−q)                               (Newton metric we use)

We keep **only** the CE curvature ``q(1−q)`` for the Newton step. The EA term's
true curvature ``−λ·(r − m)·q(1−q)(1−2q)`` is *indefinite* — its sign flips with
``(r − m)`` and ``(1 − 2q)`` — so including it would not give a positive-definite
metric. Dropping it (not "it's ~zero", it is not) leaves the stable PD CE
curvature; the value tilt enters only through the gradient. ``hess_floor`` keeps
the metric strictly positive as ``q → 0/1``.
"""

from __future__ import annotations

import numpy as np

#: Floor added to the Newton Hessian ``q(1−q)`` so the metric stays strictly
#: positive (and the leaf step finite) as ``q → 0/1``.
_HESS_FLOOR = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def blended_grad_hess(
    raw_score: np.ndarray,
    r: np.ndarray,
    m: np.ndarray,
    lam: float,
    *,
    hess_floor: float = _HESS_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-contract gradient and Hessian of ``L = CE − λ·EA`` w.r.t. the raw
    score. Arrays are aligned over the same (row, bracket) contracts."""
    q = _sigmoid(np.asarray(raw_score, dtype=float))
    r = np.asarray(r, dtype=float)
    m = np.asarray(m, dtype=float)
    grad = (q - r) - lam * (r - m) * q * (1.0 - q)
    hess = q * (1.0 - q) + hess_floor
    return grad, hess


def ea_scale_for_reference(reference: np.ndarray, *, eps: float = 1e-8) -> float:
    """Data-derived rescaling that makes ``lam`` mean the same thing across the
    GBM and torch engines.

    The two engines treat the EA gradient ``λ·(r − m)·q(1−q)`` differently:

    - **LightGBM** takes a Newton step, dividing the gradient by the Hessian
      ``q(1−q)``. The ``q(1−q)`` factor *cancels*, so the GBM's effective EA
      update is ``≈ λ·(r − m)``.
    - **The torch net** does plain (Adam) gradient descent — no Hessian
      division — so its EA gradient keeps the ``q(1−q)`` factor and is therefore
      suppressed by a factor ``≈ E[q(1−q)]`` relative to the GBM.

    Multiplying the torch EA term by ``1 / E[q(1−q)]`` restores parity. Evaluated
    at initialization the model predicts the reference, ``q ≈ m``, so the scale
    is ``1 / mean(m·(1−m))`` over the expanded (row, bracket) contracts — a
    quantity read directly off the reference prices, not a hand-tuned constant.

    Caveat: this matches the gradient scale at *initialization*; as ``q`` moves
    away from ``m`` during training the ``q(1−q)`` factor drifts, so the
    cross-engine equivalence of ``lam`` is approximate, not exact.
    """
    m = np.asarray(reference, dtype=float)
    if m.size == 0:
        raise ValueError("reference is empty; cannot derive an EA scale")
    if not np.all(np.isfinite(m)):
        raise ValueError("reference contains non-finite values (NaN/inf)")
    curvature = float(np.mean(m * (1.0 - m)))
    if curvature < eps:
        raise ValueError(
            f"mean reference curvature m(1-m)={curvature:.2e} is ~0 (degenerate "
            f"reference at 0/1); cannot derive an EA scale — pass ea_scale explicitly"
        )
    return 1.0 / curvature


def make_lgb_objective(reference: np.ndarray, lam: float, *, hess_floor: float = _HESS_FLOOR):
    """Build a LightGBM custom-objective closure for ``L = CE − λ·EA``.

    ``reference`` is the per-contract price ``m`` aligned to the training
    Dataset's row order (the expanded (row, bracket) order). Pass the result as
    ``params["objective"]`` to ``lightgbm.train``.
    """
    ref = np.asarray(reference, dtype=float)
    if lam < 0:
        raise ValueError(f"lam must be non-negative; got {lam}")
    if not np.all(np.isfinite(ref)):
        raise ValueError("reference contains non-finite values (NaN/inf)")

    def _objective(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        r = dataset.get_label()
        if r.shape[0] != ref.shape[0]:
            raise ValueError(
                f"reference length {ref.shape[0]} != dataset rows {r.shape[0]}; "
                f"reference is misaligned with the expanded training matrix"
            )
        return blended_grad_hess(preds, r, ref, lam, hess_floor=hess_floor)

    return _objective


def blended_loss(
    q: np.ndarray, r: np.ndarray, m: np.ndarray, lam: float, *, eps: float = 1e-12,
) -> float:
    """The scalar objective ``mean(CE) − λ·mean(EA)`` (lower = better). For the
    torch trainer and for tests; operates on probabilities ``q`` directly."""
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0 - eps)
    r = np.asarray(r, dtype=float)
    m = np.asarray(m, dtype=float)
    ce = -np.mean(r * np.log(q) + (1.0 - r) * np.log(1.0 - q))
    ea = np.mean((q - m) * (r - m))
    return float(ce - lam * ea)
