"""Scoring: distribution-level + contract-level.

v0.1 supplies the essentials for the e2e demo:
- dist.crps_gaussian       — CRPS for Gaussian parametric backing.
- dist.log_score_gaussian  — predictive log-likelihood.
- dist.pit                 — Probability Integral Transform values for diag.
- contract.log_loss_bracket — categorical log-loss over a bracket ladder.
- contract.brier_bracket    — multi-class Brier on a bracket ladder.

Free functions delegate via ``isinstance`` to the matching dist subclass. New
backings should add their math as methods on the subclass; free functions
remain for downstream callers that pass a dist as first arg.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats as _stats

from bracketlearn.forecast import (
    BracketForecast,
    ContractForecast,
    DistributionForecast,
    MixtureNormalForecast,
    NormalForecast,
    QuantileForecast,
)

# ---------------------------------------------------------------------------
# distribution-level
# ---------------------------------------------------------------------------


def _check_normal(dist: DistributionForecast) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(dist, NormalForecast):
        raise NotImplementedError(
            f"score expects NormalForecast; got {type(dist).__name__}"
        )
    return dist.mu, dist.sigma


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
    if not isinstance(dist, MixtureNormalForecast):
        raise NotImplementedError(
            f"log_score_mixture_normal expects MixtureNormalForecast; got "
            f"{type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    w = dist.weights
    mus = dist.mus
    sigmas = dist.sigmas
    pdfs = _stats.norm.pdf(y[:, None], loc=mus, scale=sigmas)
    px = (w * pdfs).sum(axis=1)
    px = np.maximum(px, 1e-300)
    return -np.log(px)


def pit(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """PIT values F_i(y_i). Uniform-distributed if forecast is calibrated."""
    return dist.cdf_at(np.asarray(y, dtype=float))


def crps_quantile(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Pinball-loss approximation of CRPS for quantile-backed dists.

    CRPS = 2 · ∫_0^1 pinball_τ(y, q_τ) dτ. We use the trapezoidal rule on
    the (taus, qvals) grid. Exact under linear-interpolated CDF.
    """
    if not isinstance(dist, QuantileForecast):
        raise NotImplementedError(
            f"crps_quantile expects QuantileForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    taus = dist.taus
    qvals = dist.qvals                       # (N, Q)
    diff = y[:, None] - qvals
    pinball = np.where(diff >= 0, taus[None, :] * diff, (taus[None, :] - 1.0) * diff)
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
    cautious choice that mirrors ``tail_policy="clip"``).
    """
    if not isinstance(dist, QuantileForecast):
        raise NotImplementedError(
            f"log_score_quantile expects QuantileForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    taus = dist.taus
    qvals = dist.qvals
    N, Q = qvals.shape
    dq = np.diff(qvals, axis=1)
    dt = np.diff(taus)
    safe_dq = np.where(dq > 1e-12, dq, 1e-12)
    density_bins = dt[None, :] / safe_dq

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

    where X, X' are i.i.d. draws from the predictive mixture.
    """
    if not isinstance(dist, MixtureNormalForecast):
        raise NotImplementedError(
            f"crps_mixture_normal expects MixtureNormalForecast; got "
            f"{type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    weights = dist.weights
    mus = dist.mus
    sigmas = dist.sigmas
    N, K = weights.shape
    rng = np.random.default_rng(random_state)
    cumw = np.cumsum(weights, axis=1)
    u = rng.random((N, n_samples))
    comp = (u[:, :, None] >= cumw[:, None, :]).sum(axis=2)
    comp = np.clip(comp, 0, K - 1)
    rows = np.arange(N)[:, None]
    mu_s = mus[rows, comp]
    sig_s = sigmas[rows, comp]
    z = rng.standard_normal((N, n_samples))
    x = mu_s + sig_s * z
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    x_prime = np.roll(x, 1, axis=1)
    term2 = 0.5 * np.abs(x - x_prime).mean(axis=1)
    return term1 - term2


def to_point(
    dist: DistributionForecast,
    *,
    how: str = "mean",
) -> np.ndarray:
    """Collapse any ``DistributionForecast`` to a 1-D point forecast.

    Thin wrapper over ``dist.to_point(how=how)`` — each subclass implements
    the math. Kept as a free function for callers that pass a dist as first
    positional arg.
    """
    return dist.to_point(how=how)


def log_score_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Negative log-density per row for bracket-backed dist (uniform-in-bin).

    Per-row edges supported: each row uses its own bracket grid via
    ``BracketForecast.realized_bin`` (NaN-padded tails are ignored).
    """
    if not isinstance(dist, BracketForecast):
        raise NotImplementedError(
            f"log_score_bracket expects BracketForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    edges = dist.edges
    probs = dist.probs
    widths = np.diff(edges, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(widths > 0, probs / widths, 0.0)
    bin_idx = dist.realized_bin(y)
    px = density[np.arange(density.shape[0]), bin_idx]
    px = np.maximum(px, 1e-300)
    return -np.log(px)


def crps_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """CRPS for a bracket-backed distribution.

    Computes ∫(F(z) - 1[z ≥ y])² dz under uniform-within-bin density.
    Per-row edges supported (NaN-padded tail columns are skipped).
    """
    if not isinstance(dist, BracketForecast):
        raise NotImplementedError(
            f"crps_bracket expects BracketForecast; got {type(dist).__name__}"
        )
    y = np.asarray(y, dtype=float)
    edges = dist.edges
    probs_clean = np.nan_to_num(dist.probs, nan=0.0)
    N, B_max = probs_clean.shape
    cum = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(probs_clean, axis=1)], axis=1
    )
    B_per_row = (~np.isnan(dist.probs)).sum(axis=1).astype(int)
    out = np.zeros(N)
    for k in range(B_max):
        active = B_per_row > k
        if not active.any():
            continue
        lo = edges[:, k]
        hi = edges[:, k + 1]
        width = hi - lo
        a = cum[:, k]
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


def _rows_and_onehot(
    contracts: ContractForecast,
    edges: Any,
    y: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Normalise (contracts, edges, y) into per-row (probs, onehot) pairs.

    ``edges`` accepts the same three shapes ``DistributionForecast.integrate``
    does:

    * **1-D** ``(B+1,)`` — one ladder shared by every row.
    * **2-D** ``(N, B+1)`` — a dense per-row grid, all rows the same width.
    * **ragged sequence** — ``len N``, row ``i`` of shape ``(B_i + 1,)``.

    Why this exists. These scorers used to take a single ``(B+1,)`` vector and
    use it for every row, including for ``searchsorted(edges, y)`` — the step
    that decides which bracket the outcome fell in. On a venue whose ladder
    ROTATES (Kalshi relists daily around the forecast, so Monday is
    ``[64,66,68,70,72]`` and Tuesday ``[25,27,29,31,33]``), that scores every
    row after the first against the wrong grid, and it does so **silently**:
    the shapes are compatible, so a number comes back. Measured on a 40-row
    synthetic ladder, Brier 0.8904 against a correct 0.7343.

    Rows with differing bracket COUNTS additionally broke the flat reshape, but
    loudly — that case raised. The dangerous one was always same-count,
    different-values, which is every row of a real rotating ladder.
    """
    y = np.asarray(y, dtype=float)
    fair = np.asarray(contracts.fair_price, dtype=float)

    # The grid the ladder was actually priced on, when the adapter recorded it.
    priced = getattr(contracts.contract_spec, "edges_per_row", None)

    if isinstance(edges, np.ndarray) and edges.ndim == 2:
        per_row: list[np.ndarray] = [np.asarray(e, float) for e in edges]
    elif isinstance(edges, np.ndarray) and edges.ndim == 1:
        per_row = None  # type: ignore[assignment]
        shared = edges.astype(float)
    else:
        seq = list(edges)
        if len(seq) and np.ndim(seq[0]) == 0:
            per_row = None  # type: ignore[assignment]
            shared = np.asarray(seq, dtype=float)
        else:
            per_row = [np.asarray(e, float) for e in seq]

    if per_row is None:
        B = shared.shape[0] - 1
        if fair.size % B != 0:
            raise ValueError(
                f"fair_price size {fair.size} not divisible by B={B}")
        n = fair.size // B
        # A single vector is correct only if the ladder really was shared. When
        # the adapter recorded its grid we can check instead of hope: scoring a
        # rotating ladder against one row's edges returns a plausible wrong
        # number (measured 0.8904 vs a correct 0.7343), which is exactly the
        # failure this argument used to invite.
        if priced is not None and len(priced) == n:
            mismatched = [
                i for i, e in enumerate(priced)
                if len(e) != shared.shape[0]
                or not np.allclose(np.asarray(e, float), shared, equal_nan=True)
            ]
            if mismatched:
                raise ValueError(
                    f"a single shared ladder was passed, but the contracts "
                    f"were priced on per-row edges that differ at "
                    f"{len(mismatched)} of {n} rows (first: row "
                    f"{mismatched[0]}). Scoring those against one row's grid "
                    f"puts each outcome in the wrong bracket. Pass the same "
                    f"per-row edges the ladder was priced with."
                )
        per_row = [shared] * n

    n = len(per_row)
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} entities; edges describe {n}")

    widths = [e.shape[0] - 1 for e in per_row]
    if sum(widths) != fair.size:
        raise ValueError(
            f"fair_price size {fair.size} does not match the ladder "
            f"({sum(widths)} contracts over {n} rows). If the ladder rotates, "
            f"pass the SAME per-row edges used to price it — a single shared "
            f"vector silently scores every row against row 0's grid."
        )

    probs_rows: list[np.ndarray] = []
    onehot_rows: list[np.ndarray] = []
    off = 0
    for i, e in enumerate(per_row):
        b = widths[i]
        probs_rows.append(fair[off:off + b])
        oh = np.zeros(b)
        oh[int(np.clip(np.searchsorted(e, y[i], side="right") - 1, 0, b - 1))] = 1.0
        onehot_rows.append(oh)
        off += b
    return probs_rows, onehot_rows


def log_loss_bracket(
    contracts: ContractForecast,
    edges: Any,
    y: np.ndarray,
    *,
    entity_order: np.ndarray | None = None,
) -> float:
    """Categorical log-loss over a bracket ladder.

    ``contracts`` is a long-form ContractForecast with one row per
    (entity, bin). ``y`` is the realized value per entity. ``edges`` is the
    ladder — a shared ``(B+1,)`` vector, a dense ``(N, B+1)`` grid, or a ragged
    per-row sequence. **Pass the same edges the ladder was priced with**: see
    :func:`_rows_and_onehot` for why a shared vector against a rotating ladder
    is silently wrong rather than an error.
    """
    probs_rows, onehot_rows = _rows_and_onehot(contracts, edges, y)
    p_realized = np.array([
        float(p[oh.argmax()]) for p, oh in zip(probs_rows, onehot_rows)
    ])
    p_realized = np.clip(p_realized, 1e-12, 1.0)
    return float(-np.log(p_realized).mean())


def brier_bracket(
    contracts: ContractForecast,
    edges: Any,
    y: np.ndarray,
) -> float:
    """Multi-class Brier: Σ_b (p_b - 1[y in bin b])², averaged over entities.

    ``edges`` takes the same three shapes as :func:`log_loss_bracket`, and the
    same warning applies: on a rotating ladder, pass per-row edges.
    """
    probs_rows, onehot_rows = _rows_and_onehot(contracts, edges, y)
    return float(np.mean([
        float(((p - oh) ** 2).sum())
        for p, oh in zip(probs_rows, onehot_rows)
    ]))


# ---------------------------------------------------------------------------
# reference-relative value metrics
# ---------------------------------------------------------------------------
#
# The metrics above answer "are my prices CALIBRATED?" — closeness of my price
# ``q`` to the realized outcome ``r``. A prediction-market trader has a second,
# distinct question: "is my price more VALUABLE than the one already quoted?"
# That is a *relative* question, graded against a reference price ``m`` (a
# market quote, a consensus, or any baseline forecast), not against truth.
#
# A more accurate forecast is not always a more valuable one. Value lives in the
# part of your edge that points where the *reference* is wrong, not in raw
# closeness to truth. The guide ``docs/guides/value_vs_accuracy.md`` derives why
# (a forecast's expected betting PnL is the inner product ⟨q−m, r−m⟩) and shows
# a benign synthetic case where accuracy and value disagree.
#
# These functions take three flat arrays over individual binary contracts:
#   q  model price of YES, in [0, 1]
#   m  reference/market price of YES, in [0, 1]
#   r  realized outcome, in {0, 1}
# The bracket-ladder wrappers below build (q, m, r) from a ContractForecast,
# its reference prices, and the realized values.


def _check_qmr(q: np.ndarray, m: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, ...]:
    q = np.asarray(q, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()
    if not (q.shape == m.shape == r.shape):
        raise ValueError(
            f"q, m, r must share shape; got {q.shape}, {m.shape}, {r.shape}"
        )
    if q.size == 0:
        raise ValueError("empty input to reference-relative value metric")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(m)) and np.all(np.isfinite(r))):
        raise ValueError(
            "q, m, r must be finite (no NaN/inf); a NaN reference price or "
            "padded ragged tail would silently produce a NaN score"
        )
    return q, m, r


def edge_alignment(q: np.ndarray, m: np.ndarray, r: np.ndarray) -> float:
    """Edge-Alignment (EA): mean over contracts of ``(q − m)(r − m)``.

    This is the un-thresholded, un-costed expected betting PnL of acting on the
    edge ``q − m`` against the reference price ``m``: you collect ``r − m`` per
    unit bet, sized by the edge. Positive EA means your edge points, on average,
    in the direction the reference turns out to be wrong. ``E[r] = π`` (the
    latent truth), so ``E[EA] = E[(q − m)(π − m)]`` — the soft PnL — even though
    ``π`` is never observed.

    EA is the value sibling of ``brier_bracket``: Brier measures ``‖q − r‖``
    (accuracy), EA measures alignment of ``q − m`` with ``r − m`` (value vs the
    reference). They can rank two forecasts in opposite orders.
    """
    q, m, r = _check_qmr(q, m, r)
    return float(np.mean((q - m) * (r - m)))


def edge_alignment_costed(
    q: np.ndarray,
    m: np.ndarray,
    r: np.ndarray,
    *,
    fee: float,
    tau: float | None = None,
) -> dict[str, float]:
    """Fee-aware value: realized PnL of a **unit-bet** strategy that trades a
    contract only when ``|q − m| > tau`` and pays ``fee`` per contract traded.

    Per contract: ``sign(q − m)·(r − m) − fee`` if ``|q − m| > tau`` else ``0``.
    ``tau`` defaults to ``fee`` (trade only when your edge clears the cost).

    This is the metric to select/train on when fees are non-trivial — and it
    behaves very differently from :func:`edge_alignment`. EA is frictionless and
    **linear in the edge**, so it rewards edge *magnitude* without bound (it
    always prefers a more extreme, more confident forecast). With a fee the
    *magnitude* of your stated edge stops mattering for a unit bet — only its
    **sign** and whether it clears the gate do — so over-confidence can only
    hurt: it flips signs on noisy near-fair contracts and pays ``fee`` on
    sub-fee "junk" trades. The best achievable value is ``E[(|δ| − fee)₊]`` with
    ``δ = E[r − m | x]``: a *deductible* on the true mispricing. Fees convert the
    objective from an inner product into a hinge — see
    ``docs/guides/value_with_fees.md``.

    Returns a dict: ``mean_pnl`` (per contract, the headline number),
    ``total_pnl``, ``per_trade`` (mean PnL over traded contracts), and
    ``trade_frac``.
    """
    q, m, r = _check_qmr(q, m, r)
    if fee < 0:
        raise ValueError(f"fee must be non-negative; got {fee}")
    if tau is None:
        tau = fee
    if tau < 0:
        raise ValueError(f"tau must be non-negative; got {tau}")
    edge = q - m
    side = np.where(edge > tau, 1.0, np.where(edge < -tau, -1.0, 0.0))
    pnl = side * (r - m) - np.abs(side) * fee
    n_trade = int((side != 0).sum())
    return {
        "mean_pnl": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "per_trade": float(pnl[side != 0].mean()) if n_trade else 0.0,
        "trade_frac": float(n_trade / q.size),
    }


def edge_alignment_corr(q: np.ndarray, m: np.ndarray, r: np.ndarray) -> float:
    """Normalized EA: ``corr(q − m, r − m)`` — the cosine of the angle between
    your edge and the reference's realized error. ``→ 0`` is the limit where the
    edge no longer points at the reference's mistakes (the shared-bias trap).
    """
    q, m, r = _check_qmr(q, m, r)
    eq, er = q - m, r - m
    sq, sr = eq.std(), er.std()
    if sq < 1e-15 or sr < 1e-15:
        raise ValueError("zero-variance edge or reference error; corr undefined")
    return float(np.corrcoef(eq, er)[0, 1])


def shared_bias_slope(q: np.ndarray, m: np.ndarray, r: np.ndarray) -> float:
    """OLS slope of your error ``(q − r)`` on the reference's error ``(m − r)``.

    A large positive slope means your errors coincide with the reference's — you
    are forfeiting edge to blind spots you *share* with it (e.g. both anchor to
    the same biased source). Driving this slope down is worth more for value than
    any calibration gain. Slope ``1`` means ``q = m`` (no edge); slope ``0``
    means your residual error is orthogonal to the reference's.
    """
    q, m, r = _check_qmr(q, m, r)
    x = m - r
    sxx = float(np.dot(x - x.mean(), x - x.mean()))
    if sxx < 1e-15:
        raise ValueError("zero-variance reference error; slope undefined")
    yq = q - r
    return float(np.dot(x - x.mean(), yq - yq.mean()) / sxx)


def value_report(q: np.ndarray, m: np.ndarray, r: np.ndarray) -> dict[str, float]:
    """Full reference-relative value diagnostic for one set of contracts.

    Returns ``EA`` and its exact additive split ``EA = A − B`` (no latent ``π``
    needed):

      * ``A = mean (r − m)²`` — the reference's mean-squared error (its Brier).
        How much mispricing is *available*. Outside your control.
      * ``B = mean (r − q)(r − m)`` — co-projection of your error onto the
        reference's. How much of the available mispricing your forecast *fails*
        to capture because your errors coincide with the reference's.

    Identity: ``(q−m)(r−m) = (r−m)² − (r−q)(r−m)``, so ``EA = A − B`` per
    contract. When EA moves across models or regimes, ``ΔEA = ΔA − ΔB``
    attributes the change: ``A`` down ⇒ the reference got more efficient
    (less to capture); ``B`` up ⇒ your forecast lost orthogonality (a model
    problem, fixable). Both ``A`` and ``B`` sit on the same irreducible
    Bernoulli-variance floor, which cancels in ``EA = A − B`` — read the levels
    with care, read the difference cleanly.
    """
    q, m, r = _check_qmr(q, m, r)
    A = float(np.mean((r - m) ** 2))
    B = float(np.mean((r - q) * (r - m)))
    return {
        "EA": A - B,
        "A_reference_mse": A,
        "B_non_orthogonality": B,
        "align_corr": edge_alignment_corr(q, m, r),
        "shared_bias_slope": shared_bias_slope(q, m, r),
        "n_contracts": float(q.size),
    }


def _qmr_from_bracket(
    contracts: ContractForecast,
    reference: ContractForecast | np.ndarray,
    edges: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a model ladder, a reference ladder, and realized y into (q, m, r)
    over every (entity, bracket) binary contract."""
    edges = np.asarray(edges, dtype=float)
    y = np.asarray(y, dtype=float)
    B = edges.shape[0] - 1
    q = np.asarray(contracts.fair_price, dtype=float)
    m = reference.fair_price if isinstance(reference, ContractForecast) else np.asarray(reference, dtype=float)
    m = np.asarray(m, dtype=float)
    if q.shape != m.shape:
        raise ValueError(
            f"model fair_price {q.shape} and reference {m.shape} must match"
        )
    if q.size % B != 0:
        raise ValueError(f"fair_price size {q.size} not divisible by B={B}")
    N = q.size // B
    if y.shape[0] != N:
        raise ValueError(f"y has {y.shape[0]} entities; contracts have {N}")
    onehot = np.zeros((N, B), dtype=float)
    bin_idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, B - 1)
    onehot[np.arange(N), bin_idx] = 1.0
    return q.reshape(N, B).ravel(), m.reshape(N, B).ravel(), onehot.ravel()


def edge_alignment_bracket(
    contracts: ContractForecast,
    reference: ContractForecast | np.ndarray,
    edges: np.ndarray,
    y: np.ndarray,
) -> float:
    """Edge-Alignment of a bracket ladder vs a reference ladder (a scalar).

    ``reference`` is the quoted/baseline price for the same contracts — a
    ``ContractForecast`` or a raw array matching ``contracts.fair_price``. Each
    (entity, bracket) becomes a binary contract; EA averages ``(q−m)(r−m)`` over
    all of them. See :func:`edge_alignment`.
    """
    q, m, r = _qmr_from_bracket(contracts, reference, edges, y)
    return edge_alignment(q, m, r)


def value_report_bracket(
    contracts: ContractForecast,
    reference: ContractForecast | np.ndarray,
    edges: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Full value diagnostic (``EA``, ``A``, ``B``, ``align_corr``,
    ``shared_bias_slope``) for a bracket ladder vs a reference. See
    :func:`value_report`."""
    q, m, r = _qmr_from_bracket(contracts, reference, edges, y)
    return value_report(q, m, r)


def _qmr_from_dist(
    dist: DistributionForecast,
    reference_by_id: dict,
    y: np.ndarray | dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a bracket-backed ``DistributionForecast`` (a value trainer's
    ``predict_dist`` output — ragged, NaN-padded), a per-id reference-price dict,
    and realized ``y`` into flat ``(q, m, r)`` over every (row, bracket) binary.

    Each row's model probabilities are renormalized over that row's **valid**
    (finite) brackets — the edge you would actually trade. ``reference_by_id`` is
    keyed by the dist's ids and must match each row's bracket count. ``y`` may be
    an array aligned to the dist's row order, or a dict keyed by id.
    """
    if not isinstance(dist, BracketForecast):
        raise TypeError(
            "value report needs a bracket-backed DistributionForecast "
            "(a BracketForecast with .probs / .edges), e.g. a value trainer's "
            f"predict_dist output; got {type(dist).__name__}"
        )
    ids = np.asarray(dist.ids)
    probs = np.asarray(dist.probs, dtype=np.float64)
    edges = np.asarray(dist.edges, dtype=np.float64)
    N = ids.shape[0]
    if isinstance(y, dict):
        missing = [i for i in ids if i not in y]
        if missing:
            raise KeyError(f"y missing {len(missing)} id(s); first: {missing[:3]}")
        y_arr = np.array([float(y[i]) for i in ids], dtype=float)
    else:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape[0] != N:
            raise ValueError(f"y has length {y_arr.shape[0]} but dist has {N} rows")
    q_parts, m_parts, r_parts = [], [], []
    for i in range(N):
        B_i = int(np.isfinite(probs[i]).sum())
        if B_i < 1:
            raise ValueError(f"row {i} (id {ids[i]!r}) has no finite probabilities")
        q_i = probs[i, :B_i]
        s = float(q_i.sum())
        if s <= 0:
            raise ValueError(f"row {i} (id {ids[i]!r}) probabilities sum to {s}")
        q_i = q_i / s
        rid = ids[i]
        if rid not in reference_by_id:
            raise KeyError(f"reference_by_id missing id {rid!r}")
        m_i = np.asarray(reference_by_id[rid], dtype=float)
        if m_i.shape[0] != B_i:
            raise ValueError(
                f"reference_by_id[{rid!r}] has {m_i.shape[0]} prices but row has "
                f"{B_i} brackets"
            )
        row_edges = edges[i, : B_i + 1]
        oh = np.zeros(B_i, dtype=float)
        k = int(np.clip(np.searchsorted(row_edges, y_arr[i], side="right") - 1, 0, B_i - 1))
        oh[k] = 1.0
        q_parts.append(q_i)
        m_parts.append(m_i)
        r_parts.append(oh)
    return np.concatenate(q_parts), np.concatenate(m_parts), np.concatenate(r_parts)


def edge_alignment_dist(
    dist: DistributionForecast,
    reference_by_id: dict,
    y: np.ndarray | dict,
) -> float:
    """Edge-Alignment of a fitted bracket model's ``predict_dist`` output vs a
    per-id reference — :func:`edge_alignment` with the ragged flatten done for
    you. See :func:`value_report_dist` for the full diagnostic."""
    q, m, r = _qmr_from_dist(dist, reference_by_id, y)
    return edge_alignment(q, m, r)


def value_report_dist(
    dist: DistributionForecast,
    reference_by_id: dict,
    y: np.ndarray | dict,
    *,
    fee: float | None = None,
    tau: float | None = None,
) -> dict[str, float]:
    """One-call value report for a fitted bracket model's ``predict_dist`` output.

    Pass the distribution, the **same** ``reference_by_id`` you trained with, and
    realized ``y`` (array in row order or dict by id) — this does the per-row
    ragged flatten + renormalization and returns :func:`value_report`. When
    ``fee`` is given, the costed metrics (:func:`edge_alignment_costed`) are
    merged in under ``costed_*`` keys, so ``λ`` selection by costed value is a
    single call.
    """
    q, m, r = _qmr_from_dist(dist, reference_by_id, y)
    rep = value_report(q, m, r)
    if fee is not None:
        costed = edge_alignment_costed(q, m, r, fee=fee, tau=tau)
        rep.update({f"costed_{k}": v for k, v in costed.items()})
    return rep


def bootstrap_ci(
    fn: Callable[..., float],
    *arrays: np.ndarray,
    cluster: np.ndarray | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a contract-level metric, clustered.

    ``fn`` is any of this module's scorers — ``edge_alignment``,
    ``edge_alignment_costed``, a Brier — called as ``fn(*arrays)`` on a
    resample. ``arrays`` are the per-contract vectors it takes (``q, m, r``),
    resampled together so a draw keeps each contract's triple intact.

    **Pass ``cluster`` whenever contracts share an outcome.** On a bracket
    ladder every contract for one (station, day) resolves off the SAME realized
    value: exactly one wins and the rest lose, by construction. They are not
    independent draws, and resampling contracts i.i.d. treats each as fresh
    evidence.

    How much that matters is an empirical question, not a fixed factor, and it
    is smaller here than the "one shared outcome" framing suggests. Measured on
    this repo's weather sample (~6 brackets per station-day, ~1,000 station-day
    clusters): the clustered interval is **1.02x wider on HIGH and 1.07x on
    LOW** than the i.i.d. one. EA is a per-contract product whose within-ladder
    correlation is weak even though the outcome is shared — the one-hot
    constraint ties the r's, but the (q-m) edges vary freely across brackets.
    Expect a large inflation only when the metric itself aggregates at the
    cluster level, or when clusters are few and heterogeneous.

    Use it anyway: it costs nothing, it is the correct estimator for dependent
    data, and the factor is a property of your sample rather than something to
    assume.

    ``cluster`` is a per-contract label (a station-day id). Whole clusters are
    then resampled with replacement — the block bootstrap — so the resample has
    the same dependence structure as the sample.

    Returns ``point`` (``fn`` on the full sample), ``lo``/``hi`` (the
    ``alpha/2`` and ``1-alpha/2`` percentiles), ``se`` (the bootstrap standard
    deviation), ``n_boot`` (draws that scored finite), ``n_clusters`` and
    ``n_obs``.

    Two properties worth stating, because they decide how to read the output:

    * **This is a percentile interval, not a bias-corrected one.** For a
      near-symmetric statistic like EA it is fine. For a strongly skewed one,
      prefer BCa; this function does not implement it.
    * **The CI is about sampling noise only.** It says nothing about whether
      the population is the one you meant, whether the split leaked, or whether
      the reference price is the one you could trade. A tight interval around a
      wrong number is still wrong.

    Draws whose resample makes ``fn`` undefined (a degenerate cluster draw
    giving zero-variance input, say) are skipped and excluded from ``n_boot``
    rather than counted as zero. If fewer than half the draws survive, that is
    a broken setup and it raises (Rule: no silent fallback).
    """
    if not arrays:
        raise ValueError("bootstrap_ci needs at least one data array")
    arrs = [np.asarray(a).ravel() for a in arrays]
    n = arrs[0].size
    if any(a.size != n for a in arrs):
        raise ValueError(
            f"arrays must share length; got {[a.size for a in arrs]}")
    if n == 0:
        raise ValueError("empty input to bootstrap_ci")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")

    point = float(fn(*arrs))
    rng = np.random.default_rng(seed)

    if cluster is None:
        # One observation per group: the ordinary i.i.d. bootstrap. NOT a
        # single group containing everything — that resamples the identical
        # sample every draw and yields a zero-width interval.
        groups = [np.array([i]) for i in range(n)]
    else:
        cl = np.asarray(cluster).ravel()
        if cl.size != n:
            raise ValueError(
                f"cluster must have one label per observation; got {cl.size} "
                f"for {n} observations")
        order = np.argsort(cl, kind="stable")
        _, starts = np.unique(cl[order], return_index=True)
        groups = np.split(order, starts[1:])

    n_groups = len(groups)
    stats: list[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([groups[j] for j in pick])
        try:
            v = float(fn(*[a[idx] for a in arrs]))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if math.isfinite(v):
            stats.append(v)

    if len(stats) < max(1, n_boot // 2):
        raise ValueError(
            f"only {len(stats)} of {n_boot} bootstrap draws produced a finite "
            f"score; the metric is degenerate on this sample rather than "
            f"merely uncertain")

    s = np.asarray(stats, dtype=float)
    lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(s.std(ddof=1)) if s.size > 1 else float("nan"),
        "n_boot": float(s.size),
        "n_clusters": float(n_groups),
        "n_obs": float(n),
    }
