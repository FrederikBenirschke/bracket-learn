"""Distribution-level and contract-level scoring.

v0.1 supplies the essentials for the end-to-end demo.

- dist.crps_gaussian, CRPS for a Gaussian parametric backing.
- dist.log_score_gaussian, predictive log-likelihood.
- dist.pit, Probability Integral Transform values for diagnostics.
- contract.log_loss_bracket, categorical log-loss over a bracket ladder.
- contract.brier_bracket, multi-class Brier on a bracket ladder.

Free functions delegate via ``isinstance`` to the matching dist subclass. A new
backing should add its math as methods on the subclass. The free functions
remain for downstream callers that pass a dist as the first argument.
"""

from __future__ import annotations

import math
import warnings
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

    The CDF is treated as piecewise-linear between the stored quantiles, so
    the density is piecewise-constant. Between τ_i and τ_{i+1}, the density
    at any y in [q_i, q_{i+1}] is (τ_{i+1} - τ_i) / (q_{i+1} - q_i).

    Below q_0 and above q_{Q-1} the density falls back to a tail rule. The
    local density of the outermost bin is extended, a cautious choice that
    mirrors ``tail_policy="clip"``.
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

    Thin wrapper over ``dist.to_point(how=how)``. Each subclass implements
    the math. Kept as a free function for callers that pass a dist as the
    first positional argument.
    """
    return dist.to_point(how=how)


def log_score_bracket(dist: DistributionForecast, y: np.ndarray) -> np.ndarray:
    """Negative log-density per row for bracket-backed dist (uniform-in-bin).

    Per-row edges are supported. Each row uses its own bracket grid via
    ``BracketForecast.realized_bin``, and NaN-padded tails are ignored.
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
    # An unbounded bin holding mass makes the integral diverge. The uniform
    # density there is p/inf = 0, and the tail contributes a**2 * inf. This is
    # a real divergence rather than a numerical artifact. It previously
    # surfaced as a bare nan (0 * inf) with no indication of the offending
    # row. The +-inf ladder is this library's canonical encoding, so the case
    # is refused explicitly. log_score and pit are finite here and need no
    # midpoint, and are the metrics to use on an open ladder.
    _widths = edges[:, 1:] - edges[:, :-1]
    _bad = ~np.isfinite(_widths) & (probs_clean > 0.0)
    if _bad.any():
        _rows = np.flatnonzero(_bad.any(axis=1))
        _i = int(_rows[0])
        _k = int(np.flatnonzero(_bad[_i])[0])
        raise ValueError(
            f"crps_bracket: row {_i} bin {_k} spans "
            f"[{edges[_i, _k]}, {edges[_i, _k + 1]}] and carries "
            f"{probs_clean[_i, _k]:.6g} of the mass, so the CRPS integral "
            f"diverges. {len(_rows)} of {N} row(s) affected. Score an open "
            f"ladder with log_score_bracket or pit, or re-price onto finite "
            f"outer edges."
        )
    cum = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(probs_clean, axis=1)], axis=1
    )
    B_per_row = (~np.isnan(dist.probs)).sum(axis=1).astype(int)
    out = np.zeros(N)
    for k in range(B_max):
        active = B_per_row > k
        # A zero-mass unbounded bin contributes exactly zero. It can only be
        # an outer tail, where F is 0 (left) or 1 (right) throughout, so the
        # integrand vanishes identically. Evaluating it anyway forms
        # 0**2 * inf = nan and corrupts the whole row. Bins that are both
        # unbounded and carry mass were refused above.
        active = active & ~(~np.isfinite(edges[:, k + 1] - edges[:, k])
                            & (probs_clean[:, k] == 0.0))
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

    ``edges`` accepts the same three shapes as
    ``DistributionForecast.integrate``.

    * 1-D ``(B+1,)``, one ladder shared by every row.
    * 2-D ``(N, B+1)``, a dense per-row grid, all rows the same width.
    * ragged sequence of length ``N``, row ``i`` of shape ``(B_i + 1,)``.

    These scorers previously took a single ``(B+1,)`` vector and applied it to
    every row, including in ``searchsorted(edges, y)``, the step that decides
    which bracket the outcome fell in. On a venue whose ladder rotates, every
    row after the first is then scored against the wrong grid. Kalshi relists
    daily around the forecast, so Monday is ``[64,66,68,70,72]`` and Tuesday
    ``[25,27,29,31,33]``. The shapes remain compatible and a number is
    returned. Measured on a 40-row synthetic ladder, Brier 0.8904 against a
    correct 0.7343.

    Rows with differing bracket counts also broke the flat reshape, but that
    case raised. The damaging case is same-count with different values, which
    is every row of a real rotating ladder.
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
        # A single vector is correct only if the ladder really was shared.
        # Where the adapter recorded its grid, verify rather than assume.
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
            f"pass the SAME per-row edges used to price it, a single shared "
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
    ladder, either a shared ``(B+1,)`` vector, a dense ``(N, B+1)`` grid, or a
    ragged per-row sequence. Pass the same edges the ladder was priced with.
    See :func:`_rows_and_onehot`, where a shared vector against a rotating
    ladder returns a wrong number rather than an error.
    """
    probs_rows, onehot_rows = _rows_and_onehot(contracts, edges, y)
    p_realized = np.array([
        float(p[oh.argmax()]) for p, oh in zip(probs_rows, onehot_rows, strict=True)
    ])
    p_realized = np.clip(p_realized, 1e-12, 1.0)
    return float(-np.log(p_realized).mean())


def brier_bracket(
    contracts: ContractForecast,
    edges: Any,
    y: np.ndarray,
) -> float:
    """Multi-class Brier, Σ_b (p_b - 1[y in bin b])², averaged over entities.

    ``edges`` takes the same three shapes as :func:`log_loss_bracket`. The
    same warning applies. On a rotating ladder, pass per-row edges.
    """
    probs_rows, onehot_rows = _rows_and_onehot(contracts, edges, y)
    return float(np.mean([
        float(((p - oh) ** 2).sum())
        for p, oh in zip(probs_rows, onehot_rows, strict=True)
    ]))


# ---------------------------------------------------------------------------
# reference-relative value metrics
# ---------------------------------------------------------------------------
#
# The metrics above measure calibration, the closeness of a price ``q`` to the
# realized outcome ``r``. A prediction-market trader has a second question,
# whether a price is more valuable than the one already quoted. That question
# is relative. It is graded against a reference price ``m``, which may be a
# market quote, a consensus, or any baseline forecast, rather than against
# truth.
#
# A more accurate forecast is not always a more valuable one. Value lives in
# the part of the edge that points where the reference is wrong, not in raw
# closeness to truth. The guide ``docs/guides/value_vs_accuracy.md`` derives
# the expected betting PnL as the inner product ⟨q−m, r−m⟩ and gives a benign
# synthetic case where accuracy and value disagree.
#
# These functions take three flat arrays over individual binary contracts.
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
    """Edge-Alignment (EA), the mean over contracts of ``(q − m)(r − m)``.

    EA is the un-thresholded, un-costed expected betting PnL of acting on the
    edge ``q − m`` against the reference price ``m``. Each unit bet collects
    ``r − m``, sized by the edge. Positive EA means the edge points on average
    in the direction the reference turns out to be wrong. Since ``E[r] = π``
    for the latent truth ``π``, we have ``E[EA] = E[(q − m)(π − m)]``. The soft
    PnL is recovered even though ``π`` is never observed.

    EA is the value sibling of ``brier_bracket``. Brier measures accuracy as
    ``‖q − r‖``. EA measures value against the reference as the alignment of
    ``q − m`` with ``r − m``. The two can rank forecasts in opposite orders.
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
    """Fee-aware value, the realized PnL of a unit-bet strategy.

    A contract is traded only when ``|q − m| > tau``, and ``fee`` is paid per
    contract traded. Per contract the PnL is ``sign(q − m)·(r − m) − fee`` when
    ``|q − m| > tau`` and ``0`` otherwise. ``tau`` defaults to ``fee``, so a
    trade is taken only when the edge clears the cost.

    This is the metric to select or train on when fees are non-trivial. Its
    behaviour differs from :func:`edge_alignment`. EA is frictionless and
    linear in the edge, so it rewards edge magnitude without bound and always
    prefers a more extreme, more confident forecast. Under a fee the magnitude
    of the stated edge no longer matters for a unit bet. Only its sign and
    whether it clears the gate matter, so over-confidence can only hurt. It
    flips signs on noisy near-fair contracts and pays ``fee`` on sub-fee
    trades. The best achievable value is ``E[(|δ| − fee)₊]`` with
    ``δ = E[r − m | x]``, a deductible on the true mispricing. Fees convert the
    objective from an inner product into a hinge. See
    ``docs/guides/value_with_fees.md``.

    Returns a dict with ``mean_pnl`` (per contract, the headline number),
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
    """Normalized EA, ``corr(q − m, r − m)``.

    This is the cosine of the angle between the edge and the reference's
    realized error. The limit ``→ 0`` is where the edge no longer points at the
    reference's mistakes, the shared-bias regime.
    """
    q, m, r = _check_qmr(q, m, r)
    eq, er = q - m, r - m
    sq, sr = eq.std(), er.std()
    if sq < 1e-15 or sr < 1e-15:
        raise ValueError("zero-variance edge or reference error; corr undefined")
    return float(np.corrcoef(eq, er)[0, 1])


def shared_bias_slope(q: np.ndarray, m: np.ndarray, r: np.ndarray) -> float:
    """OLS slope of your error ``(q − r)`` on the reference's error ``(m − r)``.

    A large positive slope means the two error series coincide. Edge is then
    forfeited to blind spots shared with the reference, for instance when both
    anchor to the same biased source. Driving this slope down is worth more for
    value than any calibration gain. Slope ``1`` means ``q = m`` and no edge.
    Slope ``0`` means the residual error is orthogonal to the reference's.
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

    Returns ``EA`` and its exact additive split ``EA = A − B``, which requires
    no latent ``π``.

      * ``A = mean (r − m)²``, the reference's mean-squared error, its Brier.
        This is how much mispricing is available, and is outside the model's
        control.
      * ``B = mean (r − q)(r − m)``, the co-projection of the model's error
        onto the reference's. This is how much of the available mispricing the
        forecast fails to capture because the two error series coincide.

    The identity ``(q−m)(r−m) = (r−m)² − (r−q)(r−m)`` gives ``EA = A − B`` per
    contract. When EA moves across models or regimes, ``ΔEA = ΔA − ΔB``
    attributes the change. A fall in ``A`` means the reference became more
    efficient and there is less to capture. A rise in ``B`` means the forecast
    lost orthogonality, which is a model problem and is fixable. Both ``A`` and
    ``B`` sit on the same irreducible Bernoulli-variance floor, which cancels
    in ``EA = A − B``. The levels should be read with care and the difference
    reads cleanly.
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
    edges: Any,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a model ladder, a reference ladder, and realized y into
    (q, m, r) over every (entity, bracket) binary contract.

    ``edges`` takes the three shapes documented in :func:`_rows_and_onehot`, a
    shared ``(B+1,)`` vector, a dense ``(N, B+1)`` grid, or a ragged per-row
    sequence. As in the Brier and log-loss scorers, a single vector is refused
    when the ladder was priced per-row. This function accepted only a 1-D
    vector until v0.8. On a rotating ladder that inverted the sign of EA,
    giving -0.0205 against a correct +0.0013. The consequence is more severe
    than in the accuracy scorers, since the sign of EA determines the verdict.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(contracts.fair_price, dtype=float)
    m = reference.fair_price if isinstance(reference, ContractForecast) else reference
    m = np.asarray(m, dtype=float)
    if q.shape != m.shape:
        raise ValueError(
            f"model fair_price {q.shape} and reference {m.shape} must match"
        )
    probs_rows, onehot_rows = _rows_and_onehot(contracts, edges, y)
    # _rows_and_onehot already validated the ladder against q. Since m shares
    # its shape, the same row widths slice it.
    widths = [p.size for p in probs_rows]
    m_rows, off = [], 0
    for b in widths:
        m_rows.append(m[off:off + b])
        off += b
    return (
        np.concatenate(probs_rows),
        np.concatenate(m_rows),
        np.concatenate(onehot_rows),
    )


def edge_alignment_bracket(
    contracts: ContractForecast,
    reference: ContractForecast | np.ndarray,
    edges: Any,
    y: np.ndarray,
) -> float:
    """Edge-Alignment of a bracket ladder vs a reference ladder (a scalar).

    ``reference`` is the quoted or baseline price for the same contracts,
    either a ``ContractForecast`` or a raw array matching
    ``contracts.fair_price``. Each (entity, bracket) becomes a binary contract.
    EA averages ``(q−m)(r−m)`` over all of them. See :func:`edge_alignment`.
    """
    q, m, r = _qmr_from_bracket(contracts, reference, edges, y)
    return edge_alignment(q, m, r)


def value_report_bracket(
    contracts: ContractForecast,
    reference: ContractForecast | np.ndarray,
    edges: Any,
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
    """Flatten a bracket-backed ``DistributionForecast``, a per-id
    reference-price dict, and realized ``y`` into flat ``(q, m, r)`` over every
    (row, bracket) binary contract.

    The distribution is a value trainer's ``predict_dist`` output, which is
    ragged and NaN-padded. Each row's model probabilities are renormalized over
    that row's valid, finite brackets, which is the edge actually traded.
    ``reference_by_id`` is keyed by the dist's ids and must match each row's
    bracket count. ``y`` may be an array aligned to the dist's row order, or a
    dict keyed by id.
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
    """Edge-Alignment of a fitted bracket model's ``predict_dist`` output
    against a per-id reference. This is :func:`edge_alignment` with the ragged
    flatten applied first. See :func:`value_report_dist` for the full
    diagnostic."""
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

    Pass the distribution, the same ``reference_by_id`` used for training, and
    realized ``y`` as an array in row order or a dict by id. The per-row ragged
    flatten and renormalization are applied and :func:`value_report` is
    returned. When ``fee`` is given, the costed metrics from
    :func:`edge_alignment_costed` are merged in under ``costed_*`` keys, which
    makes selection of ``λ`` by costed value a single call.
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

    ``fn`` is any of this module's scorers, such as ``edge_alignment``,
    ``edge_alignment_costed``, or a Brier. It is called as ``fn(*arrays)`` on a
    resample. ``arrays`` are the per-contract vectors it takes, ``q, m, r``,
    resampled together so a draw keeps each contract's triple intact.

    ``cluster`` should be passed whenever contracts share an outcome. On a
    bracket ladder every contract for one (station, day) resolves off the same
    realized value. Exactly one wins and the rest lose, by construction. They
    are not independent draws, and resampling contracts i.i.d. treats each as
    fresh evidence.

    How much that matters is an empirical question rather than a fixed factor,
    and it is smaller here than the shared-outcome framing suggests. Measured
    on this repo's weather sample of roughly 6 brackets per station-day and
    roughly 1,000 station-day clusters, the clustered interval is 1.02x wider
    on HIGH and 1.07x wider on LOW than the i.i.d. one. The mechanism is a
    cancellation, and it was measured rather than assumed. Both ``q`` and ``m``
    are simplex vectors, so ``q-m`` sums to exactly zero within a ladder. That
    forces its intra-cluster correlation to ``-1/(L-1)``, observed as -0.2001
    for L=6 with ``max|sum(q-m)|`` = 2.8e-16. The negative dependence offsets
    the positive dependence induced by the shared outcome, leaving the EA
    product nearly uncorrelated within a ladder, with ICC +0.004 on HIGH and
    +0.037 on LOW. The factor is larger on LOW because 1.6% of its ladders have
    an unquoted winning bracket, which partly breaks the sum-to-zero
    constraint. HIGH has one such row in 2,895. A large inflation should be
    expected only where that constraint fails, where the reference is
    unnormalised, where the metric itself aggregates at the cluster level, or
    where clusters are few and heterogeneous.

    Clustering is preferred whenever contracts share an outcome. The factor is
    sample-specific and should be measured rather than assumed.

    ``cluster`` is a per-contract label such as a station-day id. Whole
    clusters are then resampled with replacement, the block bootstrap, so the
    resample has the same dependence structure as the sample.

    Returns ``point``, the value of ``fn`` on the full sample, ``lo`` and
    ``hi``, the ``alpha/2`` and ``1-alpha/2`` percentiles, ``se``, the
    bootstrap standard deviation, ``n_boot``, the number of draws that scored
    finite, together with ``n_clusters`` and ``n_obs``.

    Two properties determine how the output should be read.

    * The interval is a percentile interval and is not bias-corrected. This is
      adequate for a near-symmetric statistic such as EA. For a strongly skewed
      statistic, BCa is preferable and is not implemented here.

    Draws whose resample leaves ``fn`` undefined, for instance a degenerate
    cluster draw giving zero-variance input, are skipped and excluded from
    ``n_boot`` rather than counted as zero. Fewer than half the draws surviving
    indicates a degenerate metric rather than an uncertain one, and raises.
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
        # One observation per group, the ordinary i.i.d. bootstrap. This is
        # not a single group containing everything, which would resample the
        # identical sample every draw and yield a zero-width interval.
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
    if n_groups < 2:
        raise ValueError(
            f"bootstrap_ci: all {n} observations fall in ONE cluster, so every "
            f"resample is the identical sample and the interval collapses to "
            f"zero width, which reads as extraordinary precision rather than "
            f"as a broken input. Pass a cluster label that varies, or "
            f"cluster=None for the i.i.d. bootstrap."
        )
    if n_groups < 20:
        warnings.warn(
            f"bootstrap_ci: only {n_groups} clusters. The percentile interval "
            f"under-covers materially below ~20 clusters (measured ~0.74 "
            f"actual coverage at K=3 for a nominal 0.95), so coarsening "
            f"clusters to be 'conservative' can do the opposite.",
            UserWarning, stacklevel=2,
        )
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
