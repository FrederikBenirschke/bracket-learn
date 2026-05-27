"""Score / metric known-answer tests.

For each metric we verify either:
- a closed-form analytical answer on a tiny hand-crafted example, or
- a Monte-Carlo expected value (with large enough N that flakiness is
  bounded), or
- a proper-score property (lower-bound at the true distribution).

These tests pin metric correctness — a regression here means a forecast
that should win the leaderboard might lose, which is the worst kind of
silent bug.
"""

from __future__ import annotations

import numpy as np

from bracketlearn.forecast import DistributionForecast, TailPolicy, TailRule
from bracketlearn.score import (
    brier_bracket,
    crps_bracket,
    crps_gaussian,
    crps_quantile,
    log_loss_bracket,
    log_score_gaussian,
    log_score_mixture_normal,
    pit,
)

# ---------------------------------------------------------------------------
# Gaussian CRPS — closed-form check.
# ---------------------------------------------------------------------------


def test_crps_gaussian_zero_at_atom(prov, ids_ts):
    """CRPS of N(μ, σ→0) against y=μ should go to 0."""
    ids, ts = ids_ts(1)
    d = DistributionForecast.from_normal(
        mu=np.array([5.0]), sigma=np.array([1e-4]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    crps = crps_gaussian(d, np.array([5.0]))
    assert crps[0] < 1e-3


def test_crps_gaussian_closed_form(prov, ids_ts):
    """CRPS(N(0,1), y=0) = 2·φ(0) − 1/√π = 2/√(2π) − 1/√π ≈ 0.2336."""
    ids, ts = ids_ts(1)
    d = DistributionForecast.from_normal(
        mu=np.array([0.0]), sigma=np.array([1.0]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    expected = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)
    np.testing.assert_allclose(crps_gaussian(d, np.array([0.0]))[0], expected, atol=1e-10)


def test_crps_gaussian_scale_invariance(prov, ids_ts):
    """CRPS(N(μ, σ), y) = σ · CRPS(N(0, 1), (y−μ)/σ)."""
    ids, ts = ids_ts(1)
    d_unit = DistributionForecast.from_normal(
        mu=np.array([0.0]), sigma=np.array([1.0]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    d_scaled = DistributionForecast.from_normal(
        mu=np.array([3.0]), sigma=np.array([2.5]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = np.array([4.2])
    unit_crps = crps_gaussian(d_unit, np.array([(y[0] - 3.0) / 2.5]))[0]
    scaled_crps = crps_gaussian(d_scaled, y)[0]
    np.testing.assert_allclose(scaled_crps, 2.5 * unit_crps, atol=1e-10)


# ---------------------------------------------------------------------------
# Log score — sanity + propriety.
# ---------------------------------------------------------------------------


def test_log_score_gaussian_minimised_at_truth(prov, ids_ts, rng):
    """Among Gaussian forecasts of varying μ, the one centered at E[y] should
    minimise expected log-score. (Proper-scoring-rule property.)"""
    ids, ts = ids_ts(5000)
    y = rng.normal(0.0, 1.0, size=5000)
    candidate_mus = np.linspace(-1.0, 1.0, 11)
    losses = []
    for mu in candidate_mus:
        d = DistributionForecast.from_normal(
            mu=np.full(5000, mu), sigma=np.ones(5000),
            ids=ids, timestamps=ts, provenance=prov,
        )
        losses.append(log_score_gaussian(d, y).mean())
    best_idx = int(np.argmin(losses))
    assert abs(candidate_mus[best_idx]) < 0.25     # very close to 0


def test_log_score_mixture_matches_normal_when_collapsed(prov, ids_ts, rng):
    """log_score_mixture_normal with all weight on one component must equal
    log_score_gaussian for that component."""
    ids, ts = ids_ts(100)
    y = rng.normal(0.0, 1.0, size=100)
    d_normal = DistributionForecast.from_normal(
        mu=np.zeros(100), sigma=np.ones(100),
        ids=ids, timestamps=ts, provenance=prov,
    )
    d_mix = DistributionForecast.from_mixture_normal(
        weights=np.tile([1.0, 0.0], (100, 1)),
        mus=np.tile([0.0, 99.0], (100, 1)),
        sigmas=np.tile([1.0, 1.0], (100, 1)),
        ids=ids, timestamps=ts, provenance=prov,
    )
    np.testing.assert_allclose(
        log_score_gaussian(d_normal, y),
        log_score_mixture_normal(d_mix, y),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Bracket scoring.
# ---------------------------------------------------------------------------


def test_log_loss_bracket_zero_on_certainty(prov, ids_ts):
    """Forecast = one-hot on the realized bin → log-loss = 0."""
    from bracketlearn.adapters import BracketLadder
    ids, ts = ids_ts(3)
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    # y = 0.5 (bin 0), 1.5 (bin 1), 2.5 (bin 2). Forecast one-hot:
    probs = np.eye(3)
    d = DistributionForecast.from_brackets(
        edges=edges, probs=probs,
        ids=ids, timestamps=ts, provenance=prov,
    )
    ladder = BracketLadder(edges_per_row=[edges] * 3)
    contracts = ladder.price(d)
    y = np.array([0.5, 1.5, 2.5])
    loss = log_loss_bracket(contracts, edges, y)
    assert loss < 1e-9


def test_brier_bracket_uniform_baseline(prov, ids_ts):
    """Uniform forecast over B bins: Brier = (B−1)/B² + (B−1)·(1/B)²
                                          = 2·(B−1)/B²."""
    from bracketlearn.adapters import BracketLadder
    ids, ts = ids_ts(100)
    B = 5
    edges = np.linspace(0.0, 5.0, B + 1)
    probs = np.full((100, B), 1.0 / B)
    d = DistributionForecast.from_brackets(
        edges=edges, probs=probs,
        ids=ids, timestamps=ts, provenance=prov,
    )
    ladder = BracketLadder(edges_per_row=[edges] * 100)
    contracts = ladder.price(d)
    # Realized bin: spread evenly. The Brier per row when realized is in some
    # bin b: (1−1/B)² + (B−1)·(1/B)² = (B−1)/B.
    y = np.full(100, 2.5)               # bin 2
    brier = brier_bracket(contracts, edges, y)
    np.testing.assert_allclose(brier, (B - 1) / B, atol=1e-10)


def test_crps_bracket_matches_normal_discretisation(prov, ids_ts):
    """Discretise a Gaussian onto a fine ladder; CRPS_bracket should approach
    CRPS_gaussian as the ladder gets finer."""
    ids, ts = ids_ts(1)
    mu, sigma = 0.0, 1.0
    d_normal = DistributionForecast.from_normal(
        mu=np.array([mu]), sigma=np.array([sigma]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = np.array([0.5])
    crps_truth = crps_gaussian(d_normal, y)[0]

    # Fine ladder covering [μ-6σ, μ+6σ] in 200 bins.
    edges = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 201)
    probs = d_normal.cdf(edges[1:]) - d_normal.cdf(edges[:-1])
    probs = probs / probs.sum(axis=1, keepdims=True)
    d_bracket = DistributionForecast.from_brackets(
        edges=edges, probs=probs,
        ids=ids, timestamps=ts, provenance=prov,
    )
    crps_disc = crps_bracket(d_bracket, y)[0]
    # Discretisation error should be tiny on a 200-bin grid.
    assert abs(crps_disc - crps_truth) < 0.01


# ---------------------------------------------------------------------------
# Quantile CRPS.
# ---------------------------------------------------------------------------


def test_crps_quantile_matches_gaussian_on_dense_grid(prov, ids_ts):
    """Build a quantile-backed dist from a dense Gaussian quantile grid;
    pinball-CRPS should approach closed-form Gaussian CRPS."""
    ids, ts = ids_ts(1)
    from scipy.stats import norm
    mu, sigma = 0.0, 1.0
    d_normal = DistributionForecast.from_normal(
        mu=np.array([mu]), sigma=np.array([sigma]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = np.array([0.7])
    crps_truth = crps_gaussian(d_normal, y)[0]

    taus = np.linspace(0.005, 0.995, 199)
    qvals = norm.ppf(taus, loc=mu, scale=sigma)[None, :]
    d_q = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    crps_q = crps_quantile(d_q, y)[0]
    # Discretisation + tail-clipping error on 199 quantiles: ~few percent.
    assert abs(crps_q - crps_truth) < 0.05


# ---------------------------------------------------------------------------
# PIT — uniformity statistic.
# ---------------------------------------------------------------------------


def test_pit_uniform_mean_half(prov, ids_ts, rng):
    """Calibrated Gaussian forecast → PIT mean ≈ 0.5."""
    n = 5000
    ids, ts = ids_ts(n)
    y = rng.normal(0.0, 1.0, size=n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    p = pit(d, y)
    assert abs(p.mean() - 0.5) < 0.02


# ---------------------------------------------------------------------------
# cdf_at — per-row CDF (B9 fix: must equal np.diag(cdf(y)) without
# materialising the (N, N) cross product).
# ---------------------------------------------------------------------------


def test_cdf_at_matches_diag_for_normal(prov, ids_ts, rng):
    n = 50
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 2.0, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.normal(0, 1, n)
    np.testing.assert_allclose(d.cdf_at(y), np.diag(d.cdf(y)), atol=1e-10)


def test_cdf_at_matches_diag_for_bracket(prov, ids_ts, rng):
    n = 30
    B = 8
    ids, ts = ids_ts(n)
    edges = np.linspace(-2.0, 2.0, B + 1)
    raw = rng.uniform(0.1, 1.0, size=(n, B))
    probs = raw / raw.sum(axis=1, keepdims=True)
    d = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.uniform(-2.5, 2.5, n)  # mix of inside and outside support
    np.testing.assert_allclose(d.cdf_at(y), np.diag(d.cdf(y)), atol=1e-10)


def test_cdf_at_matches_diag_for_quantile(prov, ids_ts, rng):
    n = 20
    Q = 5
    ids, ts = ids_ts(n)
    taus = np.linspace(0.1, 0.9, Q)
    base = np.sort(rng.normal(0, 1, size=(n, Q)), axis=1)
    d = DistributionForecast.from_quantiles(
        taus=taus, qvals=base,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.normal(0, 1, n)
    np.testing.assert_allclose(d.cdf_at(y), np.diag(d.cdf(y)), atol=1e-10)


def test_cdf_at_matches_diag_for_mixture_normal(prov, ids_ts, rng):
    n = 25
    K = 3
    ids, ts = ids_ts(n)
    raw_w = rng.uniform(0.1, 1.0, size=(n, K))
    w = raw_w / raw_w.sum(axis=1, keepdims=True)
    mus = rng.normal(0, 1, size=(n, K))
    sigmas = rng.uniform(0.3, 1.5, size=(n, K))
    d = DistributionForecast.from_mixture_normal(
        weights=w, mus=mus, sigmas=sigmas,
        ids=ids, timestamps=ts, provenance=prov,
    )
    y = rng.normal(0, 1, n)
    np.testing.assert_allclose(d.cdf_at(y), np.diag(d.cdf(y)), atol=1e-10)


def test_cdf_at_rejects_wrong_length(prov, ids_ts):
    import pytest as _pytest
    n = 5
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    with _pytest.raises(ValueError, match="cdf_at"):
        d.cdf_at(np.zeros(n + 1))
