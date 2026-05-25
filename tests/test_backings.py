"""DistributionForecast backing invariants.

Each backing must satisfy:
- Construction rejects malformed input loudly (Rule #0.5).
- CDF is monotone non-decreasing in y.
- CDF → 0 at -inf, → 1 at +inf (or at bounded support edges).
- Bracket probs sum to 1; quantile qvals are non-decreasing in tau.
- PIT of well-calibrated forecasts is approximately uniform.

These tests pin the contract that downstream metric code relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.forecast import Backing, DistributionForecast, ParametricFamily
from bracketlearn.tail import TailPolicy, TailRule


# ---------------------------------------------------------------------------
# Normal backing
# ---------------------------------------------------------------------------


class TestFromNormal:
    def test_basic_construction(self, prov, ids_ts):
        ids, ts = ids_ts(3)
        d = DistributionForecast.from_normal(
            mu=np.array([0.0, 1.0, 2.0]),
            sigma=np.array([1.0, 1.0, 2.0]),
            ids=ids, timestamps=ts, provenance=prov,
        )
        assert d.backing == Backing.PARAMETRIC
        assert d.family == ParametricFamily.NORMAL
        assert d.params["mu"].tolist() == [0.0, 1.0, 2.0]

    def test_rejects_nonpositive_sigma(self, prov, ids_ts):
        ids, ts = ids_ts(2)
        with pytest.raises(ValueError, match="sigma"):
            DistributionForecast.from_normal(
                mu=np.array([0.0, 1.0]),
                sigma=np.array([1.0, 0.0]),     # zero σ forbidden
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_rejects_shape_mismatch(self, prov, ids_ts):
        ids, ts = ids_ts(3)
        with pytest.raises(ValueError, match="shape"):
            DistributionForecast.from_normal(
                mu=np.array([0.0, 1.0]),        # length 2 vs ids length 3
                sigma=np.array([1.0, 1.0]),
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_cdf_monotone(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        d = DistributionForecast.from_normal(
            mu=np.array([5.0]), sigma=np.array([2.0]),
            ids=ids, timestamps=ts, provenance=prov,
        )
        ys = np.linspace(-10, 20, 100)
        cdfs = d.cdf(ys)[0]
        # cdf returns (N, len(ys)); pick row 0.
        assert np.all(np.diff(cdfs) >= -1e-12)
        assert cdfs[0] < 0.01 and cdfs[-1] > 0.99


# ---------------------------------------------------------------------------
# Mixture-normal backing
# ---------------------------------------------------------------------------


class TestFromMixtureNormal:
    def test_basic_construction(self, prov, ids_ts):
        ids, ts = ids_ts(2)
        w = np.array([[0.5, 0.5], [0.3, 0.7]])
        mus = np.array([[0.0, 1.0], [2.0, 3.0]])
        sigmas = np.array([[1.0, 1.0], [1.0, 1.0]])
        d = DistributionForecast.from_mixture_normal(
            weights=w, mus=mus, sigmas=sigmas,
            ids=ids, timestamps=ts, provenance=prov,
        )
        assert d.family == ParametricFamily.MIXTURE_NORMAL

    def test_rejects_weights_not_summing_to_one(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        with pytest.raises(ValueError, match="sum to 1"):
            DistributionForecast.from_mixture_normal(
                weights=np.array([[0.3, 0.3]]),
                mus=np.array([[0.0, 1.0]]),
                sigmas=np.array([[1.0, 1.0]]),
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_rejects_negative_weights(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        with pytest.raises(ValueError, match="nonneg"):
            DistributionForecast.from_mixture_normal(
                weights=np.array([[-0.5, 1.5]]),
                mus=np.array([[0.0, 1.0]]),
                sigmas=np.array([[1.0, 1.0]]),
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_collapsed_mixture_matches_single_normal(self, prov, ids_ts):
        # All weight on one component → CDF/PDF should match a plain normal.
        ids, ts = ids_ts(1)
        d_mix = DistributionForecast.from_mixture_normal(
            weights=np.array([[1.0, 0.0]]),
            mus=np.array([[5.0, 100.0]]),       # second component irrelevant
            sigmas=np.array([[2.0, 1.0]]),
            ids=ids, timestamps=ts, provenance=prov,
        )
        d_normal = DistributionForecast.from_normal(
            mu=np.array([5.0]), sigma=np.array([2.0]),
            ids=ids, timestamps=ts, provenance=prov,
        )
        ys = np.linspace(-5, 15, 20)
        np.testing.assert_allclose(d_mix.cdf(ys), d_normal.cdf(ys), atol=1e-10)


# ---------------------------------------------------------------------------
# Quantile backing
# ---------------------------------------------------------------------------


class TestFromQuantiles:
    def test_basic_construction(self, prov, ids_ts):
        ids, ts = ids_ts(2)
        taus = np.array([0.1, 0.5, 0.9])
        qvals = np.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]])
        d = DistributionForecast.from_quantiles(
            taus=taus, qvals=qvals,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=ids, timestamps=ts, provenance=prov,
        )
        assert d.backing == Backing.QUANTILE

    def test_requires_tail_policy(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        # tail_policy is a required kwarg per the constructor signature —
        # omitting it should fail at the Python call layer.
        with pytest.raises(TypeError):
            DistributionForecast.from_quantiles(   # type: ignore[call-arg]
                taus=np.array([0.1, 0.9]),
                qvals=np.array([[0.0, 1.0]]),
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_rejects_non_monotone_qvals(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        with pytest.raises(ValueError, match="monotone|increas|decreas"):
            DistributionForecast.from_quantiles(
                taus=np.array([0.1, 0.5, 0.9]),
                qvals=np.array([[0.0, 2.0, 1.0]]),    # crossed quantiles
                tail_policy=TailPolicy.same(TailRule.clip()),
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_cdf_monotone(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        d = DistributionForecast.from_quantiles(
            taus=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
            qvals=np.array([[0.0, 2.0, 5.0, 8.0, 10.0]]),
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=ids, timestamps=ts, provenance=prov,
        )
        ys = np.linspace(-5, 15, 100)
        cdfs = d.cdf(ys)[0]
        assert np.all(np.diff(cdfs) >= -1e-12)
        # clip policy: cdf hits 0 below q[0]=0 and 1 above q[-1]=10.
        assert cdfs[0] == 0.0 and cdfs[-1] == 1.0


# ---------------------------------------------------------------------------
# Bracket backing
# ---------------------------------------------------------------------------


class TestFromBrackets:
    def test_basic_construction(self, prov, ids_ts):
        ids, ts = ids_ts(2)
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        d = DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=ids, timestamps=ts, provenance=prov,
        )
        assert d.backing == Backing.BRACKET
        np.testing.assert_allclose(d.probs.sum(axis=1), 1.0)

    def test_rejects_probs_not_summing_to_one(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        with pytest.raises(ValueError, match="sum"):
            DistributionForecast.from_brackets(
                edges=np.array([0.0, 1.0, 2.0]),
                probs=np.array([[0.3, 0.3]]),   # sum 0.6
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_rejects_negative_probs(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        with pytest.raises(ValueError, match="nonneg|negative"):
            DistributionForecast.from_brackets(
                edges=np.array([0.0, 1.0, 2.0]),
                probs=np.array([[-0.1, 1.1]]),
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_rejects_non_monotone_edges(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        with pytest.raises(ValueError, match="monotone|increas|sort"):
            DistributionForecast.from_brackets(
                edges=np.array([0.0, 2.0, 1.0]),    # non-monotone
                probs=np.array([[0.5, 0.5]]),
                ids=ids, timestamps=ts, provenance=prov,
            )

    def test_cdf_step_at_edges(self, prov, ids_ts):
        ids, ts = ids_ts(1)
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([[0.2, 0.5, 0.3]])
        d = DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=ids, timestamps=ts, provenance=prov,
        )
        # CDF at edges should equal cumulative probs.
        cdfs = d.cdf(edges)[0]
        np.testing.assert_allclose(cdfs, [0.0, 0.2, 0.7, 1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# PIT calibration — well-calibrated forecasts produce uniform PIT.
# ---------------------------------------------------------------------------


def test_pit_uniform_on_calibrated_normal(prov, rng):
    """If y ~ N(μ, σ) and forecast = N(μ, σ), PIT should be ~Uniform[0, 1]."""
    from bracketlearn.score import pit

    n = 5000
    mu = np.zeros(n)
    sigma = np.ones(n)
    y = rng.normal(mu, sigma)
    d = DistributionForecast.from_normal(
        mu=mu, sigma=sigma,
        ids=np.arange(n), timestamps=np.arange(n, dtype=float),
        provenance=prov,
    )
    pits = pit(d, y)
    # PIT should be uniform: mean ≈ 0.5, std ≈ 1/√12 ≈ 0.289.
    assert abs(pits.mean() - 0.5) < 0.02
    assert abs(pits.std() - 1 / np.sqrt(12)) < 0.02
