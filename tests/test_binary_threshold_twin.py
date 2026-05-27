"""Single-strike adapters — BinaryAbove, BinaryBelow, Twin, ThresholdLadder.

These map directly onto prediction-market contracts: above/below thresholds
(Kalshi single-threshold contracts), paired YES/NO at a strike (Polymarket
spread / total contracts), and a survival-function ladder (Kalshi
"above k_1 / above k_2 / above k_3 …" markets).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as _stats

from bracketlearn.adapters import BinaryAbove, BinaryBelow, ThresholdLadder, Twin
from bracketlearn.forecast import DistributionForecast

# ---------------------------------------------------------------------------
# BinaryAbove / BinaryBelow.
# ---------------------------------------------------------------------------


def test_binary_above_known_answer_normal(prov, ids_ts):
    n = 4
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.array([0.0, 1.0, -1.0, 2.0]), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    cf = BinaryAbove(strike=0.5).price(d)
    assert cf.fair_price.shape == (n,)
    expected = 1.0 - _stats.norm.cdf(0.5, loc=[0.0, 1.0, -1.0, 2.0], scale=1.0)
    np.testing.assert_allclose(cf.fair_price, expected, atol=1e-12)
    # One contract per entity.
    np.testing.assert_array_equal(cf.entity_ids, ids)
    np.testing.assert_array_equal(cf.contract_ids, np.zeros(n, dtype=int))
    assert cf.contract_spec.kind == "binary_above"


def test_binary_below_known_answer_normal(prov, ids_ts):
    n = 3
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.array([0.0, 1.0, -1.0]), sigma=np.array([0.5, 1.0, 2.0]),
        ids=ids, timestamps=ts, provenance=prov,
    )
    cf = BinaryBelow(strike=0.0).price(d)
    expected = _stats.norm.cdf(0.0, loc=[0.0, 1.0, -1.0], scale=[0.5, 1.0, 2.0])
    np.testing.assert_allclose(cf.fair_price, expected, atol=1e-12)
    assert cf.contract_spec.kind == "binary_below"


def test_binary_above_plus_below_sum_to_one(prov, ids_ts, rng):
    """For any strike, P(X > k) + P(X ≤ k) = 1."""
    n = 10
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 2.0, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    k = 0.3
    above = BinaryAbove(strike=k).price(d)
    below = BinaryBelow(strike=k).price(d)
    np.testing.assert_allclose(above.fair_price + below.fair_price, 1.0, atol=1e-12)


def test_binary_above_works_for_quantile_backing(prov, ids_ts, rng):
    from bracketlearn.forecast import TailPolicy, TailRule

    n, Q = 8, 5
    ids, ts = ids_ts(n)
    taus = np.linspace(0.1, 0.9, Q)
    qvals = np.sort(rng.normal(0, 1, size=(n, Q)), axis=1)
    d = DistributionForecast.from_quantiles(
        taus=taus, qvals=qvals,
        tail_policy=TailPolicy.same(TailRule.clip()),
        ids=ids, timestamps=ts, provenance=prov,
    )
    cf = BinaryAbove(strike=0.0).price(d)
    # Reference: per-row interp on the qvals/taus grid.
    expected = np.array([
        1.0 - np.interp(0.0, qvals[i], taus, left=0.0, right=1.0) for i in range(n)
    ])
    np.testing.assert_allclose(cf.fair_price, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Twin — paired YES/NO at a strike.
# ---------------------------------------------------------------------------


def test_twin_pair_sums_to_one_per_entity(prov, ids_ts, rng):
    n = 6
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 1.5, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    cf = Twin(strike=0.5).price(d)
    # 2 contracts per entity → shape (2N,).
    assert cf.fair_price.shape == (2 * n,)
    for i in range(n):
        block = cf.fair_price[cf.entity_ids == ids[i]]
        np.testing.assert_allclose(block.sum(), 1.0, atol=1e-12)
    # contract_ids interleaved 0,1,0,1,…
    np.testing.assert_array_equal(
        cf.contract_ids, np.tile(np.array([0, 1]), n)
    )
    # YES = 1 - cdf(k) at contract_id=0.
    yes_rows = cf.contract_ids == 0
    expected_yes = 1.0 - _stats.norm.cdf(
        0.5, loc=d.params["mu"], scale=d.params["sigma"]
    )
    np.testing.assert_allclose(cf.fair_price[yes_rows], expected_yes, atol=1e-12)


def test_twin_group_id_pairs(prov, ids_ts):
    """Both legs of a Twin share group_id = entity_id."""
    n = 3
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    cf = Twin(strike=0.0).price(d)
    # For each entity i, both rows must have group_id == i.
    for i in ids:
        group_rows = cf.group_id[cf.entity_ids == i]
        assert group_rows.shape == (2,)
        assert (group_rows == i).all()


# ---------------------------------------------------------------------------
# ThresholdLadder — survival function at S strikes.
# ---------------------------------------------------------------------------


def test_threshold_ladder_monotone_decreasing(prov, ids_ts, rng):
    """Survival values must decrease as the strike increases."""
    n, S = 5, 4
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 1.5, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    strikes = np.array([-1.0, 0.0, 1.0, 2.0])
    cf = ThresholdLadder(strikes=strikes).price(d)
    assert cf.fair_price.shape == (n * S,)
    for i in range(n):
        block = cf.fair_price[cf.entity_ids == ids[i]]
        # Each entity's S-vector is non-increasing.
        assert np.all(np.diff(block) <= 1e-12)
    # Spot-check row 0 known answer.
    expected_row0 = 1.0 - _stats.norm.cdf(
        strikes, loc=d.params["mu"][0], scale=d.params["sigma"][0]
    )
    np.testing.assert_allclose(
        cf.fair_price[cf.entity_ids == ids[0]], expected_row0, atol=1e-12
    )


def test_threshold_ladder_rejects_non_monotone_strikes(prov, ids_ts):
    n = 2
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=np.zeros(n), sigma=np.ones(n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    with pytest.raises(ValueError, match="monotone"):
        ThresholdLadder(strikes=np.array([1.0, 0.0])).price(d)


def test_threshold_ladder_matches_binary_above_per_strike(prov, ids_ts, rng):
    """ThresholdLadder is just BinaryAbove vectorised over strikes."""
    n = 4
    ids, ts = ids_ts(n)
    d = DistributionForecast.from_normal(
        mu=rng.normal(0, 1, n), sigma=rng.uniform(0.5, 1.5, n),
        ids=ids, timestamps=ts, provenance=prov,
    )
    strikes = np.array([-0.5, 0.5, 1.5])
    ladder = ThresholdLadder(strikes=strikes).price(d)
    for s_idx, k in enumerate(strikes):
        single = BinaryAbove(strike=float(k)).price(d)
        ladder_at_s = ladder.fair_price[ladder.contract_ids == s_idx]
        np.testing.assert_allclose(ladder_at_s, single.fair_price, atol=1e-12)
