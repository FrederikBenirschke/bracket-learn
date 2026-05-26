"""Negative tests for loud-failure fixes.

Every test here verifies that a previously-silent fallback now raises
loudly. If any of these regress to silent behaviour, the test fails.

Covers:
- Stacking row-alignment + sigma fallback
- CumulativeBinary outer_edges required
- RNNHourly unknown station IDs
- SklearnPoint sample_weight introspection
- adapter stubs raise NotImplementedError
- Isotonic / _bracket_probs_from_dist raise on zero row-sum
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from bracketlearn.adapters import (
    BinaryAbove,
    BinaryBelow,
    Bracket,
    Custom,
    LinearCombo,
    PerRow,
    ThresholdLadder,
    Twin,
    VanillaCall,
    VanillaPut,
    VenueSpec,
    to_quote,
)
from bracketlearn.forecast import (
    ContractForecast,
    ContractSpec,
    DistributionForecast,
    ProvenanceMeta,
)


def _prov() -> ProvenanceMeta:
    return ProvenanceMeta(
        forecaster_name="t",
        forecaster_version="0",
        fit_window=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
        fold_idx=None,
        calibration_set_hash=None,
        random_seed=0,
        code_sha="t",
        feature_matrix_hash="t",
        created_at=datetime(2024, 1, 1),
    )


# ---------------------------------------------------------------------------
# B8 — stubs raise NotImplementedError (not silently return None).
# ---------------------------------------------------------------------------


def test_adapter_stubs_raise_not_implemented():
    """Every unimplemented adapter.price now raises NotImplementedError."""
    # A no-op dist that supports .ids attr — these will raise before touching it.
    dist = None
    for adapter in [
        BinaryAbove(strike=1.0),
        BinaryBelow(strike=1.0),
        Bracket(lo=0.0, hi=1.0),
        ThresholdLadder(strikes=np.array([1.0])),
        Twin(strike=1.0),
        VanillaCall(strike=1.0),
        VanillaPut(strike=1.0),
        LinearCombo(parts=[(1.0, BinaryAbove(strike=1.0))]),
        Custom(payoff_fn=lambda x: x, support_lo=0.0, support_hi=1.0),
    ]:
        with pytest.raises(NotImplementedError):
            adapter.price(dist)


def test_to_quote_raises_not_implemented():
    cf = ContractForecast(
        contract_ids=np.array([0]),
        entity_ids=np.array([0]),
        timestamps=np.array([0.0]),
        fair_price=np.array([0.5]),
        group_id=np.array([0]),
        contract_spec=ContractSpec(kind="binary"),
        provenance=_prov(),
    )
    with pytest.raises(NotImplementedError):
        to_quote(cf, VenueSpec(venue="venue_a", ticker="X"))


def test_dist_conversion_stubs_raise():
    dist = DistributionForecast.from_normal(
        mu=np.array([0.0]), sigma=np.array([1.0]),
        ids=np.array([0]), timestamps=np.array([0.0]),
        provenance=_prov(),
    )
    with pytest.raises(NotImplementedError):
        dist.to_quantiles(np.array([0.5]))
    with pytest.raises(NotImplementedError):
        dist.to_brackets(np.array([0.0, 1.0]))
    with pytest.raises(NotImplementedError):
        dist.to_normal()
    with pytest.raises(NotImplementedError):
        dist.is_lossless_to(dist.backing)


def test_from_empirical_stub_raises():
    from bracketlearn.tail import TailPolicy, TailRule
    with pytest.raises(NotImplementedError):
        DistributionForecast.from_empirical(
            members=np.zeros((2, 3)),
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.array([0, 1]),
            timestamps=np.array([0.0, 1.0]),
            provenance=_prov(),
        )


# ---------------------------------------------------------------------------
# B6 — SklearnPoint introspects fit signature instead of swallowing TypeError.
# ---------------------------------------------------------------------------


def test_sklearn_point_introspects_sample_weight_signature():
    """An estimator without sample_weight in its signature gets called
    without weights; the TypeError that would have fired silently in
    v0.1 never occurs because we don't pass the kwarg in the first place.
    """
    from bracketlearn.trainers import SklearnPoint, _estimator_accepts_sample_weight

    class NoWeightFit:
        def fit(self, X, y):
            self.fitted = True
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

    est = NoWeightFit()
    assert _estimator_accepts_sample_weight(est) is False
    # Pipeline-style call with weights should NOT raise (signature check
    # decides to drop them).
    sp = SklearnPoint(estimator=est)
    X = np.random.default_rng(0).standard_normal((10, 3))
    y = np.zeros(10)
    sp.fit(X, y, sample_weight=np.ones(10))
    assert est.fitted


def test_sklearn_point_raises_genuine_typeerror_inside_fit():
    """If the estimator's fit DOES accept sample_weight but raises a
    TypeError for an unrelated reason (e.g. wrong dtype), v0.1 would
    have silently retried without weights and produced different
    output. v0.2 lets the original error propagate.
    """
    from bracketlearn.trainers import SklearnPoint

    class BadFit:
        def fit(self, X, y, sample_weight=None):
            raise TypeError("bad dtype somewhere deep")

        def predict(self, X):
            return np.zeros(X.shape[0])

    sp = SklearnPoint(estimator=BadFit())
    with pytest.raises(TypeError, match="bad dtype"):
        sp.fit(np.zeros((3, 2)), np.zeros(3), sample_weight=np.ones(3))


# ---------------------------------------------------------------------------
# B3 — CumulativeBinary requires explicit outer_edges.
# ---------------------------------------------------------------------------


def test_cumulative_binary_requires_outer_edges_param():
    """outer_edges is a required constructor argument (no invented pad)."""
    pytest.importorskip("lightgbm")
    from bracketlearn.trainers import CumulativeBinary
    with pytest.raises(TypeError, match="outer_edges"):
        CumulativeBinary(cutpoints=np.array([1.0, 2.0]))  # type: ignore[call-arg]


def test_cumulative_binary_rejects_inside_outer_edges():
    pytest.importorskip("lightgbm")
    from bracketlearn.trainers import CumulativeBinary
    with pytest.raises(ValueError, match="outer_edges"):
        CumulativeBinary(cutpoints=np.array([1.0, 2.0]), outer_edges=(1.5, 3.0))
    with pytest.raises(ValueError, match="outer_edges"):
        CumulativeBinary(cutpoints=np.array([1.0, 2.0]), outer_edges=(0.0, 1.5))


# ---------------------------------------------------------------------------
# B5 — RNNHourly raises on unknown station IDs (no clip).
# ---------------------------------------------------------------------------


def test_rnn_hourly_raises_on_unknown_station_ids():
    pytest.importorskip("torch")
    from bracketlearn.trainers import RNNHourly

    rng = np.random.default_rng(0)
    N, H, C = 20, 24, 3
    X = rng.standard_normal((N, H, C)).astype(np.float32)
    y = X[:, :, 0].max(axis=1).astype(float) + rng.standard_normal(N) * 0.1
    ids = np.arange(N)
    ts = ids.astype(float)
    stations = np.zeros(N, dtype=np.int64)  # only station 0 in training
    rnn = RNNHourly(hidden=4, epochs=2, batch_size=16)
    rnn.fit(X, y, station_ids=stations)

    # Predict with an unknown station ID — must raise.
    unknown = np.array([0, 0, 5], dtype=np.int64)  # 5 was not in training
    with pytest.raises(ValueError, match="trained range"):
        rnn.predict(X[:3], ids=ids[:3], timestamps=ts[:3], station_ids=unknown)


# ---------------------------------------------------------------------------
# B10 — Isotonic + _bracket_probs_from_dist raise on zero row-sum.
# ---------------------------------------------------------------------------


def test_bracket_probs_from_dist_raises_on_zero_row_sum():
    """Bracket grid that lies entirely outside the distribution support."""
    from bracketlearn.lift import _bracket_probs_from_dist
    dist = DistributionForecast.from_normal(
        mu=np.array([0.0]), sigma=np.array([0.001]),
        ids=np.array([0]), timestamps=np.array([0.0]),
        provenance=_prov(),
    )
    # Edges far from the dist's mass: cdf_hi - cdf_lo ≈ 0 everywhere.
    edges = np.array([100.0, 110.0, 120.0])
    with pytest.raises(ValueError, match="zero total bracket mass"):
        _bracket_probs_from_dist(dist, edges)


# ---------------------------------------------------------------------------
# B2 — Stacking row-alignment + sigma fallback.
# ---------------------------------------------------------------------------


def test_stacking_raises_on_misaligned_upstream_ids():
    from bracketlearn.trainers import Stacking

    N = 8
    ids = np.arange(N)
    ts = ids.astype(float)
    # Two upstream dists with mismatched .ids vectors.
    d1 = DistributionForecast.from_normal(
        mu=np.zeros(N), sigma=np.ones(N), ids=ids, timestamps=ts, provenance=_prov(),
    )
    d2 = DistributionForecast.from_normal(
        mu=np.zeros(N), sigma=np.ones(N), ids=ids[::-1], timestamps=ts, provenance=_prov(),
    )
    stack = Stacking(deps=("a", "b"))
    with pytest.raises(ValueError, match="does not match"):
        stack.fit(np.zeros((N, 2)), np.zeros(N), deps_oof={"a": d1, "b": d2})


def test_stacking_raises_on_degenerate_sigma():
    """sigma_<=0 was silently floored to 1e-3 in v0.1. Now it raises."""
    from bracketlearn.trainers import Stacking

    N = 5
    ids = np.arange(N)
    ts = ids.astype(float)
    # Upstream μ perfectly equals y → meta-OLS residuals all zero → sigma=0.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = DistributionForecast.from_normal(
        mu=y.copy(), sigma=np.ones(N), ids=ids, timestamps=ts, provenance=_prov(),
    )
    stack = Stacking(deps=("a",))
    with pytest.raises(ValueError, match="degenerate"):
        stack.fit(np.zeros((N, 2)), y, deps_oof={"a": d})
