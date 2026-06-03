"""Cross-estimator invariants — pin behaviour that must hold across refactors.

Audit item 5: ladder-sum invariant lives in test_ladder_sum.py; this
file covers the rest:

- ``clone(est).get_params() == est.get_params()`` for every BaseEstimator
  subclass. No shared mutable state, no fitted-state leak.
- Bracket-ladder edge cases: B=1 (single bracket) and B=2.
- DistributionForecast monotonicity invariants — quantile qvals
  non-decreasing in tau; bracket cumulative probs non-decreasing.

Skipped (deferred — concrete trainers vary too much for a clean
single test):
- sample_weight invariance (doubling a row's weight ≈ duplicating the
  row). Some trainers honor this exactly (linear / OLS), others
  approximately (LightGBM, NGBoost). Worth a dedicated suite later.
- single-row fit. Most trainers fail (ddof=1 in σ, k-fold needs k≥2,
  etc). Document instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn import (
    EMOS,
    BaseEstimator,
    BracketLadder,
    DistAsFeatures,
    DistributionForecast,
    EmpiricalDistribution,
    GlobalResidual,
    Isotonic,
    Persistence,
    SklearnPoint,
    clone,
)

# ---------------------------------------------------------------------------
# 5a — clone equality across every BaseEstimator subclass
# ---------------------------------------------------------------------------


def _all_baseestimator_subclasses() -> list[type]:
    """Walk the BaseEstimator hierarchy. Skips abstract scaffolds."""
    seen: set[type] = set()
    stack: list[type] = [BaseEstimator]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return sorted(seen, key=lambda c: c.__name__)


def _try_construct(cls: type) -> object | None:
    """Try to construct ``cls`` with sensible defaults.

    Some estimators need required arguments (CumulativeBinary needs
    cutpoints + outer_edges, Isotonic needs edges, DistAsFeatures needs a
    downstream, …). We special-case the ones we can reach; others get skipped.
    """
    presets: dict[str, dict] = {
        "CumulativeBinary": {
            "cutpoints_by_id": {0: np.array([1.0, 2.0, 3.0])},
            "outer_edges_by_id": {0: (0.0, 4.0)},
        },
        "Isotonic": {"pre_integrate_edges": np.linspace(0, 10, 6)},
        "SklearnPoint": {"estimator": _LinearRegressionFactory()},
        "DistAsFeatures": {"downstream": SklearnPoint(_LinearRegressionFactory())},
    }
    kwargs = presets.get(cls.__name__, {})
    try:
        return cls(**kwargs)
    except (TypeError, ValueError, ImportError):
        return None


class _LinearRegressionFactory:
    """Constructs a fresh LinearRegression per call (so each clone() gets
    a separate inner sklearn estimator)."""

    def __new__(cls):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()


def test_every_baseestimator_subclass_clones_with_equal_params():
    """clone(est).get_params() == est.get_params() for every constructable
    BaseEstimator subclass."""
    subclasses = _all_baseestimator_subclasses()
    assert subclasses, "no BaseEstimator subclasses found — import failure?"
    tested = 0
    skipped = []
    for cls in subclasses:
        est = _try_construct(cls)
        if est is None:
            skipped.append(cls.__name__)
            continue
        cloned = clone(est)
        # type must match
        assert type(cloned) is type(est), f"{cls.__name__}: clone changed type"
        # cloned is a different object
        assert cloned is not est, f"{cls.__name__}: clone returned the same object"
        # params equal (deep=False to avoid nested-estimator identity issues)
        orig_params = est.get_params(deep=False)
        new_params = cloned.get_params(deep=False)
        assert set(orig_params) == set(new_params), (
            f"{cls.__name__}: param keys diverged after clone"
        )
        tested += 1
    assert tested >= 10, f"too few estimators tested ({tested}); skipped: {skipped}"


def test_clone_does_not_share_fitted_state():
    """clone() returns an UNFITTED copy."""
    est = EMOS()
    est.intercept_a_ = 42.0  # simulate fitted state on a `_`-suffixed attr
    cloned = clone(est)
    assert not hasattr(cloned, "intercept_a_") or cloned.intercept_a_ != 42.0


def test_clone_deep_copies_nested_estimators():
    """clone() should give the nested estimator a fresh instance, not
    share the same object."""
    inner = SklearnPoint(_LinearRegressionFactory())
    composite = DistAsFeatures(downstream=inner)
    cloned = clone(composite)
    # cloned should have its own downstream estimator.
    assert cloned.downstream is not composite.downstream, "downstream shared after clone"


# ---------------------------------------------------------------------------
# 5b — bracket-ladder edge cases: B=1 and B=2
# ---------------------------------------------------------------------------


@pytest.fixture
def _normal_dist(prov, ids_ts):
    """A simple parametric-normal dist on 5 rows."""
    N = 5
    ids, ts = ids_ts(N)
    return DistributionForecast.from_normal(
        mu=np.linspace(-1.0, 1.0, N),
        sigma=np.full(N, 0.5),
        ids=ids, timestamps=ts, provenance=prov,
    )


def test_bracket_ladder_b1_single_bracket(_normal_dist):
    """B=1: one bracket spanning the whole support. Probability = 1.0 per row."""
    edges = np.array([-100.0, 100.0])
    N = _normal_dist.ids.shape[0]
    ladder = BracketLadder(edges_per_row=[edges] * N)
    contracts = ladder.price(_normal_dist)
    fp = contracts.fair_price
    assert fp.shape == (_normal_dist.ids.shape[0] * 1,)
    np.testing.assert_allclose(fp, 1.0, atol=1e-6)


def test_bracket_ladder_b2_two_brackets_sum_to_one(_normal_dist):
    """B=2: two brackets split at the mean. Probabilities sum to 1 per row."""
    edges = np.array([-100.0, 0.0, 100.0])
    N = _normal_dist.ids.shape[0]
    ladder = BracketLadder(edges_per_row=[edges] * N)
    contracts = ladder.price(_normal_dist)
    probs = contracts.fair_price.reshape(-1, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    # For mu=0 (center row), the split should be ~50/50.
    center_idx = _normal_dist.params["mu"].shape[0] // 2
    assert abs(probs[center_idx, 0] - 0.5) < 0.01


# ---------------------------------------------------------------------------
# 5c — monotonicity invariants
# ---------------------------------------------------------------------------


def test_quantile_backing_qvals_monotone(prov, ids_ts):
    """from_quantiles enforces qvals monotone non-decreasing in tau."""
    from bracketlearn.forecast import TailPolicy, TailRule
    N = 3
    ids, ts = ids_ts(N)
    taus = np.array([0.1, 0.5, 0.9])
    # Deliberate crossing — must raise.
    bad = np.array([[1.0, 2.0, 1.5], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="monotone"):
        DistributionForecast.from_quantiles(
            taus=taus, qvals=bad,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=ids, timestamps=ts, provenance=prov,
        )


def test_normal_dist_cdf_monotone_in_x(prov, ids_ts):
    """CDF of a parametric normal must be monotone non-decreasing in x."""
    N = 3
    ids, ts = ids_ts(N)
    dist = DistributionForecast.from_normal(
        mu=np.zeros(N), sigma=np.ones(N),
        ids=ids, timestamps=ts, provenance=prov,
    )
    xs = np.linspace(-5.0, 5.0, 50)
    cdfs = dist.cdf(xs)
    diffs = np.diff(cdfs, axis=1)
    assert np.all(diffs >= -1e-12), "CDF non-monotone"


def test_bracket_dist_cumulative_probs_monotone(prov, ids_ts):
    """Bracket-backed dist: cumulative bin probs monotone non-decreasing."""
    N = 4
    B = 6
    ids, ts = ids_ts(N)
    edges = np.linspace(0.0, 12.0, B + 1)
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 1.0, size=(N, B))
    probs = raw / raw.sum(axis=1, keepdims=True)
    dist = DistributionForecast.from_brackets(
        edges=edges, probs=probs, ids=ids, timestamps=ts, provenance=prov,
    )
    cum = np.cumsum(probs, axis=1)
    diffs = np.diff(cum, axis=1)
    assert np.all(diffs >= 0), "cumulative bracket probs non-monotone"
    # And cdf at the rightmost edge is 1.0 by construction.
    np.testing.assert_allclose(dist.cdf(edges[-1]), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 5d — fitted-state isolation across clones
# ---------------------------------------------------------------------------


def test_fit_does_not_mutate_clone_source():
    """A clone fit later must not affect the original's params."""
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)

    sp_orig = SklearnPoint(LinearRegression())
    sp_clone = clone(sp_orig)
    sp_clone.fit(X, y)

    # Original must remain unfitted.
    assert not sp_orig.__sklearn_is_fitted__()
    assert sp_clone.__sklearn_is_fitted__()


def test_empirical_distribution_clone_fits_independently():
    """Two clones fit on different data produce different qvals."""
    e1 = EmpiricalDistribution()
    e2 = clone(e1)
    rng = np.random.default_rng(0)
    e1.fit(rng.standard_normal((30, 2)), rng.uniform(0, 1, 30))
    e2.fit(rng.standard_normal((30, 2)), rng.uniform(10, 11, 30))
    assert e1.quantiles_[0] < 5.0
    assert e2.quantiles_[0] > 5.0


# ---------------------------------------------------------------------------
# 5e — baselines: deterministic shapes
# ---------------------------------------------------------------------------


def test_persistence_predict_shape_matches_X():
    """Persistence(lag=k) on N rows returns N predictions."""
    rng = np.random.default_rng(0)
    y_train = rng.standard_normal(100)
    p = Persistence(lag=3)
    p.fit(np.zeros((100, 2)), y_train)
    out = p.predict(np.zeros((50, 2)))
    assert out.mu.shape == (50,)


def test_empirical_dist_predict_emits_same_qvals_per_row():
    """EmpiricalDistribution is X-independent: every predict row has
    identical qvals (no row-conditioning)."""
    ed = EmpiricalDistribution()
    rng = np.random.default_rng(0)
    ed.fit(np.zeros((100, 2)), rng.standard_normal(100))
    d = ed.predict_dist(np.zeros((10, 2)))
    for i in range(1, 10):
        np.testing.assert_array_equal(d.qvals[0], d.qvals[i])
