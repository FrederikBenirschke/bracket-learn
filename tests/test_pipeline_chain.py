"""Pipeline (flat sequential chain) — identity reproduction + normalization.

- `Pipeline([EMOS()])` and `Pipeline([IdentityTransformer(), EMOS()])` must
  reproduce a bare `EMOS()` bit-for-bit (the chain is inert without a real
  transformer).
- `Pipeline([GroupByZScore(...), EMOS()])` fits in z-space and maps the
  forecast back to the original scale; equals the hand-rolled
  normalize→fit→inverse path.
- construction validates stage ordering / unsupported kinds.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn import EMOS, GroupByZScore, IdentityTransformer, Pipeline


class _PointStub:
    """Minimal PointForecaster-shaped stage (has predict, no predict_dist)."""
    name = "point_stub"

    def fit(self, X, y, **kw):
        return self

    def predict(self, X, *, ids, timestamps):
        raise NotImplementedError


def _synthetic(n=60, k=4, seed=0):
    rng = np.random.default_rng(seed)
    members = rng.normal(50.0, 5.0, size=(n, k))   # ensemble members (°F-ish)
    y = members.mean(axis=1) + rng.normal(0, 2.0, size=n)
    ids = np.arange(n)
    ts = np.arange(n, dtype=float)
    return members, y, ids, ts


def test_pipeline_single_dist_reproduces_bare_forecaster():
    X, y, ids, ts = _synthetic()
    bare = EMOS().fit(X, y).predict_dist(X, ids=ids, timestamps=ts)
    piped = Pipeline([EMOS()]).fit(X, y, ids=ids, timestamps=ts).predict_dist(
        X, ids=ids, timestamps=ts,
    )
    np.testing.assert_array_equal(piped.mu, bare.mu)
    np.testing.assert_array_equal(piped.sigma, bare.sigma)


def test_identity_transformer_is_inert():
    X, y, ids, ts = _synthetic()
    bare = EMOS().fit(X, y).predict_dist(X, ids=ids, timestamps=ts)
    piped = (
        Pipeline([IdentityTransformer(), EMOS()])
        .fit(X, y, ids=ids, timestamps=ts)
        .predict_dist(X, ids=ids, timestamps=ts)
    )
    np.testing.assert_array_equal(piped.mu, bare.mu)
    np.testing.assert_array_equal(piped.sigma, bare.sigma)


def test_groupbyzscore_pipeline_matches_manual_normalize_fit_inverse():
    X, y, ids, ts = _synthetic()
    # per-station groups + a per-row center (climo-like)
    stations = np.array([f"S{i % 3}" for i in range(len(y))])
    center = np.where(stations == "S0", 48.0, np.where(stations == "S1", 52.0, 55.0))

    # Pipeline path
    pipe = Pipeline([GroupByZScore(), EMOS()], name="emos_norm")
    pipe.fit(X, y, ids=stations, timestamps=ts, center=center)
    got = pipe.predict_dist(X, ids=stations, timestamps=ts, center=center)

    # Manual path: same transform, fit in z, inverse via affine.
    gz = GroupByZScore().fit(X, y, ids=stations, center=center)
    Xz = gz.transform(X, ids=stations, center=center)
    yz = gz.transform_target(y)
    dz = EMOS().fit(Xz, yz).predict_dist(Xz, ids=stations, timestamps=ts)
    want = gz.inverse_dist(dz)

    np.testing.assert_allclose(got.mu, want.mu, atol=1e-9)
    np.testing.assert_allclose(got.sigma, want.sigma, atol=1e-9)
    assert pipe.name == "emos_norm"


def test_pipeline_rejects_unsupported_and_misordered_stages():
    with pytest.raises(ValueError, match="at least one stage"):
        Pipeline([])
    with pytest.raises(ValueError, match="forecaster"):
        Pipeline([GroupByZScore()])                       # no forecaster
    with pytest.raises(ValueError, match="following Lifter"):
        Pipeline([_PointStub()])                          # point with no lifter
    with pytest.raises(ValueError, match="one core forecaster"):
        Pipeline([EMOS(), EMOS()])                        # two cores
    with pytest.raises(ValueError, match="must precede the forecaster"):
        Pipeline([EMOS(), GroupByZScore()])               # transformer after model
