"""Per-row affine reparametrization of DistributionForecast (`affine`).

The invariance the normalization design rests on: under the per-row map
``v ↦ v·s + c`` (s>0), a forecast's CDF satisfies
``dist.affine(c, s).cdf_at(y) == dist.cdf_at((y − c) / s)`` and bracket
probabilities are unchanged when the edges are mapped by the same affine —
so a z-space forecast integrated over z-edges == the °F forecast over °F edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.forecast import (
    BracketForecast,
    DistributionForecast,
    MixtureNormalForecast,
    NormalForecast,
    QuantileForecast,
    StudentTForecast,
    TailPolicy,
    TailRule,
)
from bracketlearn.forecast._meta import ProvenanceMeta

N = 4
IDS = np.arange(N)
TS = np.zeros(N)
PROV = ProvenanceMeta.placeholder("test_affine")
C = np.array([10.0, -5.0, 60.0, 0.5])      # per-row shift (climo-like)
S = np.array([2.0, 7.0, 1.5, 11.0])        # per-row scale (σ_station-like, >0)
PROBE = np.array([0.3, -1.2, 2.1, 0.0])    # a point per row, in z-space


def _normal():
    return NormalForecast.from_arrays(
        mu=np.array([0.1, -0.5, 1.0, 0.2]), sigma=np.array([1.0, 2.0, 0.5, 3.0]),
        ids=IDS, timestamps=TS, provenance=PROV,
    )


def _student_t():
    return StudentTForecast.from_arrays(
        mu=np.array([0.0, 0.3, -0.4, 1.1]), sigma=np.array([1.0, 1.5, 0.7, 2.0]),
        df=np.full(N, 5.0), ids=IDS, timestamps=TS, provenance=PROV,
    )


def _mixture():
    return MixtureNormalForecast.from_arrays(
        weights=np.tile([0.4, 0.6], (N, 1)),
        mus=np.array([[-0.5, 0.5], [0.0, 1.0], [-1.0, 0.2], [0.3, 0.9]]),
        sigmas=np.array([[1.0, 0.8], [1.2, 0.9], [0.7, 1.1], [2.0, 1.0]]),
        ids=IDS, timestamps=TS, provenance=PROV,
    )


def _quantile():
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    base = np.linspace(-2.0, 2.0, taus.size)
    qvals = np.stack([base + shift for shift in (0.0, 0.5, -0.3, 1.0)])
    return QuantileForecast.from_arrays(
        taus=taus, qvals=qvals, tail_policy=TailPolicy.same(TailRule.clip()),
        ids=IDS, timestamps=TS, provenance=PROV,
    )


CONTINUOUS = {
    "normal": _normal, "student_t": _student_t,
    "mixture": _mixture, "quantile": _quantile,
}


@pytest.mark.parametrize("name", list(CONTINUOUS))
def test_affine_cdf_invariance(name):
    dist = CONTINUOUS[name]()
    y = PROBE * S + C                         # affine image of PROBE
    got = dist.affine(C, S).cdf_at(y)
    want = dist.cdf_at(PROBE)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_affine_bracket_prob_invariance():
    """Normal integrated over °F edges == affine-Normal over affined edges."""
    dist = _normal()
    edges_z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])          # shared z-edges
    probs_z = dist.integrate(edges_z).probs
    # Map edges per row by the same affine, integrate the affined dist.
    edges_aff = edges_z[None, :] * S[:, None] + C[:, None]   # (N, B+1)
    probs_aff = dist.affine(C, S).integrate(edges_aff).probs
    np.testing.assert_allclose(probs_aff, probs_z, atol=1e-9)


def test_affine_requires_positive_finite_scale():
    dist = _normal()
    with pytest.raises(ValueError, match="strictly positive"):
        dist.affine(C, np.array([1.0, -1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        dist.affine(0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        dist.affine(np.inf, 1.0)


def test_affine_bracket_forecast_maps_edges_keeps_probs_and_nan_padding():
    # Ragged rows: row 0 has 3 bins (4 edges), row 1 has 2 bins (3 edges,
    # 1 NaN-padded edge + 1 NaN-padded prob).
    edges = np.array([[-2.0, -0.5, 0.5, 2.0],
                      [-1.0, 0.0, 1.0, np.nan]])
    probs = np.array([[0.2, 0.5, 0.3],
                      [0.6, 0.4, np.nan]])
    bf = BracketForecast.from_arrays(
        edges=edges, probs=probs, ids=np.arange(2), timestamps=np.zeros(2),
        provenance=PROV,
    )
    c = np.array([10.0, 60.0])
    s = np.array([2.0, 1.5])
    out = bf.affine(c, s)
    exp_edges = edges * s[:, None] + c[:, None]   # NaN·s + c == NaN
    np.testing.assert_allclose(out.edges, exp_edges, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(out.probs, probs, atol=1e-9, equal_nan=True)  # mass unchanged
    # CDF invariance at an interior probe per row.
    probe = np.array([0.0, 0.5])
    np.testing.assert_allclose(
        out.cdf_at(probe * s + c), bf.cdf_at(probe), atol=1e-9,
    )


def test_affine_scalar_broadcast_matches_per_row():
    dist = _normal()
    per_row = dist.affine(np.full(N, 3.0), np.full(N, 2.0))
    scalar = dist.affine(3.0, 2.0)
    np.testing.assert_allclose(scalar.mu, per_row.mu)
    np.testing.assert_allclose(scalar.sigma, per_row.sigma)
