"""Tests for bracketlearn.component_lift: the paper's §4.2 step-0.

The substantive tests assert PARAMETER RECOVERY: given components built
with known (a, b, σ), the fit must return them. A lift that cannot recover
a planted slope cannot be trusted to report that a real vendor has one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bracketlearn.component_lift import (
    _NU_GRID,
    AffineNormal,
    ComponentFit,
)


@pytest.fixture
def planted():
    """Three components with deliberately different (a, b, σ).

    Component 1 is unbiased and sharp; 2 has a shift; 3 OVER-REACTS
    (b_true < 1 when regressing y on x), the failure an additive de-bias
    cannot correct.
    """
    rng = np.random.default_rng(17)
    n = 3000
    truth = rng.normal(70.0, 10.0, n)
    x1 = truth + rng.normal(0.0, 1.0, n)
    x2 = truth + 3.0 + rng.normal(0.0, 2.0, n)
    x3 = 70.0 + 1.5 * (truth - 70.0) + rng.normal(0.0, 1.5, n)
    return np.stack([x1, x2, x3]), truth, ["sharp", "shifted", "overreacting"]


def test_recovers_per_component_sigma(planted):
    """The paper's Table 9: each member gets its OWN fitted scale."""
    X, y, names = planted
    est = AffineNormal(bias="affine", scale="per_component").fit(X, y, names=names)
    sig = [f.sigma for f in est.fits_]
    assert sig[0] < sig[2] < sig[1], f"sigma ordering wrong: {sig}"
    # Component 1: y = truth + noise(1.0), regressing y on x1 leaves ~1/sqrt(2)
    # residual scale. Just assert the components are separated by >20%.
    assert max(sig) / min(sig) > 1.2


def test_recovers_planted_slope(planted):
    """A component that over-reacts must fit b < 1: the amplitude
    miscalibration an additive shift is structurally blind to."""
    X, y, names = planted
    est = AffineNormal(bias="affine").fit(X, y, names=names)
    by_name = {f.name: f for f in est.fits_}
    assert by_name["overreacting"].slope < 0.9, (
        f"expected b<0.9 for the over-reacting component; "
        f"got {by_name['overreacting'].slope:.3f}"
    )
    assert by_name["overreacting"].is_amplitude_miscalibrated
    assert abs(by_name["sharp"].slope - 1.0) < 0.1
    assert not by_name["sharp"].is_amplitude_miscalibrated


def test_recovers_planted_intercept(planted):
    X, y, names = planted
    est = AffineNormal(bias="affine").fit(X, y, names=names)
    by_name = {f.name: f for f in est.fits_}
    # x2 = truth + 3, so the correction should subtract ~3.
    pred_at_70 = by_name["shifted"].intercept + by_name["shifted"].slope * 73.0
    assert abs(pred_at_70 - 70.0) < 1.0


def test_shift_form_cannot_fix_amplitude(planted):
    """bias='shift' is add_skill_blend's form. It must pin b=1, which is
    exactly why it cannot correct the over-reacting component: the
    comparison this module exists to make."""
    X, y, names = planted
    est = AffineNormal(bias="shift").fit(X, y, names=names)
    assert all(f.slope == 1.0 for f in est.fits_)
    aff = AffineNormal(bias="affine").fit(X, y, names=names)
    shift_sigma = {f.name: f.sigma for f in est.fits_}["overreacting"]
    aff_sigma = {f.name: f.sigma for f in aff.fits_}["overreacting"]
    assert aff_sigma < shift_sigma, (
        "affine should leave smaller residuals than shift on a component "
        "whose amplitude is wrong"
    )


def test_shared_scale_gives_every_component_the_same_sigma(planted):
    """BMA / Raftery eq. (12): one common sigma across components."""
    X, y, names = planted
    est = AffineNormal(scale="shared").fit(X, y, names=names)
    sig = {f.sigma for f in est.fits_}
    assert len(sig) == 1
    assert est.shared_sigma_ is not None


def test_shared_sigma_is_the_slp_spread_ratio(planted):
    """The paper's §4.2 identity: sigma_shared / sigma_i is SLP's c.

    Their BMA fits 1.566 against members 1.958-2.214, a ratio of
    0.707-0.800, bracketing SLP's fitted c = 0.768.
    """
    X, y, names = planted
    per = AffineNormal(scale="per_component").fit(X, y, names=names)
    sha = AffineNormal(scale="shared").fit(X, y, names=names)
    ratios = [sha.shared_sigma_ / f.sigma for f in per.fits_]
    assert all(r > 0 for r in ratios)
    # The shared sigma must sit inside the per-component spread, not outside.
    lo, hi = min(f.sigma for f in per.fits_), max(f.sigma for f in per.fits_)
    assert lo <= sha.shared_sigma_ <= hi


def test_moments_are_nan_where_a_component_is_silent():
    """A silent vendor must drop out, never be imputed (Rule #0.5)."""
    rng = np.random.default_rng(18)
    n = 500
    truth = rng.normal(70.0, 8.0, n)
    X = np.stack([truth + rng.normal(0, 1, n), truth + rng.normal(0, 2, n)])
    X[1, :200] = np.nan  # second component silent on the first 200 rows
    est = AffineNormal().fit(X, truth, names=["a", "b"])
    mu, sig = est.moments(X)
    assert np.all(np.isnan(mu[1, :200]))
    assert np.all(np.isnan(sig[1, :200]))
    assert np.all(np.isfinite(mu[0]))
    assert est.fits_[1].coverage == pytest.approx(0.6, abs=0.01)


def test_conditional_scale_widens_with_the_covariate():
    """scale='conditional': spread varies by ROW, not only by component."""
    rng = np.random.default_rng(19)
    n = 4000
    z = rng.uniform(0.0, 1.0, n)          # disagreement proxy
    truth = rng.normal(70.0, 8.0, n)
    # Noise deliberately grows with z.
    X = np.stack([truth + rng.normal(0, 1, n) * (0.5 + 3.0 * z)])
    est = AffineNormal(scale="conditional").fit(X, truth, names=["v"], z=z)
    _mu, sig = est.moments(X, z=z)
    lo = np.nanmean(sig[0][z < 0.2])
    hi = np.nanmean(sig[0][z > 0.8])
    assert hi > lo * 1.5, f"sigma should grow with z: {lo:.3f} -> {hi:.3f}"


def test_crps_scale_is_more_robust_than_mle_to_outliers():
    """CRPS is offered instead of MLE precisely for heavy tails."""
    rng = np.random.default_rng(20)
    n = 2000
    truth = rng.normal(70.0, 8.0, n)
    x = truth + rng.normal(0, 1.0, n)
    x[:20] += rng.normal(0, 30.0, 20)     # a few gross errors
    X = np.stack([x])
    mle = AffineNormal(fit_method="mle").fit(X, truth, names=["v"])
    crps = AffineNormal(fit_method="crps").fit(X, truth, names=["v"])
    assert crps.fits_[0].sigma < mle.fits_[0].sigma, (
        "CRPS-fitted sigma should be less inflated by outliers than MLE"
    )


def test_refuses_a_component_with_too_few_rows():
    rng = np.random.default_rng(21)
    n = 200
    truth = rng.normal(70.0, 8.0, n)
    X = np.stack([truth + rng.normal(0, 1, n), truth + rng.normal(0, 1, n)])
    X[1, 25:] = np.nan       # only 25 finite rows
    with pytest.raises(ValueError, match="finite rows"):
        AffineNormal().fit(X, truth, names=["ok", "thin"])


def test_moments_reject_wrong_component_count(planted):
    X, y, names = planted
    est = AffineNormal().fit(X, y, names=names)
    with pytest.raises(ValueError, match="k=2"):
        est.moments(X[:2])


def test_conditional_scale_requires_z(planted):
    X, y, names = planted
    with pytest.raises(ValueError, match="needs a per-row covariate"):
        AffineNormal(scale="conditional").fit(X, y, names=names)


def test_report_renders(planted):
    X, y, names = planted
    est = AffineNormal().fit(X, y, names=names)
    txt = est.report()
    # "sd spread", not "sigma spread": for a Student-t fit the scale is not
    # the SD, so the summary line reports the SD to stay comparable across
    # dist families.
    assert "sd spread" in txt
    for nm in names:
        assert nm in txt


def test_moments_feed_the_pool(planted):
    """The point of the module: components become inputs to bracketlearn.pool,
    which unlocks SLP and BMA: both need per-component moments and so cannot
    run on PMF-only experts."""
    from bracketlearn.pool import fit_pool, spread_adjusted_cdfs

    X, y, names = planted
    est = AffineNormal().fit(X, y, names=names)
    mu, sig = est.moments(X)
    edges = np.linspace(30.0, 110.0, 41)
    r_idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(edges) - 2)
    F = spread_adjusted_cdfs(mu, sig, edges, spread_c=1.0)
    slp = fit_pool(F, r_idx, formula="slp", moments=(mu, sig), edges=edges, y=y)
    bma = fit_pool(F, r_idx, formula="bma", moments=(mu, sig), edges=edges, y=y)
    assert slp.converged and bma.converged
    assert abs(float(bma.weights.sum()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Student-t scale family
#
# The paper fits a Gaussian at step 0 because its 8 components were members of
# ONE ensemble, near-exchangeable and homogeneous. Vendor residuals here mix
# provider outages, station siting and gross errors, which is the generating
# story that produces heavy tails, so the t is offered alongside the normal.
# These tests pin that the fitter RESOLVES nu rather than defaulting to a
# convenient value, which is what makes a measured nu evidence about tails.
# ---------------------------------------------------------------------------


def test_student_t_recovers_planted_degrees_of_freedom():
    """A planted nu must come back off the grid, not a grid endpoint.

    Without this, "every vendor fits nu=4" could equally mean the fitter
    always says 4.
    """
    from scipy.stats import t as student_t

    rng = np.random.default_rng(0)
    n = 20000
    for nu_true in (3.0, 5.0, 10.0, 30.0):
        x = rng.normal(60, 8, n)
        y = 1.5 + 0.9 * x + student_t.rvs(
            df=nu_true, scale=2.0, size=n, random_state=1
        )
        est = AffineNormal(dist="student_t").fit(x[None, :], y, names=["v"])
        f = est.fits_[0]
        assert f.nu == nu_true, f"planted nu={nu_true}, fitted {f.nu}"
        assert abs(f.sigma - 2.0) < 0.1, f"scale off at nu={nu_true}"


def test_gaussian_data_lands_at_the_flat_top_of_the_nu_grid():
    """Normally distributed residuals must NOT be reported as heavy-tailed.

    The t nests the normal as nu -> inf, so a t fit on Gaussian data should
    pick the largest nu on the grid and reproduce the normal's sigma. This is
    the false-positive guard on is_heavy_tailed.
    """
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.normal(60, 8, n)
    y = 1.5 + 0.9 * x + rng.normal(0, 2.0, n)

    t_fit = AffineNormal(dist="student_t").fit(x[None, :], y, names=["v"]).fits_[0]
    n_fit = AffineNormal(dist="normal").fit(x[None, :], y, names=["v"]).fits_[0]

    assert t_fit.nu == max(_NU_GRID)
    assert not t_fit.is_heavy_tailed
    # The t's SD, not its scale, is what compares to the normal's sigma.
    assert abs(t_fit.sd - n_fit.sigma) < 0.02


def test_student_t_scale_is_not_the_standard_deviation():
    """sigma is the SCALE; sd inflates it by sqrt(nu/(nu-2)).

    Reading .sigma across families is the trap this property exists to stop:
    a t_3 with scale 2.0 has SD 3.46, so a scale-based dispersion comparison
    would call it much sharper than an equivalent normal.
    """
    f = ComponentFit(
        name="v", intercept=0.0, slope=1.0, sigma=2.0,
        n_rows=100, coverage=1.0, nu=3.0, dist="student_t",
    )
    assert abs(f.sd - 2.0 * math.sqrt(3.0)) < 1e-12
    assert f.sd > f.sigma

    # A normal's sd IS its sigma: the property must not inflate it.
    g = ComponentFit(
        name="v", intercept=0.0, slope=1.0, sigma=2.0,
        n_rows=100, coverage=1.0,
    )
    assert g.sd == 2.0


def test_student_t_sd_is_infinite_at_nu_two():
    """t_2 has no finite variance. Report inf, never a silently finite number."""
    f = ComponentFit(
        name="v", intercept=0.0, slope=1.0, sigma=2.0,
        n_rows=100, coverage=1.0, nu=2.0, dist="student_t",
    )
    assert math.isinf(f.sd)


def test_cdf_dispatches_on_the_fitted_family():
    """A t fit scored against a normal CDF is a silent dispersion bug.

    est.cdf exists so no call site hard-codes norm.cdf; this pins that it
    actually differs between the two families in the tails.
    """
    from scipy.stats import norm
    from scipy.stats import t as student_t

    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(60, 8, n)
    y = 1.5 + 0.9 * x + student_t.rvs(df=3.0, scale=2.0, size=n, random_state=3)

    t_est = AffineNormal(dist="student_t").fit(x[None, :], y, names=["v"])
    n_est = AffineNormal(dist="normal").fit(x[None, :], y, names=["v"])

    mu = np.array([0.0, 0.0])
    sig = np.array([1.0, 1.0])
    thr = np.array([3.0, -3.0])   # tail thresholds, where the families diverge

    assert np.allclose(n_est.cdf(thr, mu, sig), norm.cdf(thr))
    t_vals = t_est.cdf(thr, mu, sig)
    assert not np.allclose(t_vals, norm.cdf(thr)), (
        "student_t fit returned the normal CDF: dispatch is broken"
    )
    # A t has fatter tails: more mass beyond +3 and below -3.
    assert t_vals[0] < norm.cdf(3.0)
    assert t_vals[1] > norm.cdf(-3.0)
