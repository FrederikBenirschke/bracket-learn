"""Tests for bracketlearn.pool, the combination formulas.

The substantive tests here assert the *theorems* of Gneiting & Ranjan
(2013) rather than only the plumbing. If the arithmetic drifts, Theorem 3.1
stops holding on synthetic data and these fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.pool import (
    NEUTRAL_PIT_VAR,
    beta_warp,
    fit_pool,
    pit_variance,
    pool_cdf,
    spread_adjusted_cdfs,
)


def _normal_edge_cdfs(mu, sigma, edges):
    """(N, B+1) cumulative mass at shared edges for N(mu, sigma) rows."""
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)[:, None]
    sigma = np.asarray(sigma, dtype=float)[:, None]
    e = np.asarray(edges, dtype=float)[None, :]
    c = norm.cdf((e - mu) / sigma)
    # Force the outer edges to carry the full mass so each row is a proper
    # distribution over the ladder.
    c[:, 0] = 0.0
    c[:, -1] = 1.0
    return c


def _realized_bracket(y, edges):
    return np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(edges) - 2)


@pytest.fixture
def synthetic():
    """Two distinct, individually well-calibrated forecasters.

    Y = X + eps with X and eps independent standard normals. Forecaster i
    sees a noisy X and predicts N(x_i, 1). Each is close to ideal, and they
    disagree row by row. That disagreement is the condition Theorem 3.1
    needs, namely Fi != Fj with positive probability.
    """
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n)
    edges = np.linspace(-8.0, 8.0, 33)

    x1 = x + 0.25 * rng.standard_normal(n)
    x2 = x + 0.25 * rng.standard_normal(n)
    sd = np.sqrt(1.0 + 0.25**2)
    F = np.stack(
        [
            _normal_edge_cdfs(x1, np.full(n, sd), edges),
            _normal_edge_cdfs(x2, np.full(n, sd), edges),
        ]
    )
    return F, _realized_bracket(y, edges), edges


def _pit(edge_cdfs, r_idx):
    rows = np.arange(edge_cdfs.shape[0])
    mass = edge_cdfs[rows, r_idx + 1] - edge_cdfs[rows, r_idx]
    return edge_cdfs[rows, r_idx] + 0.5 * mass


# ---------------------------------------------------------------------------
# The theorems.
# ---------------------------------------------------------------------------


def test_theorem_3_1_linear_pool_increases_dispersion(synthetic):
    """Thm 3.1(a) and (b), that the linear pool is more dispersed than its
    least-dispersed component, so its var(PIT) is strictly lower.

    A lower var(PIT) means more dispersed (Definition 2.7(d)), and neutral is
    1/12. This is the defect that motivates the whole module, so it is
    asserted directly.
    """
    F, r_idx, _ = synthetic
    w = np.array([0.5, 0.5])

    comp_vars = [pit_variance(_pit(F[i], r_idx)) for i in range(F.shape[0])]
    pooled = pool_cdf(F, weights=w, formula="tlp")
    pooled_var = pit_variance(_pit(pooled, r_idx))

    assert pooled_var < min(comp_vars), (
        f"linear pool var(PIT)={pooled_var:.5f} should be strictly below "
        f"every component {comp_vars} (more dispersed, Thm 3.1b)"
    )


def test_theorem_3_1_pool_of_neutral_components_is_overdispersed(synthetic):
    """Thm 3.1(c), that neutrally dispersed components pool to an
    overdispersed forecast, with var(PIT) below the neutral 1/12."""
    F, r_idx, _ = synthetic
    for i in range(F.shape[0]):
        v = pit_variance(_pit(F[i], r_idx))
        assert abs(v - NEUTRAL_PIT_VAR) < 0.02, (
            f"component {i} var(PIT)={v:.5f} is not close enough to neutral "
            f"{NEUTRAL_PIT_VAR:.5f} for this test's premise to hold"
        )
    pooled = pool_cdf(F, weights=np.array([0.5, 0.5]), formula="tlp")
    assert pit_variance(_pit(pooled, r_idx)) < NEUTRAL_PIT_VAR


def test_theorem_3_9_blp_reaches_neutral_dispersion(synthetic):
    """Thm 3.9, that BLP is flexibly dispersive, so some (α, β) restores
    neutrality.

    The linear pool of these components cannot do so (Thm 3.3). Sharpening
    with α = β > 1 should pull var(PIT) back up toward 1/12.
    """
    F, r_idx, _ = synthetic
    w = np.array([0.5, 0.5])
    tlp_var = pit_variance(_pit(pool_cdf(F, weights=w, formula="tlp"), r_idx))

    best, best_gap = None, float("inf")
    for a in np.linspace(1.0, 3.0, 41):
        v = pit_variance(
            _pit(pool_cdf(F, weights=w, formula="blp", alpha=a, beta=a), r_idx)
        )
        if abs(v - NEUTRAL_PIT_VAR) < best_gap:
            best, best_gap = (a, v), abs(v - NEUTRAL_PIT_VAR)

    assert best_gap < abs(tlp_var - NEUTRAL_PIT_VAR), (
        f"BLP at α=β={best[0]:.2f} gives var(PIT)={best[1]:.5f}; should be "
        f"closer to neutral than TLP's {tlp_var:.5f}"
    )
    assert best_gap < 0.006, f"BLP should reach near-neutral; best={best}"


def test_blp_nests_tlp_at_alpha_beta_one(synthetic):
    """α = β = 1 makes the beta transform the identity (eq. 8)."""
    F, _, _ = synthetic
    w = np.array([0.3, 0.7])
    tlp = pool_cdf(F, weights=w, formula="tlp")
    blp = pool_cdf(F, weights=w, formula="blp", alpha=1.0, beta=1.0)
    np.testing.assert_allclose(tlp, blp, atol=1e-6)


def test_beta_warp_is_monotone_and_bounded():
    p = np.linspace(0.0, 1.0, 101)
    for a, b in [(1.0, 1.0), (1.467, 1.467), (0.5, 2.0), (3.0, 0.7)]:
        out = beta_warp(p, a, b)
        assert np.all(np.diff(out) >= -1e-12), f"not monotone at α={a} β={b}"
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_glp_probit_allows_extremization(synthetic):
    """Example 3.5, weights above 1 under the probit link. This is the only
    coherent formula in the paper, and it sharpens rather than flattens."""
    F, r_idx, _ = synthetic
    convex = pool_cdf(
        F, weights=np.array([0.5, 0.5]), formula="glp", link="probit"
    )
    extremized = pool_cdf(
        F, weights=np.array([0.9, 0.9]), formula="glp", link="probit"
    )
    assert pit_variance(_pit(extremized, r_idx)) > pit_variance(
        _pit(convex, r_idx)
    ), "weights > 1 under probit should sharpen (raise var(PIT))"


# ---------------------------------------------------------------------------
# Fitting.
# ---------------------------------------------------------------------------


def test_fit_pool_blp_improves_log_score_over_tlp(synthetic):
    F, r_idx, _ = synthetic
    tlp = fit_pool(F, r_idx, formula="tlp")
    blp = fit_pool(F, r_idx, formula="blp")
    assert blp.mean_log_score >= tlp.mean_log_score - 1e-9, (
        "BLP nests TLP, so its fitted log score cannot be worse"
    )
    assert blp.converged


def test_fit_pool_moves_dispersion_toward_neutral(synthetic):
    F, r_idx, _ = synthetic
    tlp = fit_pool(F, r_idx, formula="tlp")
    blp = fit_pool(F, r_idx, formula="blp")
    assert abs(blp.pit_var_fit - NEUTRAL_PIT_VAR) < abs(
        tlp.pit_var_fit - NEUTRAL_PIT_VAR
    )


def test_k1_calibration_mode(synthetic):
    """§3.4, where BLP with k=1 is a pure calibration and dispersion
    correction."""
    F, r_idx, _ = synthetic
    single = F[:1]
    fit = fit_pool(
        single, r_idx, formula="blp", fixed_weights=np.array([1.0])
    )
    assert fit.weights.shape == (1,)
    assert fit.converged


def test_leak_check_fires_on_in_sample_components():
    """The directional in-sample failure. Components that saw the outcome
    have too-small residuals, so fit-row var(PIT) reads far below
    holdout."""
    rng = np.random.default_rng(5)
    n = 800
    edges = np.linspace(-8.0, 8.0, 33)
    y = rng.standard_normal(n) * 2.0

    # "In-sample" component, centred on the realized value itself.
    leaky = _normal_edge_cdfs(y, np.full(n, 1.0), edges)[None, :, :]
    r_idx = _realized_bracket(y, edges)

    # Honest holdout, the same model form on rows it never saw.
    y_h = rng.standard_normal(400) * 2.0
    honest = _normal_edge_cdfs(
        np.zeros(400), np.full(400, 2.0), edges
    )[None, :, :]
    r_h = _realized_bracket(y_h, edges)

    fit = fit_pool(
        leaky, r_idx, formula="blp",
        fixed_weights=np.array([1.0]), holdout=(honest, r_h),
    )
    assert fit.leak_warning_ is not None, (
        "fitting on components centred on the realized value must trip the "
        "leak check"
    )


def test_fit_pool_rejects_too_few_rows(synthetic):
    F, r_idx, _ = synthetic
    with pytest.raises(ValueError, match="need ≥30 rows"):
        fit_pool(F[:, :10], r_idx[:10], formula="blp")


def test_pool_cdf_rejects_component_axis_mismatch(synthetic):
    F, _, _ = synthetic
    with pytest.raises(ValueError, match="component axis mismatch"):
        pool_cdf(F, weights=np.array([1.0]), formula="tlp")


def test_pool_cdf_accepts_every_formula_fit_pool_accepts(parametric):
    """Any formula fit_pool takes, pool_cdf must also take.

    In the original regression fit_pool accepted 'bma' while pool_cdf raised
    unknown-formula on it. A model fitted with BMA then failed on the predict
    path alone, after a full training run and on every fold.
    """
    mu, sigma, y, edges, r_idx = parametric
    F = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    w = np.array([0.5, 0.5])
    for formula in ("tlp", "slp", "blp", "glp", "bma"):
        out = pool_cdf(F, weights=w, formula=formula)
        assert out.shape == F.shape[1:], f"{formula} returned {out.shape}"
        assert np.all(np.diff(out, axis=1) >= -1e-9), f"{formula} not a CDF"


def test_pool_cdf_rejects_unknown_formula(synthetic):
    F, _, _ = synthetic
    with pytest.raises(ValueError, match="unknown formula"):
        pool_cdf(F, weights=np.array([0.5, 0.5]), formula="nope")


# ---------------------------------------------------------------------------
# SLP and BMA, the two routes to the same spread adjustment.
# ---------------------------------------------------------------------------


@pytest.fixture
def parametric():
    """Same generating process as `synthetic`, but keeping (mu, sigma).

    The components are deliberately overdispersed, with sigma inflated 1.35x.
    That is the regime where SLP's c < 1 and BMA's shrink both have something
    to do, and it is what a linear pool of neutral components produces
    (Thm 3.1c).
    """
    rng = np.random.default_rng(23)
    n = 3000
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n)
    edges = np.linspace(-8.0, 8.0, 33)
    sd = np.sqrt(1.0 + 0.25**2) * 1.35
    mu = np.stack([x + 0.25 * rng.standard_normal(n) for _ in range(2)])
    sigma = np.full_like(mu, sd)
    return mu, sigma, y, edges, _realized_bracket(y, edges)


def test_slp_recovers_c_below_one_for_overdispersed_components(parametric):
    """Neutrally dispersed or overdispersed components call for c < 1 (§3.3).

    The paper's Seattle-Tacoma fit is 0.768, and Berrocal et al. report
    0.65-1.03.
    """
    mu, sigma, y, edges, r_idx = parametric
    F = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    fit = fit_pool(
        F, r_idx, formula="slp", moments=(mu, sigma), edges=edges, y=y,
    )
    assert fit.spread_c < 1.0, (
        f"overdispersed components should fit c < 1; got {fit.spread_c:.3f}"
    )
    assert fit.converged


def test_slp_beats_tlp_on_log_score(parametric):
    mu, sigma, y, edges, r_idx = parametric
    F = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    tlp = fit_pool(F, r_idx, formula="tlp")
    slp = fit_pool(
        F, r_idx, formula="slp", moments=(mu, sigma), edges=edges, y=y,
    )
    assert slp.mean_log_score > tlp.mean_log_score


def test_slp_c_equals_one_recovers_tlp(parametric):
    """c = 1 is the traditional linear pool (eq. 6)."""
    mu, sigma, _y, edges, _r = parametric
    w = np.array([0.5, 0.5])
    at_one = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    np.testing.assert_allclose(
        pool_cdf(at_one, weights=w, formula="slp"),
        pool_cdf(at_one, weights=w, formula="tlp"),
        atol=1e-12,
    )


def test_bma_implied_spread_ratio_tracks_slp_c(parametric):
    """The paper's §4.2 finding, that BMA's sigma_hat/sigma_i and SLP's c are
    the same shrink reached by different estimators, at 0.707-0.800 against
    0.768 on their data.

    This is the test that justifies putting both behind one interface.
    """
    mu, sigma, y, edges, r_idx = parametric
    F = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    slp = fit_pool(
        F, r_idx, formula="slp", moments=(mu, sigma), edges=edges, y=y,
    )
    bma = fit_pool(
        F, r_idx, formula="bma", moments=(mu, sigma), edges=edges, y=y,
    )
    assert bma.spread_c < 1.0
    assert abs(bma.spread_c - slp.spread_c) < 0.15, (
        f"BMA implied ratio {bma.spread_c:.3f} should track SLP c "
        f"{slp.spread_c:.3f}: they are the same shrink (paper §4.2)"
    )


def test_bma_reports_sigma_common_and_weights(parametric):
    mu, sigma, y, edges, r_idx = parametric
    F = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    bma = fit_pool(
        F, r_idx, formula="bma", moments=(mu, sigma), edges=edges, y=y,
    )
    assert bma.notes["sigma_common"] > 0
    assert abs(float(bma.weights.sum()) - 1.0) < 1e-9
    assert bma.notes["n_iter"] >= 1


def test_slp_refuses_bracket_only_input(synthetic):
    """SLP is not local. Without moments it would need within-bracket
    interpolation, which imposes a shape the data never specified."""
    F, r_idx, _ = synthetic
    with pytest.raises(ValueError, match="requires moments"):
        fit_pool(F, r_idx, formula="slp")


def test_moments_rejected_for_local_formulas(parametric):
    mu, sigma, y, edges, r_idx = parametric
    F = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    with pytest.raises(ValueError, match="only used by formula"):
        fit_pool(F, r_idx, formula="blp", moments=(mu, sigma), edges=edges, y=y)


def test_spread_adjusted_cdfs_rejects_bad_c(parametric):
    mu, sigma, _y, edges, _r = parametric
    with pytest.raises(ValueError, match="spread_c must be > 0"):
        spread_adjusted_cdfs(mu, sigma, edges, spread_c=0.0)


def test_pooled_cdf_stays_monotone_across_edges(synthetic):
    """A pooled CDF must remain a CDF, non-decreasing along the edge
    axis."""
    F, _, _ = synthetic
    w = np.array([0.4, 0.6])
    for kwargs in [
        {"formula": "tlp"},
        {"formula": "blp", "alpha": 1.467, "beta": 1.467},
        {"formula": "blp", "alpha": 0.6, "beta": 2.2},
        {"formula": "glp", "link": "log"},
        {"formula": "glp", "link": "probit"},
        {"formula": "glp", "link": "harmonic"},
    ]:
        out = pool_cdf(F, weights=w, **kwargs)
        assert np.all(np.diff(out, axis=1) >= -1e-9), f"not a CDF: {kwargs}"


# ---------------------------------------------------------------------------
# Student-t components through the pooling path
#
# AffineNormal(dist="student_t") produces components whose sigma is a scale
# rather than a standard deviation, and whose tails are heavier than a
# Gaussian's. Pooling them through the Gaussian path raises no error and
# yields plausible bracket probabilities with the wrong tails. These tests
# therefore pin that the family travels with the moments, and that omitting
# it is an error rather than a default.
# ---------------------------------------------------------------------------


def test_student_t_components_differ_from_gaussian_in_the_tails():
    """The family must actually change the CDF.

    If this passes trivially, with the values equal, then the dist argument
    is being ignored and every t component is being pooled as a Gaussian.
    """
    mu = np.zeros((2, 3))
    sigma = np.ones((2, 3))
    edges = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])

    g = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    t = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0,
                             dist="student_t", nu=3.0)

    assert not np.allclose(g, t), "dist='student_t' was ignored"
    # Heavier tails, so more mass below -2 and above +2 for the t.
    assert np.all(t[:, :, 1] > g[:, :, 1])
    interior = slice(1, -1)
    assert np.all(t[:, :, 3][interior] <= g[:, :, 3][interior] + 1e-12)


def test_student_t_without_nu_raises_rather_than_defaulting():
    """Rule #0.5. A missing nu must not fall back to a Gaussian."""
    mu = np.zeros((2, 3))
    sigma = np.ones((2, 3))
    edges = np.array([-8.0, 0.0, 8.0])
    with pytest.raises(ValueError, match="needs nu"):
        spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0, dist="student_t")


def test_unknown_dist_raises():
    mu = np.zeros((1, 2))
    sigma = np.ones((1, 2))
    edges = np.array([-8.0, 0.0, 8.0])
    with pytest.raises(ValueError, match="unknown dist"):
        spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0, dist="cauchy")


def test_student_t_components_still_carry_full_mass_over_the_ladder():
    """Outer edges pinned to 0/1, as for the Gaussian path.

    A t's heavier tails leak more mass beyond a finite ladder. If the pinning
    were family-specific, this is where it would show.
    """
    rng = np.random.default_rng(3)
    mu = rng.normal(70, 5, (4, 20))
    sigma = np.full((4, 20), 2.5)
    edges = np.linspace(40.0, 100.0, 31)
    t = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0,
                             dist="student_t", nu=2.5)
    assert np.allclose(t[:, :, 0], 0.0)
    assert np.allclose(t[:, :, -1], 1.0)
    # Monotone in the threshold, a CDF per component and row.
    assert np.all(np.diff(t, axis=2) >= -1e-12)


def test_large_nu_converges_to_the_gaussian_path():
    """t_nu tends to a normal as nu grows. This guards the parameterisation.

    Confusing the scale with the standard deviation inside the t branch would
    leave a persistent gap here that no nu closes.
    """
    rng = np.random.default_rng(4)
    mu = rng.normal(70, 5, (3, 15))
    sigma = np.full((3, 15), 3.0)
    edges = np.linspace(50.0, 90.0, 21)

    g = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0)
    gaps = [
        float(np.max(np.abs(
            spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0,
                                 dist="student_t", nu=nu) - g
        )))
        for nu in (5.0, 30.0, 500.0)
    ]
    assert gaps == sorted(gaps, reverse=True), "gap must shrink as nu grows"
    assert gaps[-1] < 0.005


def test_pooled_t_components_give_a_valid_probability_vector():
    """End to end, from t components through pool_cdf to bracket
    probabilities."""
    rng = np.random.default_rng(5)
    k, n = 4, 12
    mu = rng.normal(70, 4, (k, n))
    sigma = np.full((k, n), 2.5)
    edges = np.linspace(50.0, 90.0, 21)
    F = spread_adjusted_cdfs(mu, sigma, edges, spread_c=1.0,
                             dist="student_t", nu=4.0)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    for formula in ("tlp", "blp", "glp"):
        cdf = pool_cdf(F, weights=w, formula=formula, alpha=1.0, beta=1.0)
        probs = np.diff(cdf, axis=1)
        assert np.all(probs >= -1e-12), f"{formula} produced negative mass"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)
