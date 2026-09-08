"""The calibration suite, and the failures var(PIT) alone cannot see.

Each test in the first block PLANTS a specific miscalibration, asserts that
var(PIT) is blind to it, and asserts that some other column in the suite
catches it. That pairing is the argument for the suite existing: without
the "var(PIT) is blind" half, the extra columns are just more numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from bracketlearn.calibration import (
    NEUTRAL_PIT_VAR,
    calibration_suite,
    pit_values,
)


def _normal_cdf(mu, sigma):
    return lambda thr: norm.cdf((np.asarray(thr) - mu) / sigma)


def _normal_q(mu, sigma):
    return lambda lv: mu + sigma * norm.ppf(np.asarray(lv))


# ---------------------------------------------------------------------------
# What var(PIT) misses
# ---------------------------------------------------------------------------


def test_bias_is_MISREAD_by_var_pit_as_overdispersion():
    """The headline reason var(PIT) cannot stand alone.

    A forecast that is 3°F warm with a PERFECTLY CORRECT spread has its PIT
    mass pushed toward one end, which SHRINKS the variance: 0.0835 -> 0.0558
    here. On the documented reading rule that is "overdispersed": the
    intervals are too wide, which is precisely the wrong diagnosis. The
    spread needs no change at all; the centre does.

    So var(PIT) is not merely blind to bias, it MISLABELS it. Two failures
    with opposite remedies produce the same verdict:

        bias +3F, spread correct    -> pit_var 0.0558, "overdispersed"
        no bias, spread 1.7x wide   -> pit_var 0.0415, "overdispersed"

    Only pit_mean separates them (0.24 vs 0.50). This matters directly for
    the vendor lift: an overdispersion finding read off var(PIT) alone could
    be an uncorrected bias wearing a dispersion verdict.
    """
    rng = np.random.default_rng(0)
    n = 200_000
    y = rng.normal(70.0, 3.0, n)
    mu = np.full(n, 73.0)          # 3 degF warm
    sd = np.full(n, 3.0)           # spread is correct

    s = calibration_suite(_normal_cdf(mu, sd), y, sd=sd,
                          quantile=_normal_q(mu, sd))

    # var(PIT) reports "overdispersed" although the spread is exactly right.
    assert s["pit_var"] < NEUTRAL_PIT_VAR * 0.90, (
        "the planted bias must actually produce the wrong dispersion verdict"
    )
    # The suite catches the true failure four independent ways.
    assert s["pit_mean"] < 0.35, "pit_mean must see the shift"
    assert s["pit_ks"] > 0.15, "KS must reject uniformity"
    assert s["tail_left"] > 0.20, "mass piles into the left tail"
    assert s["reliability_mae"] > 0.05


def test_bias_and_overdispersion_are_indistinguishable_on_var_pit():
    """The confound, stated directly: same verdict, opposite remedies."""
    rng = np.random.default_rng(12)
    n = 200_000
    y = rng.normal(70.0, 3.0, n)

    biased = calibration_suite(
        _normal_cdf(np.full(n, 73.0), np.full(n, 3.0)), y)
    wide = calibration_suite(
        _normal_cdf(np.full(n, 70.0), np.full(n, 5.1)), y)

    # Both are called overdispersed by the variance alone.
    assert biased["pit_var"] < NEUTRAL_PIT_VAR * 0.90
    assert wide["pit_var"] < NEUTRAL_PIT_VAR * 0.90
    # pit_mean is what tells them apart.
    assert abs(wide["pit_mean"] - 0.5) < 0.01, "a pure spread error stays centred"
    assert abs(biased["pit_mean"] - 0.5) > 0.20, "a bias moves the centre"


def test_asymmetric_tails_share_one_var_pit_but_differ_in_skew():
    """Two forecasts, mirror-image tail errors, identical var(PIT).

    Skew is what separates 'too heavy on the left' from 'too heavy on the
    right'. A variance is symmetric by construction and cannot.
    """
    rng = np.random.default_rng(1)
    n = 120_000
    mu = np.zeros(n)
    sd = np.ones(n)
    # Right-skewed and left-skewed outcome distributions around the same mean.
    y_right = rng.gumbel(-0.5772, 1.0, n) * 0.6
    y_left = -y_right

    a = calibration_suite(_normal_cdf(mu, sd), y_right)
    b = calibration_suite(_normal_cdf(mu, sd), y_left)

    # Same dispersion diagnosis...
    assert abs(a["pit_var"] - b["pit_var"]) < 0.002
    # ...opposite skew, and the tail masses swap.
    assert a["pit_skew"] * b["pit_skew"] < 0, "skew must have opposite signs"
    assert abs(a["pit_skew"]) > 0.10
    assert (a["tail_left"] > a["tail_right"]) != (b["tail_left"] > b["tail_right"])


def test_heavy_tails_and_a_narrow_body_are_one_verdict_on_var_pit():
    """A shape error that no rescaling can fix, and how the suite says so.

    Outcomes from a t_3 scored under a normal of the SAME SCALE: the body is
    too narrow AND both tails are too heavy. var(PIT) reports 0.1004, i.e.
    "underdispersed: widen the intervals". But widening makes the body worse;
    the fault is the family, not the scale.

    Measured, t_3 outcomes under a normal, 400k rows:

        matching          pit_var   tail_L   tail_R   pit_ks
        scale-matched      0.1004   0.0993   0.0987   0.0494
        variance-matched   0.0579   0.0329   0.0326   0.0871

    (calibrated is 0.0833 / 0.05 / 0.05 / 0)

    The two ways of matching a normal to the same data give OPPOSITE
    dispersion verdicts. That is the argument against a single dispersion
    number: what it reports depends on a modelling choice it does not show
    you. The tail masses do show it: 0.05/0.05 is the calibrated value, and
    both rows are wrong on both sides simultaneously, which a scale change
    cannot repair in either direction.
    """
    n = 400_000
    y = student_t.rvs(df=3.0, size=n, random_state=7)
    mu = np.zeros(n)

    scale_matched = calibration_suite(_normal_cdf(mu, np.ones(n)), y)
    var_matched = calibration_suite(
        _normal_cdf(mu, np.full(n, np.sqrt(3.0))), y)

    # Opposite dispersion verdicts from the same outcomes.
    assert scale_matched["pit_var"] > NEUTRAL_PIT_VAR * 1.10   # "too narrow"
    assert var_matched["pit_var"] < NEUTRAL_PIT_VAR * 0.90     # "too wide"

    # Tails say what is really wrong, in both cases. Calibrated is 0.05.
    assert scale_matched["tail_left"] > 0.09
    assert scale_matched["tail_right"] > 0.09
    assert var_matched["tail_left"] < 0.04
    assert var_matched["tail_right"] < 0.04

    # KS rejects both, which a dispersion number cannot do on its own. At
    # n=400k the 1% critical value is ~0.0026, so both are decisive
    # rejections (measured 0.049 and 0.087).
    assert scale_matched["pit_ks"] > 0.04
    assert var_matched["pit_ks"] > 0.04


# ---------------------------------------------------------------------------
# The suite is correct on a perfectly specified forecast
# ---------------------------------------------------------------------------


def test_perfect_forecast_hits_every_neutral_value():
    rng = np.random.default_rng(3)
    n = 200_000
    mu = rng.normal(70, 8, n)
    sd = np.full(n, 3.0)
    y = mu + rng.normal(0, 3.0, n)

    s = calibration_suite(_normal_cdf(mu, sd), y, sd=sd,
                          quantile=_normal_q(mu, sd))

    assert abs(s["pit_mean"] - 0.5) < 0.005
    assert abs(s["pit_var"] - NEUTRAL_PIT_VAR) < 0.002
    assert abs(s["pit_skew"]) < 0.05
    assert s["pit_ks"] < 0.01
    assert abs(s["tail_left"] - 0.05) < 0.005
    assert abs(s["tail_right"] - 0.05) < 0.005
    assert abs(s["coverage_50"] - 0.50) < 0.01
    assert abs(s["coverage_90"] - 0.90) < 0.01
    assert s["reliability_mae"] < 0.01
    assert abs(s["rmv"] - 3.0) < 0.01


@pytest.mark.parametrize("factor,expect", [(0.5, "under"), (2.0, "over")])
def test_var_pit_signs_dispersion_in_the_documented_direction(factor, expect):
    """Guard on the reading rule in the docstring.

    Too-narrow intervals give a U-shaped PIT (variance ABOVE 1/12); too-wide
    give a hump (BELOW). Getting this backwards would invert every dispersion
    verdict in the repo, and the direction is easy to talk oneself out of.
    """
    rng = np.random.default_rng(4)
    n = 100_000
    mu = np.zeros(n)
    y = rng.normal(0, 1.0, n)
    sd = np.full(n, factor)

    s = calibration_suite(_normal_cdf(mu, sd), y)
    if expect == "under":
        assert s["pit_var"] > NEUTRAL_PIT_VAR * 1.10
    else:
        assert s["pit_var"] < NEUTRAL_PIT_VAR * 0.90


def test_sharpness_is_measured_without_the_outcome():
    """Sharpness is a property of the forecast alone.

    Two runs with identical forecasts but different outcomes must report the
    same sharpness: otherwise it is an accuracy metric in disguise, and
    "sharp subject to calibrated" stops meaning anything.
    """
    rng = np.random.default_rng(5)
    n = 20_000
    mu = np.zeros(n)
    sd = np.full(n, 2.0)
    a = calibration_suite(_normal_cdf(mu, sd), rng.normal(0, 2, n),
                          sd=sd, quantile=_normal_q(mu, sd))
    b = calibration_suite(_normal_cdf(mu, sd), rng.normal(9, 7, n),
                          sd=sd, quantile=_normal_q(mu, sd))
    assert a["rmv"] == b["rmv"]
    assert a["sharpness_iqr"] == b["sharpness_iqr"]
    # ...while calibration differs wildly, confirming the forecasts were
    # genuinely being scored against different outcomes.
    assert abs(a["pit_ks"] - b["pit_ks"]) > 0.3


# ---------------------------------------------------------------------------
# Discrete outcomes
# ---------------------------------------------------------------------------


def test_discrete_pit_needs_its_own_neutral_reference():
    """The correction is to the TARGET, not only to the statistic.

    Rounding the outcome removes the within-cell spread a continuous PIT
    would have, so a perfectly calibrated forecast on a grid attains

        Var[PIT_mid] = 1/12 - E[p^2]/12   (p = the realized cell's probability)

    which is strictly below 1/12. Comparing a discrete PIT against 1/12
    therefore manufactures an "overdispersed" verdict out of the grid alone.

    This test pins the identity, and pins that the naive PIT, which happens
    to sit numerically closer to 1/12: is nonetheless the wrong quantity.
    """
    rng = np.random.default_rng(6)
    n = 400_000
    mu = rng.normal(70, 8, n)
    for sigma in (1.0, 2.0, 3.0):
        sd = np.full(n, sigma)
        y_cont = mu + rng.normal(0, sigma, n)
        y = np.round(y_cont)                    # integer settlement

        s = calibration_suite(_normal_cdf(mu, sd), y, grid_step=1.0)

        # The forecast IS calibrated, so the excess over the correct target
        # must be ~0: even though pit_var itself is well below 1/12.
        assert abs(s["pit_var_excess"]) < 0.002, (
            f"sigma={sigma}: calibrated forecast must show ~zero excess"
        )
        assert s["pit_var_neutral"] < NEUTRAL_PIT_VAR, (
            "the discrete target must sit below the continuous 1/12"
        )
        # Against the WRONG (continuous) reference the same forecast looks
        # overdispersed. This is the error the reference exists to prevent.
        if sigma <= 2.0:
            assert s["pit_var"] < NEUTRAL_PIT_VAR * 0.995


def test_naive_discrete_pit_is_biased_up_and_is_the_wrong_quantity():
    """Naive F(y) on rounded outcomes reads ABOVE the true continuous value.

    It is biased toward "underdispersed". It can look closer to 1/12 than the
    corrected statistic, which is exactly the trap: the corrected statistic is
    being compared to the wrong target, not computing the wrong thing.
    """
    rng = np.random.default_rng(13)
    n = 400_000
    mu = rng.normal(70, 8, n)
    sigma = 2.0
    sd = np.full(n, sigma)
    y_cont = mu + rng.normal(0, sigma, n)
    y = np.round(y_cont)

    true_cont = float(np.var(norm.cdf((y_cont - mu) / sigma)))
    naive = calibration_suite(_normal_cdf(mu, sd), y)          # no grid_step
    fixed = calibration_suite(_normal_cdf(mu, sd), y, grid_step=1.0)

    assert naive["pit_var"] > true_cont, "naive must be biased upward"
    # The corrected statistic matches ITS OWN target far better than the naive
    # one matches the continuous target.
    assert abs(fixed["pit_var_excess"]) < abs(naive["pit_var"] - true_cont)


def test_neutral_reference_is_continuous_when_outcome_is():
    rng = np.random.default_rng(14)
    n = 10_000
    mu = np.zeros(n)
    sd = np.ones(n)
    s = calibration_suite(_normal_cdf(mu, sd), rng.normal(0, 1, n))
    assert s["pit_var_neutral"] == NEUTRAL_PIT_VAR
    assert s["pit_var_excess"] == s["pit_var"] - NEUTRAL_PIT_VAR


def test_grid_step_is_deterministic_not_randomised():
    """No RNG: repeated calls on the same input give identical values.

    This repo bans the randomised PIT (it shifted the mean +0.093..+0.111 on
    every model over 85,250 forecasts). The mid-interval form is its
    deterministic analogue and must be reproducible.
    """
    rng = np.random.default_rng(7)
    n = 5_000
    mu = rng.normal(70, 8, n)
    sd = np.full(n, 2.0)
    y = np.round(mu + rng.normal(0, 2.0, n))
    cdf = _normal_cdf(mu, sd)
    a = pit_values(cdf, y, grid_step=1.0)
    b = pit_values(cdf, y, grid_step=1.0)
    assert np.array_equal(a, b)


def test_discrete_log_score_is_comparable_across_families():
    """Cell probability, not density.

    A density's units depend on the family's parameterisation, so densities
    across a normal and a t are not on one axis. The cell probability is a
    probability in both cases.
    """
    n = 50_000
    mu = np.zeros(n)
    y = np.round(student_t.rvs(df=4.0, size=n, random_state=9))
    sd_n = np.full(n, np.sqrt(4.0 / 2.0))

    def t_cdf(thr):
        return student_t.cdf(np.asarray(thr), df=4.0)

    ls_norm = calibration_suite(_normal_cdf(mu, sd_n), y, grid_step=1.0)["log_score"]
    ls_t = calibration_suite(t_cdf, y, grid_step=1.0)["log_score"]
    # The t generated the data, so it must score better on its own outcomes.
    assert ls_t < ls_norm


def test_log_score_is_nan_without_a_grid_step():
    """A continuous outcome has no cell probability. NaN, not a silent 0."""
    rng = np.random.default_rng(9)
    n = 1_000
    mu = np.zeros(n)
    sd = np.ones(n)
    s = calibration_suite(_normal_cdf(mu, sd), rng.normal(0, 1, n))
    assert np.isnan(s["log_score"])


# ---------------------------------------------------------------------------
# Loud failures (Rule #0.5)
# ---------------------------------------------------------------------------


def test_optional_columns_are_omitted_not_approximated():
    """Without `quantile`, coverage is absent rather than guessed.

    Approximating a t's quantiles with a Gaussian shape would be a silent
    fallback producing plausible-looking wrong coverage.
    """
    rng = np.random.default_rng(10)
    n = 1_000
    mu = np.zeros(n)
    sd = np.ones(n)
    s = calibration_suite(_normal_cdf(mu, sd), rng.normal(0, 1, n))
    for k in ("coverage_50", "coverage_90", "reliability_mae",
              "sharpness_iqr", "rmv", "crps"):
        assert k not in s


def test_non_finite_pit_raises():
    n = 100
    y = np.zeros(n)
    def bad(thr):
        return np.full(np.asarray(thr).shape, np.nan)

    with pytest.raises(ValueError, match="non-finite"):
        calibration_suite(bad, y)


def test_bad_sd_raises():
    rng = np.random.default_rng(11)
    n = 500
    mu = np.zeros(n)
    sd = np.ones(n)
    y = rng.normal(0, 1, n)
    with pytest.raises(ValueError, match="finite and positive"):
        calibration_suite(_normal_cdf(mu, sd), y, sd=np.zeros(n))


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_bad_grid_step_raises(bad):
    """NaN and inf too: a NaN step NaNs every PIT, an inf step makes them
    all 0.5, and both read as a result rather than as a failure."""
    with pytest.raises(ValueError, match="grid_step must be positive"):
        pit_values(lambda t: np.zeros_like(np.asarray(t)),
                   np.zeros(10), grid_step=bad)
