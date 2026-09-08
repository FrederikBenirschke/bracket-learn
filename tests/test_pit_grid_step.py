"""Discrete-outcome PIT: the continuity correction and its reference value.

The trap these pin: on a grid, the plain F(y) is not the Rosenblatt PIT, and
the calibrated var(PIT) is strictly below 1/12. Judging a corrected variance
against 1/12 manufactures an "underdispersed" verdict for a forecast that is
in fact calibrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn import Pipeline, WalkForward
from bracketlearn.calibration import NEUTRAL_PIT_VAR, neutral_pit_var
from bracketlearn.trainers import EMOS


def _integer_panel(n=4000, sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    ens = rng.normal(70.0, 5.0, n)
    y = np.round(ens + rng.normal(0.0, sigma, n))
    X = np.column_stack([ens, np.full(n, sigma)])
    return X, y


def _fit_dist(X, y):
    return EMOS().fit(X, y).predict_dist(X)


class TestContinuityCorrection:
    def test_grid_step_none_is_the_plain_cdf(self):
        X, y = _integer_panel()
        d = _fit_dist(X, y)
        np.testing.assert_allclose(d.pit(y), d.cdf_at(y))

    def test_grid_step_is_the_mid_interval_form(self):
        X, y = _integer_panel()
        d = _fit_dist(X, y)
        h = 0.5
        expected = d.cdf_at(y - h) + 0.5 * (d.cdf_at(y + h) - d.cdf_at(y - h))
        np.testing.assert_allclose(d.pit(y, grid_step=1.0), expected)

    def test_correction_lowers_the_variance_on_a_grid(self):
        """The naive form is biased UP; the corrected one sits below 1/12."""
        X, y = _integer_panel(sigma=1.0)
        d = _fit_dist(X, y)
        naive = float(d.pit(y).var())
        corrected = float(d.pit(y, grid_step=1.0).var())
        assert corrected < naive
        assert corrected < NEUTRAL_PIT_VAR

    def test_deterministic(self):
        """No RNG: repeated calls agree exactly (not the randomised PIT)."""
        X, y = _integer_panel()
        d = _fit_dist(X, y)
        np.testing.assert_array_equal(
            d.pit(y, grid_step=1.0), d.pit(y, grid_step=1.0)
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_grid_step_raises(self, bad):
        X, y = _integer_panel(n=200)
        d = _fit_dist(X, y)
        with pytest.raises(ValueError, match="grid_step"):
            d.pit(y, grid_step=bad)


class TestNeutralReference:
    """The reference value, checked against an EXACTLY calibrated forecast.

    A fitted model is not exactly calibrated (its sigma is estimated), so it
    cannot pin the reference to better than the fit's own error. These use a
    forecast that is correct by construction: y is drawn from the very
    N(mu, sigma) being scored, then rounded.
    """

    @pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
    def test_calibrated_forecast_lands_on_its_own_target(self, sigma):
        """var(PIT_mid) tracks 1/12 - E[p^2]/12, NOT 1/12."""
        from scipy.stats import norm

        rng = np.random.default_rng(0)
        n = 200_000
        mu = np.zeros(n)
        y = np.round(rng.normal(mu, sigma))

        lo = norm.cdf(y - 0.5, mu, sigma)
        hi = norm.cdf(y + 0.5, mu, sigma)
        measured = float((lo + 0.5 * (hi - lo)).var())
        target = neutral_pit_var(hi - lo)

        assert target < NEUTRAL_PIT_VAR
        assert abs(measured - target) < 1e-3
        # Judging the same number against 1/12 is the larger error, which is
        # the reason the neutral value is reported at all.
        assert abs(measured - NEUTRAL_PIT_VAR) > abs(measured - target)

    def test_a_fitted_model_is_near_but_not_on_the_target(self):
        """In-sample fit error is larger than the formula's own error.

        Pins the distinction the test above rests on: the gap a real model
        shows against the neutral value is dominated by ITS calibration, not
        by the reference being wrong.
        """
        X, y = _integer_panel(sigma=1.0, n=40_000)
        d = _fit_dist(X, y)
        cell = d.cdf_at(y + 0.5) - d.cdf_at(y - 0.5)
        target = neutral_pit_var(cell)
        measured = float(d.pit(y, grid_step=1.0).var())
        assert target < NEUTRAL_PIT_VAR
        assert abs(measured - target) < 0.01


class TestPipelineWiring:
    def _result(self):
        X, y = _integer_panel(n=3000)
        model = Pipeline([EMOS()], name="emos")
        wf = WalkForward(cv="expanding-window", n_folds=3)
        return wf.fit_predict([model], X, y, ids=np.arange(len(y)),
                              timestamps=np.arange(len(y))), y

    def test_score_reports_neutral_and_excess(self):
        res, y = self._result()
        row = res.score(y, metrics=["pit"], grid_step=1.0)["emos"]
        for k in ("pit_mean", "pit_var", "pit_var_neutral", "pit_var_excess"):
            assert k in row
        assert row["pit_var_neutral"] < NEUTRAL_PIT_VAR
        assert row["pit_var_excess"] == pytest.approx(
            row["pit_var"] - row["pit_var_neutral"]
        )

    def test_grid_step_changes_the_verdict_not_just_the_number(self):
        """Without grid_step the same forecast is judged against 1/12."""
        res, y = self._result()
        naive = res.score(y, metrics=["pit"])["emos"]
        corrected = res.score(y, metrics=["pit"], grid_step=1.0)["emos"]
        assert naive["pit_var_neutral"] == NEUTRAL_PIT_VAR
        assert corrected["pit_var_neutral"] < NEUTRAL_PIT_VAR
        # The corrected excess is the smaller distortion.
        assert abs(corrected["pit_var_excess"]) < abs(
            corrected["pit_var"] - NEUTRAL_PIT_VAR
        )

    def test_to_table_columns_do_not_collide(self):
        res, y = self._result()
        table = res.to_table(y, metrics=["pit"], grid_step=1.0)
        header = table.splitlines()[0]
        assert "pit_var_neutral" in header
        assert "pit_var_excess" in header
        # header cells stay separated
        assert "pit_varpit_var_neutral" not in header
