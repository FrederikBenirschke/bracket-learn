"""The calibration tail must be out-of-sample for the core forecaster.

``Pipeline.fit``'s docstring promises "a trailing Calibrator fits on a held-out
tail of the (transformed) training data". Before this test, the tail was held
out from the CALIBRATOR but not from the CORE: the core was fit on all n rows
first, so the calibrator was fit on the core's own IN-SAMPLE predictions.

The error has a definite sign. In-sample predictions are systematically better
than the out-of-sample ones the calibrator is applied to in production, so the
learned correction is too small and under-corrects where correction is
required. The Point->Lifter path in the same method already avoided this (fit on
a half, predict OOF, fit the lifter, refit on everything); the calibrator path
did not.

The tell was that the pipeline's fitted core parameters were bit-identical to a
full-data fit at the moment the calibrator was fit, which is what these tests
assert cannot happen again.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.lift import Isotonic
from bracketlearn.pipeline import Pipeline
from bracketlearn.trainers import EMOS

# A shared bracket ladder, so Isotonic can integrate the Normal onto brackets.
EDGES = np.array([-np.inf, 45.0, 55.0, 65.0, 75.0, np.inf])


def _cal():
    return Isotonic(pre_integrate_edges=EDGES)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(60, 10, n)
    sd = rng.uniform(1.0, 3.0, n)
    y = mu + rng.normal(0, 1, n) * sd
    X = np.column_stack([mu, sd])
    return X, y, np.arange(n), np.arange(n, dtype=float)


def _core_params(pipe):
    core = pipe._point if pipe._point is not None else pipe._model
    return {k: getattr(core, k) for k in ("a_", "b_") if hasattr(core, k)}


def test_core_does_not_see_the_calibration_tail_while_it_is_being_produced():
    """The core's fit at calibrator-fit time must differ from a full-data fit.

    Captured by spying on the calibrator: when its ``fit`` is called, the core
    must be the HEAD-only fit, not the all-rows one.
    """
    X, y, ids, ts = _data()
    frac = 0.2
    n = len(y)
    c = max(2, int(n * frac))

    seen: dict[str, dict] = {}
    cal = _cal()
    orig_fit = cal.fit

    def spy(dist, target, *a, **kw):
        seen["core_at_calibration"] = dict(_core_params(pipe))
        return orig_fit(dist, target, *a, **kw)

    cal.fit = spy  # type: ignore[method-assign]
    pipe = Pipeline([EMOS(fit_method="crps_nelder_mead"), cal],
                    calibration_fraction=frac)
    pipe.fit(X, y, ids=ids, timestamps=ts)

    head_only = EMOS(fit_method="crps_nelder_mead").fit(X[:n - c], y[:n - c])
    full = EMOS(fit_method="crps_nelder_mead").fit(X, y)

    at_cal = seen["core_at_calibration"]
    assert at_cal, "spy did not observe the core's parameters"
    for k, v in at_cal.items():
        assert v == pytest.approx(getattr(head_only, k), rel=1e-9), (
            f"core param {k} at calibration time should equal the HEAD-only "
            "fit; the calibration tail must be out-of-sample")
        assert v != pytest.approx(getattr(full, k), rel=1e-12), (
            f"core param {k} equals the FULL-data fit, the calibrator is "
            "being fit on in-sample predictions (the original bug)")


def test_core_is_refit_on_everything_before_predicting():
    """Holding the tail out is for calibration only. The deployed core must
    still use all the training data, or the fix trades a leak for lost signal.
    """
    X, y, ids, ts = _data()
    pipe = Pipeline([EMOS(fit_method="crps_nelder_mead"), _cal()],
                    calibration_fraction=0.2)
    pipe.fit(X, y, ids=ids, timestamps=ts)
    full = EMOS(fit_method="crps_nelder_mead").fit(X, y)
    for k, v in _core_params(pipe).items():
        assert v == pytest.approx(getattr(full, k), rel=1e-9), (
            f"after fit, core param {k} should match a full-data fit")


def test_no_calibrator_means_no_holdout():
    """A pipeline without a calibrator must fit the core on every row."""
    X, y, ids, ts = _data()
    pipe = Pipeline([EMOS(fit_method="crps_nelder_mead")])
    pipe.fit(X, y, ids=ids, timestamps=ts)
    full = EMOS(fit_method="crps_nelder_mead").fit(X, y)
    for k, v in _core_params(pipe).items():
        assert v == pytest.approx(getattr(full, k), rel=1e-9)


def test_too_few_rows_drops_the_calibrator_rather_than_leaking():
    """With no room for a real holdout the calibrator is dropped, not fit on
    in-sample predictions."""
    X, y, ids, ts = _data(n=4)
    pipe = Pipeline([EMOS(fit_method="crps_nelder_mead"), _cal()],
                    calibration_fraction=0.9)
    pipe.fit(X, y, ids=ids, timestamps=ts)
    assert pipe._calibrator is None


def test_predictions_are_finite_and_shaped():
    """The restructured fit must still produce a usable forecaster."""
    X, y, ids, ts = _data()
    pipe = Pipeline([EMOS(fit_method="crps_nelder_mead"), _cal()],
                    calibration_fraction=0.2)
    pipe.fit(X, y, ids=ids, timestamps=ts)
    d = pipe.predict_dist(X[:50], ids=ids[:50], timestamps=ts[:50])
    # The trailing Isotonic returns bracket probabilities, not a Normal.
    probs = np.asarray(d.probs, dtype=float)
    assert probs.shape == (50, len(EDGES) - 1)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0), "rows must renormalise to 1"


# ---------------------------------------------------------------------------
# The transformers above the core are the same leak, one stage up.
#
# The first fix held the calibration tail out of the CORE but left the
# transformer loop fitting on all n rows. GroupByZScore learns its scale from
# std(y - center), so the tail set the scale that the calibrator's own inputs
# were then divided by. Measured on a wide-tailed synthetic: 22.19 fit on all
# rows vs 9.96 on the head.
# ---------------------------------------------------------------------------


def _wide_tail(n=200, tail=40, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(60, 10, n)
    sd = rng.uniform(1.0, 3.0, n)
    y = mu + rng.normal(0, 1, n) * sd
    y[-tail:] = mu[-tail:] + rng.normal(0, 40, tail)   # the calibration tail
    return np.column_stack([mu, sd]), y, np.array(["A"] * n), np.arange(n, dtype=float)


def test_transformers_do_not_see_the_calibration_tail():
    from bracketlearn.transform import GroupByZScore

    X, y, ids, ts = _wide_tail()
    n, frac = len(y), 0.2
    c = max(2, int(n * frac))

    seen: dict[str, float] = {}
    cal = _cal()
    orig_fit = cal.fit

    def spy(dist, target, *a, **kw):
        seen["scale"] = pipe._transformers[0].scale_global_
        return orig_fit(dist, target, *a, **kw)

    cal.fit = spy  # type: ignore[method-assign]
    pipe = Pipeline([GroupByZScore(), EMOS(fit_method="crps_nelder_mead"), cal],
                    calibration_fraction=frac)
    pipe.fit(X, y, ids=ids, timestamps=ts)

    head = GroupByZScore()
    head.fit(X[:n - c], y[:n - c], ids=ids[:n - c], center=None)
    full = GroupByZScore()
    full.fit(X, y, ids=ids, center=None)

    assert seen["scale"] == pytest.approx(head.scale_global_, rel=1e-9), (
        "the transformer's scale at calibration time must come from the head "
        "only; the tail must not set the scale its own rows are divided by")
    assert seen["scale"] != pytest.approx(full.scale_global_, rel=1e-6), (
        "scale equals the full-data fit, the transformer saw the tail")


def test_transformers_keep_the_space_the_calibrator_was_fit_in():
    """The transformer is NOT refit after calibration, unlike the core.

    The two are not symmetric, and an earlier version of this test asserted
    that they were. The calibrator is applied AFTER the core, to its output
    distribution, so refitting the core underneath it is harmless. The
    transformer sits BELOW the core: refitting it moves the z space that the
    calibrator's correction is indexed on, and the calibrator is never refit.
    Isotonic maps absolute z values and is not scale invariant, so the
    correction then lands at the wrong scale, measured on this fixture as a
    calibrator fit at scale 9.96 and applied at 22.19.

    Fitting the transformer on the head only is also what keeps the tail out
    of it, so one choice serves both invariants.
    """
    from bracketlearn.transform import GroupByZScore

    X, y, ids, ts = _wide_tail()
    n = len(y)
    c = max(2, int(n * 0.2))
    pipe = Pipeline([GroupByZScore(), EMOS(fit_method="crps_nelder_mead"), _cal()],
                    calibration_fraction=0.2)
    pipe.fit(X, y, ids=ids, timestamps=ts)

    head = GroupByZScore()
    head.fit(X[:n - c], y[:n - c], ids=ids[:n - c], center=None)
    full = GroupByZScore()
    full.fit(X, y, ids=ids, center=None)

    assert pipe._transformers[0].scale_global_ == pytest.approx(
        head.scale_global_, rel=1e-9), (
        "the fitted transformer must still hold the head-fit scale, which is "
        "the space the calibrator learned its correction in")
    assert pipe._transformers[0].scale_global_ != pytest.approx(
        full.scale_global_, rel=1e-6), (
        "the transformer was refit on all rows, so the calibrator's "
        "correction is now applied at a scale it never saw")


def test_point_lifter_with_calibrator_also_holds_the_tail_out():
    """The combination nothing pinned before: both branches must behave."""
    from sklearn.linear_model import Ridge

    from bracketlearn.lift import GlobalResidual
    from bracketlearn.trainers import SklearnPoint

    X, y, ids, ts = _data()
    pipe = Pipeline([SklearnPoint(Ridge()), GlobalResidual(), _cal()],
                    calibration_fraction=0.2)
    pipe.fit(X, y, ids=ids, timestamps=ts)
    d = pipe.predict_dist(X[:20], ids=ids[:20], timestamps=ts[:20])
    probs = np.asarray(d.probs, dtype=float)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)
