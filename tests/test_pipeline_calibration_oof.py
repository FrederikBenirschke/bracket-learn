"""The calibration tail must be out-of-sample for the core forecaster.

``Pipeline.fit``'s docstring promises "a trailing Calibrator fits on a held-out
tail of the (transformed) training data". Before this test, the tail was held
out from the CALIBRATOR but not from the CORE: the core was fit on all n rows
first, so the calibrator was fit on the core's own IN-SAMPLE predictions.

That matters in one direction. In-sample predictions are systematically better
than the out-of-sample ones the calibrator is applied to in production, so the
correction it learns is too small — it under-corrects exactly when it is
needed. The Point->Lifter path in the same method already avoided this (fit on
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
            f"core param {k} equals the FULL-data fit — the calibrator is "
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
