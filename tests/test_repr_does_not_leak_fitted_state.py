"""``repr()`` must show constructor parameters, never fitted state.

``BaseEstimator.__repr__`` elides parameters that equal their default, the
sklearn convention. Every estimator here is also a ``@dataclass``, and the
generated ``__dataclass_repr__`` sits EARLIER in the MRO, so the hand-written
one was dead code on all 24 of them. What printed instead was every field
including the fitted ones: ``QuantileReg`` after ``fit`` rendered 1692
characters embedding eleven ``LGBMRegressor`` reprs, and
``HierarchicalNormal`` exposed nine fitted arrays.

That is what a notebook prints for a bare expression and what lands in a
leaderboard or log line, so the leak was in the default output path rather
than in a debugging corner.

The fix is ``@dataclass(repr=False)``, which is easy to omit on a new
estimator and produces no error when omitted. Hence this test.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

import bracketlearn as bl
from bracketlearn.base import BaseEstimator


def _estimator_classes():
    for name in sorted(dir(bl)):
        obj = getattr(bl, name)
        if (inspect.isclass(obj) and issubclass(obj, BaseEstimator)
                and obj is not BaseEstimator):
            yield name, obj


def _constructible():
    for name, cls in _estimator_classes():
        try:
            yield name, cls()
        except Exception:
            continue          # needs required args; covered by the class test


def test_there_are_estimators_to_check():
    """Guard against the collection silently going empty."""
    assert len(list(_constructible())) >= 20


@pytest.mark.parametrize("name,cls", list(_estimator_classes()))
def test_estimator_dataclasses_disable_the_generated_repr(name, cls):
    """A dataclass estimator must set ``repr=False``.

    Checked structurally rather than by inspecting output, so a new estimator
    that happens to have no fitted fields yet still fails here rather than
    passing until someone adds one.
    """
    if not dataclasses.is_dataclass(cls):
        pytest.skip(f"{name} is not a dataclass")
    assert "__repr__" not in cls.__dict__, (
        f"{name} carries a dataclass-generated __repr__, which shadows "
        f"BaseEstimator.__repr__ and prints fitted state. Decorate it "
        f"@dataclass(repr=False)."
    )


@pytest.mark.parametrize("name,inst", list(_constructible()))
def test_repr_contains_no_fitted_field(name, inst):
    """No trailing-underscore field may appear in the rendered repr."""
    fitted = [
        f.name for f in dataclasses.fields(inst)
        if f.name.endswith("_") and not f.name.startswith("_")
    ] if dataclasses.is_dataclass(inst) else []
    r = repr(inst)
    leaked = [f for f in fitted if f + "=" in r]
    assert not leaked, f"{name} repr leaks fitted state: {leaked}\n  {r}"


def test_repr_survives_fitting_and_stays_short():
    """The regression that motivated this: a fitted repr must not explode."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 3))
    y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 1, 60)
    est = bl.EMOS(fit_method="crps_nelder_mead")
    before = repr(est)
    est.fit(X, y)
    after = repr(est)
    assert after == before, (
        "fitting changed the repr, so fitted state is reaching it")
    assert len(after) < 120


def test_repr_still_shows_non_default_parameters():
    """repr=False must not silence the parameters, only the fitted state."""
    r = repr(bl.EMOS(fit_method="crps_nelder_mead"))
    assert "crps_nelder_mead" in r
    assert r.startswith("EMOS(")
    assert repr(bl.EMOS()) == "EMOS()", (
        "defaults should be elided, the sklearn convention "
        "BaseEstimator.__repr__ implements")
