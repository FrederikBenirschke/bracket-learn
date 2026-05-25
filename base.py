"""BaseEstimator — sklearn-style contract for bracketlearn forecasters.

Provides `get_params` / `set_params` / `__repr__` / `clone()` semantics.
Modelled on `sklearn.base.BaseEstimator` so a future migration to
inheriting directly from sklearn is one import swap. We don't actually
inherit from sklearn here to keep the dependency optional at the protocol
layer (forecasters that don't use sklearn estimators internally don't
have to install scikit-learn).

Contract:
- ``__init__`` parameters are stored on ``self`` under the same name.
- ``get_params()`` returns a dict of constructor params; fitted state
  (attributes ending in ``_``) is excluded.
- ``clone(estimator)`` returns a fresh, *unfitted* copy of the estimator
  with the same constructor params. Used by the pipeline to give each
  CV fold its own forecaster instance.

The pipeline calls ``clone(forecaster)`` before each fold's fit so the
user-supplied forecaster instance is never mutated and folds cannot
contaminate one another via shared fitted state.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Self


class BaseEstimator:
    """Mixin providing sklearn-compatible param introspection + cloning.

    Subclasses (forecasters, lifters, calibrators) should declare their
    hyperparameters as plain ``__init__`` arguments. Fitted state goes on
    attributes whose name ends with ``_`` (sklearn convention) so
    ``get_params`` can distinguish them.
    """

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Inspect ``__init__`` and return the public hyperparameter names."""
        init = cls.__init__
        if init is object.__init__:
            return []
        sig = inspect.signature(init)
        return [
            name for name, p in sig.parameters.items()
            if name != "self" and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor params. If ``deep`` and a param is itself a
        BaseEstimator, prefix its params with ``<name>__`` (sklearn convention)."""
        out: dict[str, Any] = {}
        for name in self._get_param_names():
            value = getattr(self, name, None)
            out[name] = value
            if deep and isinstance(value, BaseEstimator):
                for sub_name, sub_val in value.get_params(deep=True).items():
                    out[f"{name}__{sub_name}"] = sub_val
        return out

    def set_params(self, **params: Any) -> Self:
        """In-place setter. Supports ``__``-nested params (sklearn convention)."""
        if not params:
            return self
        valid = self._get_param_names()
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            if "__" in key:
                head, _, tail = key.partition("__")
                if head not in valid:
                    raise ValueError(
                        f"Invalid parameter {head!r} for estimator {type(self).__name__}"
                    )
                nested.setdefault(head, {})[tail] = value
            else:
                if key not in valid:
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {type(self).__name__}; "
                        f"valid params: {valid}"
                    )
                setattr(self, key, value)
        for head, sub in nested.items():
            sub_est = getattr(self, head)
            if not isinstance(sub_est, BaseEstimator):
                raise ValueError(
                    f"Cannot set nested param on {head!r}: not a BaseEstimator"
                )
            sub_est.set_params(**sub)
        return self

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        # Drop params that equal the constructor default (sklearn-style brevity).
        try:
            sig = inspect.signature(type(self).__init__)
            defaults = {
                n: p.default for n, p in sig.parameters.items()
                if p.default is not inspect.Parameter.empty
            }
        except (TypeError, ValueError):
            defaults = {}
        shown = {
            k: v for k, v in params.items()
            if k not in defaults or not _equal(v, defaults[k])
        }
        body = ", ".join(f"{k}={v!r}" for k, v in shown.items())
        return f"{type(self).__name__}({body})"


def _equal(a: Any, b: Any) -> bool:
    """Defensive equality for repr-diffing. Numpy arrays need .all()."""
    try:
        import numpy as np
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.array_equal(a, b))
    except ImportError:
        pass
    try:
        return bool(a == b)
    except (ValueError, TypeError):
        return a is b


def clone(estimator: Any, *, safe: bool = True) -> Any:
    """Return a fresh, unfitted copy of ``estimator``.

    For ``BaseEstimator`` subclasses: reconstruct via ``__init__(**get_params())``
    after deep-copying any param whose value is mutable (this preserves
    sklearn-equivalent semantics where ``clone`` does not deep-copy
    *primitives* but does deep-copy *nested estimators*).

    For non-BaseEstimator objects: fall back to ``copy.deepcopy``. This
    catches LightGBM/NGBoost/sklearn estimators that get wrapped without
    inheriting from our base.

    Used by the pipeline before each fold's fit so the user-supplied
    forecaster instance is never mutated.
    """
    if isinstance(estimator, BaseEstimator):
        params = estimator.get_params(deep=False)
        new_params: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, BaseEstimator):
                new_params[k] = clone(v)
            else:
                new_params[k] = copy.deepcopy(v)
        return type(estimator)(**new_params)
    if not safe:
        return copy.deepcopy(estimator)
    # Best-effort deep-copy: handles SklearnPoint(LinearRegression()), etc.
    return copy.deepcopy(estimator)
