"""BaseEstimator — sklearn-style contract for bracketlearn forecasters.

Inherits from ``sklearn.base.BaseEstimator`` so bracketlearn estimators
are isinstance-compatible with sklearn helpers (``check_is_fitted``,
``__sklearn_tags__``, anything that does
``isinstance(est, sklearn.base.BaseEstimator)``). Note that
``sklearn.utils.estimator_checks.check_estimator`` will NOT pass on a
bracketlearn forecaster — our ``predict`` returns a ``PointForecast``,
``predict_dist`` returns a ``DistributionForecast``, neither of which is
the ndarray sklearn expects. The isinstance interop is the actual win
of subclassing; ``check_estimator`` compliance is a separate workstream.

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
import functools
import inspect
from typing import Any, Self

from sklearn.base import BaseEstimator as _SklearnBaseEstimator


def _auto_fill_ids_ts(method):
    """Wrap a method so it auto-fills `ids` and `timestamps` kwargs.

    Lets sklearn-style callers write ``est.fit(X, y)`` or
    ``est.predict(X)`` without supplying the bracketlearn-specific
    ``ids=`` / ``timestamps=`` kwargs. We infer them from the first
    positional argument (X) — ``ids = np.arange(N)``,
    ``timestamps = np.arange(N, dtype=float)``.

    Idempotent: if the caller explicitly passes ids/timestamps we
    leave them alone. Does nothing for methods whose signature
    doesn't actually take these kwargs.
    """
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return method
    params = sig.parameters
    takes_ids = "ids" in params
    takes_ts = "timestamps" in params
    if not (takes_ids or takes_ts):
        return method

    @functools.wraps(method)
    def wrapper(self, X, *args, **kwargs):
        import numpy as np

        X_arr = np.asarray(X) if not hasattr(X, "shape") else X
        try:
            N = X_arr.shape[0]
        except (AttributeError, IndexError):
            return method(self, X, *args, **kwargs)
        if takes_ids and "ids" not in kwargs:
            kwargs["ids"] = np.arange(N)
        if takes_ts and "timestamps" not in kwargs:
            kwargs["timestamps"] = np.arange(N, dtype=float)
        return method(self, X, *args, **kwargs)

    wrapper.__wrapped__ = method  # type: ignore[attr-defined]
    return wrapper


class BaseEstimator(_SklearnBaseEstimator):
    """sklearn-compatible base for bracketlearn forecasters / lifters / calibrators.

    Inherits ``get_params`` / ``set_params`` / ``__repr__`` / ``_get_param_names``
    from ``sklearn.base.BaseEstimator``. Adds bracketlearn-specific
    behaviour: auto-fill of ``ids`` / ``timestamps`` kwargs,
    ``__sklearn_is_fitted__`` based on ``_``-suffixed attributes,
    ``_record_input_signature`` helper for ``n_features_in_``.

    Subclasses (forecasters, lifters, calibrators) should declare their
    hyperparameters as plain ``__init__`` arguments. Fitted state goes on
    attributes whose name ends with ``_`` (sklearn convention) so
    ``get_params`` can distinguish them.

    Subclasses get the following sklearn-compat behaviour for free:

    - ``fit`` / ``predict`` / ``predict_dist`` wrapped so callers may omit
      ``ids=`` / ``timestamps=`` (auto-filled to ``arange(N)``).
    - ``__sklearn_is_fitted__`` returns True iff any attribute ending in
      ``_`` (sklearn convention for fitted state) is set to a non-None
      value.
    - ``n_features_in_`` / ``feature_names_in_`` set on ``fit`` via the
      ``_record_input_signature`` helper (subclasses may opt in by
      calling it from their ``fit``).
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("fit", "predict", "predict_dist"):
            method = cls.__dict__.get(name)
            if method is None or not callable(method):
                continue
            if getattr(method, "__wrapped__", None) is not None:
                continue  # already wrapped, e.g. via decorator
            setattr(cls, name, _auto_fill_ids_ts(method))

    def __sklearn_is_fitted__(self) -> bool:
        """sklearn convention: ``hasattr(est, attr_ending_in_underscore)``.

        Returns True iff any attribute ending in ``_`` (but not ``__``)
        is set to a non-None value. Lets ``sklearn.utils.validation.
        check_is_fitted(est)`` work on bracketlearn estimators.
        """
        for name in vars(self):
            if name.endswith("_") and not name.endswith("__"):
                if getattr(self, name, None) is not None:
                    return True
        return False

    def _record_input_signature(self, X) -> None:
        """Set ``n_features_in_`` and ``feature_names_in_`` from X.

        Call from a subclass's ``fit`` to populate these sklearn
        attributes. ``feature_names_in_`` is only set when X is a
        pandas DataFrame (matches sklearn's behaviour).
        """
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return
        if hasattr(X, "shape") and len(X.shape) >= 2:
            self.n_features_in_ = int(X.shape[1])  # type: ignore[attr-defined]
        if hasattr(X, "columns"):
            try:
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                pass

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
