"""Input/target standardizers — the `Transformer` stage of a `Pipeline`.

`GroupByZScore` is the per-group standardized-anomaly transform behind the
weather normalization win: each row is mapped by a per-row affine
``v ↦ (v − center) / scale`` where ``center`` is a per-row anchor (e.g.
seasonal climatology) threaded in by the Pipeline and ``scale`` is a
per-group (e.g. per-station) constant learned at fit as ``std(y − center)``.

It implements the `Transformer` protocol (``fit`` / ``transform`` /
``transform_target`` / ``inverse_dist``): features go to z-space, the target
goes to z-space at fit, and the forecaster's predicted distribution is mapped
back to the original scale via ``DistributionForecast.affine`` — so a
forecaster never sees normalization and downstream bracket integration is
unchanged.

Per Rule #0.5: a group with too few observations falls back to the global
scale **explicitly**; scale is never a silent 1.0, and a non-positive global
scale raises.
"""

from __future__ import annotations

import numpy as np

_MIN_GROUP_OBS = 5


class IdentityTransformer:
    """No-op `Transformer`: passes features/target through unchanged and
    leaves the forecast unchanged. The degenerate transformer — also the
    shim shape for composing a plain sklearn X-only transformer (override
    ``transform``; target + inverse stay identity)."""

    def fit(self, X, y=None, *, ids=None, center=None, **kwargs):
        return self

    def transform(self, X, *, ids=None, center=None):
        return np.asarray(X, dtype=float)

    def transform_target(self, y):
        return np.asarray(y, dtype=float)

    def inverse_dist(self, dist):
        return dist


class GroupByZScore:
    """Per-group standardized-anomaly `Transformer`.

    Parameters
    ----------
    spread_cols
        Column indices that are *spreads* (e.g. an ensemble std): mapped
        ``v → v / scale`` (divide only, no centering). A negative/zero or
        NaN spread is left to the estimator's own validation.
    passthrough_cols
        Column indices left untouched (e.g. binary missing-indicator flags
        that must not be centered by a temperature climatology).
    level_cols
        Explicit *level* column indices: ``v → (v − center) / scale``. When
        ``None`` (default) every column that is neither a spread nor an
        explicit passthrough is treated as a level. When given, ONLY these
        indices are levels and all other (non-spread) columns pass through —
        so ``level_cols=()`` normalizes nothing on the feature side and the
        transform reduces to **target-only** standardization (``transform``
        passes X through, but ``transform_target`` / ``inverse_dist`` still
        z-score the target and map the forecast back). Target-only is the
        right mode when X carries heterogeneous columns (mixed vendor temps
        + non-temperature features) whose roles aren't known by index — the
        location confound lives in the *target*, and a tree/boosting model's
        feature splits are scale-invariant anyway.
    min_group
        Minimum per-group observations to trust a learned per-group scale;
        smaller groups use the global scale.

    ``center`` is the per-row anchor passed to ``fit``/``transform`` (default
    0 when absent); ``scale`` is learned per group from ``std(y − center)``.
    """

    def __init__(
        self,
        *,
        spread_cols: tuple[int, ...] = (),
        passthrough_cols: tuple[int, ...] = (),
        level_cols: tuple[int, ...] | None = None,
        min_group: int = _MIN_GROUP_OBS,
    ):
        self.spread_cols = tuple(spread_cols)
        self.passthrough_cols = tuple(passthrough_cols)
        self.level_cols = None if level_cols is None else tuple(level_cols)
        self.min_group = min_group
        # learned
        self.scale_by_: dict | None = None
        self.scale_global_: float | None = None
        # stamped by the most recent transform()
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    # ---- Transformer protocol ----

    def fit(self, X, y, *, ids, center=None, **kwargs):
        y = np.asarray(y, dtype=float)
        n = y.shape[0]
        c = self._center_array(center, n)
        ids = np.asarray(ids)
        anom = y - c
        finite = np.isfinite(anom)
        if finite.sum() < 2:
            raise ValueError("GroupByZScore.fit: <2 finite (y − center) rows")
        g = float(np.std(anom[finite], ddof=0))
        if not (np.isfinite(g) and g > 0):
            raise ValueError(
                f"GroupByZScore.fit: global scale not finite-positive (got {g})"
            )
        self.scale_global_ = g
        self.scale_by_ = {}
        for gid in np.unique(ids):
            m = (ids == gid) & finite
            ng = int(m.sum())
            if ng >= self.min_group:
                sg = float(np.std(anom[m], ddof=0))
                self.scale_by_[gid] = sg if (np.isfinite(sg) and sg > 0) else g
            else:
                self.scale_by_[gid] = g
        return self

    def transform(self, X, *, ids, center=None):
        if self.scale_by_ is None:
            raise RuntimeError("GroupByZScore.transform called before fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"GroupByZScore: X must be 2-D; got {X.shape}")
        n, f = X.shape
        c = self._center_array(center, n)
        s = self._scale_array(np.asarray(ids))
        self._center, self._scale = c, s            # stamp for target/inverse
        spread = set(self.spread_cols)
        passth = set(self.passthrough_cols)
        level = None if self.level_cols is None else set(self.level_cols)
        out = np.empty_like(X)
        for j in range(f):
            if j in spread:
                out[:, j] = X[:, j] / s
            elif j in passth:
                out[:, j] = X[:, j]
            elif level is None:                      # default: level = complement
                out[:, j] = (X[:, j] - c) / s
            elif j in level:                         # explicit level
                out[:, j] = (X[:, j] - c) / s
            else:                                    # explicit levels given, j not one → passthrough
                out[:, j] = X[:, j]
        return out

    def transform_target(self, y):
        if self._center is None or self._scale is None:
            raise RuntimeError("GroupByZScore.transform_target before transform")
        return (np.asarray(y, dtype=float) - self._center) / self._scale

    def inverse_dist(self, dist):
        if self._center is None or self._scale is None:
            raise RuntimeError("GroupByZScore.inverse_dist before transform")
        return dist.affine(shift=self._center, scale=self._scale)

    # ---- helpers ----

    def _center_array(self, center, n: int) -> np.ndarray:
        if center is None:
            return np.zeros(n, dtype=float)
        c = np.asarray(center, dtype=float)
        if c.ndim == 0:
            c = np.full(n, float(c))
        if c.shape != (n,):
            raise ValueError(
                f"GroupByZScore: center must be scalar or length-N={n}; got {c.shape}"
            )
        return c

    def _scale_array(self, ids: np.ndarray) -> np.ndarray:
        g = self.scale_global_
        return np.array(
            [self.scale_by_.get(gid, g) for gid in ids], dtype=float
        )
