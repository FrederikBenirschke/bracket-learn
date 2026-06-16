"""Bracket-native DistForecaster.

CumulativeBinary fits binary cutpoint classifiers directly on each row's
own bracket grid. The bracket-emitting *combiners* that consume upstream
forecasts (TailSpecialist, CDFBoostBracket, LinearPoolDist) now live in
``bracketlearn.trainers.combiners``.

The old ``BracketClassifier`` / ``BracketRegressor`` classes were
removed in v0.5.0 — they conflated per-row -> per-(row, bracket)
expansion with model fitting and hardcoded the target as a bracket-hit
indicator. The two concerns now live separately: callers compose
``bracketlearn.transformers.BracketExpander`` with any sklearn-style
estimator they like. See ``BracketExpander`` for the migration recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    DistributionForecast,
    ProvenanceMeta,
    bracket_probs_from_cdf_at_edges,
)

# ---------------------------------------------------------------------------
# CumulativeBinary — one classifier on (X ⊕ cutpoint) → 1[y ≤ cutpoint].
# ---------------------------------------------------------------------------


def _validate_cumulative_grids(
    cutpoints_by_id: dict[Any, np.ndarray],
    outer_edges_by_id: dict[Any, tuple[float, float]],
    *,
    owner: str = "CumulativeBinary",
) -> None:
    """Validate the per-row cutpoint grids + outer edges (the id-keyed dicts now
    passed to fit/predict). Raises on empty / non-1-D / non-monotone cutpoints, a
    missing outer-edge entry, or outer edges that don't bracket the cutpoints."""
    if not isinstance(cutpoints_by_id, dict) or not cutpoints_by_id:
        raise ValueError(
            f"{owner} needs a non-empty cutpoints_by_id dict (id → 1-D cutpoint array)"
        )
    if not isinstance(outer_edges_by_id, dict) or not outer_edges_by_id:
        raise ValueError(
            f"{owner} needs a non-empty outer_edges_by_id dict (id → (lo, hi) tuple)"
        )
    for k, cuts in cutpoints_by_id.items():
        cuts_arr = np.asarray(cuts, dtype=float)
        if cuts_arr.ndim != 1 or cuts_arr.size == 0:
            raise ValueError(
                f"cutpoints_by_id[{k!r}] must be 1-D non-empty; got shape {cuts_arr.shape}"
            )
        if np.any(np.diff(cuts_arr) <= 0):
            raise ValueError(f"cutpoints_by_id[{k!r}] must be strictly increasing")
        if k not in outer_edges_by_id:
            raise ValueError(
                f"cutpoints_by_id has id {k!r} but outer_edges_by_id does not"
            )
        lo, hi = outer_edges_by_id[k]
        if not (lo < cuts_arr[0]):
            raise ValueError(
                f"outer_edges_by_id[{k!r}][0]={lo} must be < cutpoints[0]={cuts_arr[0]}"
            )
        if not (hi > cuts_arr[-1]):
            raise ValueError(
                f"outer_edges_by_id[{k!r}][1]={hi} must be > cutpoints[-1]={cuts_arr[-1]}"
            )


@dataclass
class CumulativeBinary(BaseEstimator):
    """Single LightGBM binary classifier on augmented features.

    Fits one classifier over (X, cutpoint) → 1[y ≤ cutpoint], then at
    predict time queries P(y ≤ k) for each cutpoint k in the row's own
    grid and emits a per-row bracket-backed dist.

    Data contract — construction is hyperparameters only
    ----------------------------------------------------
    Each market/event has its own cutpoint grid (the interior bracket
    edges) and its own outer-edge pair (left/right boundaries
    absorbing tail mass). Both are passed as id-keyed dicts **at call
    time** (alongside ``X`` / ``y``), not at construction::

        fit(X, y, *, ids, cutpoints_by_id, outer_edges_by_id, sample_weight=None)
        predict_dist(X, *, ids, timestamps, cutpoints_by_id, outer_edges_by_id)

      ``cutpoints_by_id``:    dict mapping row id → 1-D float array of
                              cutpoints (length K_i ≥ 1).
      ``outer_edges_by_id``:  dict mapping row id → (lo_i, hi_i) tuple
                              with ``lo_i < cutpoints[0]`` and
                              ``hi_i > cutpoints[-1]``.

    Both dicts must cover every id in that call (they may cover more —
    fit/predict select by the ``ids`` you hand them). Each row i
    contributes K_i augmented training examples to the LGBM model — the
    examples for different rows may have different cutpoint counts and
    spacings. Cutpoint values are passed as a feature, so the model
    naturally generalises across grids.

    The model itself is global (one LGBM); only the per-row augmentation
    differs. Predict-time emits a BracketForecast on each row's full
    ladder ``[lo_i, cutpoints_i..., hi_i]``. Under ``WalkForward`` the dicts
    are forwarded verbatim — pass them once to ``fit_predict``.
    """

    n_estimators: int = 80
    learning_rate: float = 0.05
    num_leaves: int = 7
    min_child_samples: int = 100
    monotone: bool = True
    name: str = "CumulativeBinary"
    model_: Any = field(default=None, init=False)
    _requires_explicit_ids = True   # grids keyed by id; never auto-fill arange(N)

    def _augment(
        self,
        X: np.ndarray,
        ids: np.ndarray,
        cutpoints_by_id: dict[Any, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Build the per-row augmented design matrix. Returns
        ``(X_aug, offsets, per_row_cuts)`` where ``offsets[i]:offsets[i+1]``
        covers row i's augmented examples and ``per_row_cuts[i]`` is its
        cutpoint array."""
        N = X.shape[0]
        per_row_cuts = []
        missing = []
        for k in ids:
            try:
                per_row_cuts.append(np.asarray(cutpoints_by_id[k], dtype=float))
            except KeyError:
                missing.append(k)
        if missing:
            raise KeyError(
                f"CumulativeBinary: cutpoints_by_id missing {len(missing)} id(s); "
                f"first: {missing[:3]}"
            )
        K_per_row = np.array([c.size for c in per_row_cuts], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(K_per_row)])
        M = int(offsets[-1])                       # total augmented rows
        n_feat = X.shape[1]
        X_aug = np.empty((M, n_feat + 1), dtype=float)
        for i in range(N):
            sl = slice(offsets[i], offsets[i + 1])
            X_aug[sl, :n_feat] = X[i]
            X_aug[sl, n_feat] = per_row_cuts[i]
        return X_aug, offsets, per_row_cuts

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        cutpoints_by_id: dict[Any, np.ndarray],
        outer_edges_by_id: dict[Any, tuple[float, float]],
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        import lightgbm as lgb

        _validate_cumulative_grids(cutpoints_by_id, outer_edges_by_id, owner=self.name)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        if X.shape[0] != y.shape[0] or X.shape[0] != ids.shape[0]:
            raise ValueError(
                f"shape mismatch: X N={X.shape[0]} y N={y.shape[0]} ids N={ids.shape[0]}"
            )
        X_aug, offsets, per_row_cuts = self._augment(X, ids, cutpoints_by_id)
        N = X.shape[0]
        y_aug = np.empty(X_aug.shape[0], dtype=int)
        for i in range(N):
            sl = slice(offsets[i], offsets[i + 1])
            y_aug[sl] = (y[i] <= per_row_cuts[i]).astype(int)
        n_feat = X_aug.shape[1]
        monotone = [0] * (n_feat - 1) + [1] if self.monotone else None
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            objective="binary",
            verbose=-1,
            monotone_constraints=monotone,
        )
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            sw_aug = np.empty(X_aug.shape[0], dtype=float)
            for i in range(N):
                sl = slice(offsets[i], offsets[i + 1])
                sw_aug[sl] = sw[i]
            self.model_.fit(X_aug, y_aug, sample_weight=sw_aug)
        else:
            self.model_.fit(X_aug, y_aug)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        cutpoints_by_id: dict[Any, np.ndarray],
        outer_edges_by_id: dict[Any, tuple[float, float]],
    ) -> DistributionForecast:
        if self.model_ is None:
            raise RuntimeError("CumulativeBinary.predict_dist called before fit")
        _validate_cumulative_grids(cutpoints_by_id, outer_edges_by_id, owner=self.name)
        X = np.asarray(X, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        N = X.shape[0]
        X_aug, offsets, per_row_cuts = self._augment(X, ids, cutpoints_by_id)
        proba_aug = self.model_.predict_proba(X_aug)[:, 1]
        # Reassemble per-row, then build each row's ladder + probs.
        # Rows can have different K_i — collect into a padded 2-D
        # BracketForecast (NaN-padded ragged columns).
        K_per_row = np.array([c.size for c in per_row_cuts], dtype=int)
        # Row ladder is [lo, cutpoints..., hi]: K_i + 2 edges, K_i + 1 bins.
        B_per_row = K_per_row + 1
        B_max = int(B_per_row.max())
        edges = np.full((N, B_max + 1), np.nan, dtype=float)
        probs = np.full((N, B_max), np.nan, dtype=float)
        for i in range(N):
            sl = slice(offsets[i], offsets[i + 1])
            p_cuts = proba_aug[sl]
            # Per-row isotonic repair on cutpoint-wise CDF.
            p_cuts = np.maximum.accumulate(p_cuts)
            lo, hi = outer_edges_by_id[ids[i]]
            row_edges = np.concatenate([[lo], per_row_cuts[i], [hi]])
            cdf_at_edges = np.concatenate([[0.0], p_cuts, [1.0]])
            row_probs = bracket_probs_from_cdf_at_edges(
                cdf_at_edges[None, :], source="CumulativeBinary.predict_dist",
            )[0]
            B_i = row_edges.size - 1
            edges[i, : B_i + 1] = row_edges
            probs[i, : B_i] = row_probs
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=ids, timestamps=timestamps,
            provenance=prov,
        )


