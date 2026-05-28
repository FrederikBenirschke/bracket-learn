"""Stateful transformers that expand per-row data into per-(row, bracket).

Background
----------
``BracketClassifier`` and ``BracketRegressor`` (and several siblings)
historically conflated two concerns inside their ``fit``:

  1. **Expand**: a per-row design ``(N, F)`` becomes a per-(row, bracket)
     design ``(M, F + 2)`` with ``[..., lo_b, hi_b]`` appended.
  2. **Fit**: an sklearn estimator on that expanded design and a target
     derived from ``y`` — *always* the bracket-hit indicator
     ``1[y ∈ bracket_b]``.

That hardcoded target made the regressor inflexible: any caller who
wanted to learn a *different* per-(row, bracket) target (e.g. the
mispricing residual ``hit − market_p``) had to fork the class. From
v0.5.0 the two concerns are separated:

- ``BracketExpander`` (this module): owns the per-row ↔ per-(row, bracket)
  conversion. Builds ``X_expanded`` and, by default, a bracket-hit
  target ``y_expanded`` — but the caller can ignore that and supply any
  target of shape ``(M,)`` they like.

- ``BracketClassifier`` / ``BracketRegressor`` (in
  ``bracketlearn.trainers.bracket``): become *plain sklearn* — their
  ``fit(X, y)`` takes whatever ``(X, y)`` the caller hands them, with
  no internal target derivation. ``predict`` returns raw scores. Assembling
  those scores back into a per-row ``BracketForecast`` is a separate
  method on the expander (``assemble_dist``), so callers can reach into
  the middle of the pipeline if they want.

API shape
---------
::

    expander = BracketExpander(brackets_by_id={...})

    # Train side: expand X and (optionally) derive a default target.
    X_expanded, y_expanded = expander.fit_transform(X, y, ids=train_ids)
    reg = BracketRegressor(estimator=LGBMRegressor()).fit(X_expanded, y_expanded)

    # Predict side: expand X only, predict, then assemble back to a dist.
    X_pred_expanded, _ = expander.transform(X_pred, ids=pred_ids)
    raw_preds = reg.predict(X_pred_expanded)
    dist = expander.assemble_dist(raw_preds, ids=pred_ids, timestamps=...)

Callers who want a custom target build it themselves on top of the
expansion::

    X_expanded, y_hit = expander.fit_transform(X, y, ids=train_ids)
    market_p_expanded = ...  # caller-supplied, shape (M,)
    X_expanded = np.column_stack([X_expanded, market_p_expanded])
    y_target = y_hit - market_p_expanded
    BracketRegressor(estimator=LGBMRegressor()).fit(X_expanded, y_target)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bracketlearn.forecast import DistributionForecast, ProvenanceMeta
from bracketlearn.trainers._common import (
    _augment_with_bracket_bounds,
    _validate_brackets_by_id,
)


@dataclass
class BracketExpander:
    """Per-row ↔ per-(row, bracket) transformer.

    Stores the per-id edge ladders and exposes:

    - ``fit_transform(X, y, *, ids)`` → ``(X_expanded, y_expanded)``.
      ``X_expanded`` is ``(M, F + 2)``: each original row ``i`` becomes
      ``B_i`` rows of ``[X_i..., lo_b, hi_b]``, where ``M = Σ B_i``.
      ``y_expanded`` is the default bracket-hit target ``1[y_i ∈ bracket_b]``,
      shape ``(M,)``; pass ``y=None`` to get only ``X_expanded`` back.

    - ``transform(X, *, ids)`` → ``(X_expanded, None)``. Predict-side
      counterpart; never returns a target.

    - ``assemble_dist(predictions, *, ids, timestamps, name=..., clip_eps=...)``
      → ``DistributionForecast``. Inverse of ``transform`` on the
      prediction side: per-row clip + renormalise + pack into a
      ``BracketForecast`` whose ``edges`` and ``probs`` match the ids'
      ladders.

    State is captured at construction: ``brackets_by_id`` is the
    authoritative dict. ``fit_transform`` does *not* mutate it — callers
    that need to add per-row ladders at predict time should construct
    a new expander or mutate the dict directly before calling
    ``transform`` / ``assemble_dist``.

    Output X column layout
    ----------------------
    ``[X_0, X_1, ..., X_{F-1}, lo, hi]`` — original features first, then
    the two bracket-bound columns. Callers extending the feature set
    (e.g. with ``market_p``) should append AFTER ``hi``::

        X_expanded, _ = expander.fit_transform(X, ids=ids)
        extras = build_extras(...)             # shape (M, E)
        X_full = np.column_stack([X_expanded, extras])
    """

    brackets_by_id: dict[Any, np.ndarray]
    name: str = "BracketExpander"
    clip_eps: float = 1e-6

    # Populated by fit_transform / transform; exposed for callers that
    # build their own targets without re-running the row walk.
    offsets_: np.ndarray | None = field(default=None, init=False)
    per_row_edges_: list[np.ndarray] | None = field(default=None, init=False)
    last_ids_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _validate_brackets_by_id(self.brackets_by_id, owner="BracketExpander")
        if not (0.0 < self.clip_eps < 0.5):
            raise ValueError(
                f"clip_eps must be in (0, 0.5); got {self.clip_eps}"
            )

    # ----- core transform ---------------------------------------------------

    def transform(
        self, X: np.ndarray, *, ids: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        """Expand X without computing any target. Predict-time path."""
        X_expanded, offsets, per_row_edges = _augment_with_bracket_bounds(
            np.asarray(X, dtype=float),
            np.asarray(ids),
            self.brackets_by_id,
            owner=self.name,
        )
        self.offsets_ = offsets
        self.per_row_edges_ = per_row_edges
        self.last_ids_ = np.asarray(ids)
        return X_expanded, None

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Expand X and (optionally) compute the default bracket-hit target.

        ``y`` is shape ``(N,)`` — per-row realized values. The returned
        ``y_expanded`` is shape ``(M,)`` with ``1`` where ``y_i`` falls
        inside bracket ``b`` of row ``i``, else ``0``. Bin-membership
        uses right-open intervals matching ``np.searchsorted(..., side='right')``,
        with the last bracket closed on the right (the standard
        bracketlearn convention).

        Pass ``y=None`` to skip target construction.
        """
        X_expanded, _ = self.transform(X, ids=ids)
        if y is None:
            return X_expanded, None
        y_per_row = np.asarray(y, dtype=float)
        if y_per_row.shape[0] != np.asarray(ids).shape[0]:
            raise ValueError(
                f"{self.name}.fit_transform: y has length {y_per_row.shape[0]} "
                f"but ids has length {np.asarray(ids).shape[0]}"
            )
        assert self.offsets_ is not None and self.per_row_edges_ is not None
        N = y_per_row.shape[0]
        M = int(self.offsets_[-1])
        y_expanded = np.zeros(M, dtype=float)
        for i in range(N):
            e_i = self.per_row_edges_[i]
            k = int(np.searchsorted(e_i, y_per_row[i], side="right") - 1)
            if 0 <= k < e_i.size - 1:
                y_expanded[int(self.offsets_[i]) + k] = 1.0
        return X_expanded, y_expanded

    # ----- inverse: assemble per-row dist from per-(row, bracket) preds ----

    def assemble_dist(
        self,
        predictions: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        name: str | None = None,
        clip_eps: float | None = None,
    ) -> DistributionForecast:
        """Pack ``(M,)`` per-(row, bracket) predictions into a per-row dist.

        Each row's slice of ``predictions`` is clipped to
        ``[clip_eps, 1 - clip_eps]`` and renormalised to sum to 1, then
        attached to the row's edge ladder. Returns a
        ``BracketForecast``-backed ``DistributionForecast``.

        ``ids`` / ``timestamps`` must match the row order of the
        most-recent ``transform`` / ``fit_transform`` call — same shape,
        same ordering. ``predictions`` must have length
        ``offsets_[-1]``.
        """
        if self.offsets_ is None or self.per_row_edges_ is None:
            raise RuntimeError(
                f"{self.name}.assemble_dist called before transform / "
                f"fit_transform — no offsets recorded"
            )
        eps = self.clip_eps if clip_eps is None else clip_eps
        if not (0.0 < eps < 0.5):
            raise ValueError(f"clip_eps must be in (0, 0.5); got {eps}")
        preds = np.asarray(predictions, dtype=float)
        expected = int(self.offsets_[-1])
        if preds.shape[0] != expected:
            raise ValueError(
                f"{self.name}.assemble_dist: predictions length "
                f"{preds.shape[0]} != offsets_[-1]={expected}; "
                f"transform and predict are out of sync"
            )
        ids_arr = np.asarray(ids)
        ts_arr = np.asarray(timestamps)
        N = ids_arr.shape[0]
        if N != len(self.per_row_edges_):
            raise ValueError(
                f"{self.name}.assemble_dist: ids length {N} != "
                f"per_row_edges_ length {len(self.per_row_edges_)}; "
                f"ids do not match the last transform call"
            )

        p_aug = np.clip(preds, eps, 1.0 - eps)
        B_per_row = np.array(
            [e.size - 1 for e in self.per_row_edges_], dtype=int,
        )
        B_max = int(B_per_row.max())
        edges_out = np.full((N, B_max + 1), np.nan, dtype=float)
        probs_out = np.full((N, B_max), np.nan, dtype=float)
        for i in range(N):
            sl = slice(int(self.offsets_[i]), int(self.offsets_[i + 1]))
            p_row = p_aug[sl]
            s = float(p_row.sum())
            if s <= 0:
                raise RuntimeError(
                    f"{self.name}.assemble_dist row {i}: clipped scores "
                    f"summed to {s}; should be unreachable given clip_eps > 0"
                )
            p_row = p_row / s
            B_i = int(B_per_row[i])
            probs_out[i, :B_i] = p_row
            edges_out[i, : B_i + 1] = self.per_row_edges_[i]
        prov = ProvenanceMeta.placeholder(
            name or self.name, sigma_source="native",
        )
        return DistributionForecast.from_brackets(
            edges=edges_out, probs=probs_out,
            ids=ids_arr, timestamps=ts_arr,
            provenance=prov,
        )
