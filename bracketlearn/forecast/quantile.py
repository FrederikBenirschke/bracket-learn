"""QuantileForecast — distribution stored as per-row quantile pairs (τ, q_τ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bracketlearn.forecast._helpers import _resolve_tail_kinds
from bracketlearn.forecast._meta import ProvenanceMeta, TailPolicy
from bracketlearn.forecast.base import DistributionForecast


@dataclass(frozen=True)
class QuantileForecast(DistributionForecast):
    taus: np.ndarray             # (Q,)
    qvals: np.ndarray            # (N, Q)
    tail_policy: TailPolicy

    @classmethod
    def from_arrays(
        cls,
        *,
        taus: np.ndarray,
        qvals: np.ndarray,
        tail_policy: TailPolicy,
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> QuantileForecast:
        taus = np.asarray(taus, dtype=float)
        qvals = np.asarray(qvals, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if taus.ndim != 1 or taus.shape[0] < 2:
            raise ValueError(f"taus must be 1-D with ≥2 entries; got {taus.shape}")
        if qvals.ndim != 2 or qvals.shape[1] != taus.shape[0]:
            raise ValueError(
                f"qvals shape {qvals.shape} incompatible with taus {taus.shape}"
            )
        if qvals.shape[0] != ids.shape[0]:
            raise ValueError(f"N mismatch: qvals N={qvals.shape[0]} ids N={ids.shape[0]}")
        if np.any(np.diff(taus) <= 0):
            raise ValueError("taus must be strictly increasing")
        if np.any((taus <= 0) | (taus >= 1)):
            raise ValueError("taus must lie strictly in (0, 1)")
        diffs = np.diff(qvals, axis=1)
        if np.any(diffs < 0):
            row_range = qvals.max(axis=1, keepdims=True) - qvals.min(axis=1, keepdims=True)
            tol = np.maximum(1e-9 * row_range, 1e-12)
            worst = float(diffs.min())
            if np.any(diffs < -tol):
                raise ValueError(
                    f"qvals must be monotone non-decreasing along Q; "
                    f"worst crossing = {worst:.6g} (use isotonic-repair upstream)"
                )
            qvals = np.maximum.accumulate(qvals, axis=1)
        return cls(
            ids=ids, timestamps=timestamps, provenance=provenance,
            taus=taus, qvals=qvals, tail_policy=tail_policy,
        )

    def affine(self, shift, scale) -> QuantileForecast:
        c, s = self._affine_csc(shift, scale)
        return QuantileForecast.from_arrays(
            taus=self.taus, qvals=self.qvals * s[:, None] + c[:, None],
            tail_policy=self.tail_policy,
            ids=self.ids, timestamps=self.timestamps, provenance=self.provenance,
        )

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"taus": self.taus, "qvals": self.qvals}

    @property
    def tail_support(self) -> str:
        return "finite-quantile"

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        taus = self.taus
        qvals = self.qvals
        N, _ = qvals.shape
        out = np.zeros((N, x_arr.shape[0]))
        left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
        for i in range(N):
            interp = np.interp(
                x_arr, qvals[i], taus,
                left=0.0 if left_rule == "clip" else np.nan,
                right=1.0 if right_rule == "clip" else np.nan,
            )
            if np.any(np.isnan(interp)):
                raise NotImplementedError(
                    f"tail policy {left_rule!r}/{right_rule!r} not implemented yet"
                )
            out[i] = interp
        return out[:, 0] if scalar else out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        N_rows = self.ids.shape[0]
        if y_arr.shape[0] != N_rows:
            raise ValueError(
                f"cdf_at: y has {y_arr.shape[0]} rows, dist has {N_rows}"
            )
        taus = self.taus
        qvals = self.qvals
        left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
        out = np.empty(N_rows, dtype=float)
        for i in range(N_rows):
            v = np.interp(
                y_arr[i:i + 1], qvals[i], taus,
                left=0.0 if left_rule == "clip" else np.nan,
                right=1.0 if right_rule == "clip" else np.nan,
            )
            if np.isnan(v).any():
                raise NotImplementedError(
                    f"tail policy {left_rule!r}/{right_rule!r} not implemented yet"
                )
            out[i] = v[0]
        return out

    def cdf_at_grid(self, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError(f"cdf_at_grid: y must be 2-D (N, M); got shape {y_arr.shape}")
        N_rows = self.ids.shape[0]
        if y_arr.shape[0] != N_rows:
            raise ValueError(
                f"cdf_at_grid: y has {y_arr.shape[0]} rows, dist has {N_rows}"
            )
        M = y_arr.shape[1]
        nan_mask = np.isnan(y_arr)
        y_safe = np.where(nan_mask, 0.0, y_arr)
        taus = self.taus
        qvals = self.qvals
        left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
        out = np.empty((N_rows, M), dtype=float)
        for i in range(N_rows):
            v = np.interp(
                y_safe[i], qvals[i], taus,
                left=0.0 if left_rule == "clip" else np.nan,
                right=1.0 if right_rule == "clip" else np.nan,
            )
            if np.isnan(v).any():
                raise NotImplementedError(
                    f"tail policy {left_rule!r}/{right_rule!r} not implemented yet"
                )
            out[i] = v
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        taus = self.taus
        qvals = self.qvals
        N, _ = qvals.shape
        out = np.empty((N, tau_arr.shape[0]))
        left_rule, right_rule = _resolve_tail_kinds(self.tail_policy)
        if left_rule != "clip" or right_rule != "clip":
            raise NotImplementedError(
                f"ppf on quantile backing only supports clip tails; "
                f"got left={left_rule!r} right={right_rule!r}"
            )
        for i in range(N):
            out[i] = np.interp(tau_arr, taus, qvals[i])
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        raise NotImplementedError(
            "pdf on quantile backing not implemented in v0.2; use bracket integration."
        )

    def mean(self):
        taus = self.taus
        qvals = self.qvals
        d = np.empty_like(taus)
        d[0] = (taus[1] - 0.0) * 0.5
        d[-1] = (1.0 - taus[-2]) * 0.5
        d[1:-1] = (taus[2:] - taus[:-2]) * 0.5
        return (qvals * d[None, :]).sum(axis=1)

    def variance(self):
        raise NotImplementedError("variance on quantile backing not implemented in v0.2")

    def crps(self, y):
        from bracketlearn.score import crps_quantile
        return crps_quantile(self, y)

    def log_score(self, y):
        from bracketlearn.score import log_score_quantile
        return log_score_quantile(self, y)

    def to_point(self, *, how: str = "mean"):
        if how not in ("mean", "median", "mode"):
            raise ValueError(f"how={how!r} not in 'mean'/'median'/'mode'")
        taus = self.taus
        qvals = self.qvals
        if how == "median":
            j = int(np.argmin(np.abs(taus - 0.5)))
            return qvals[:, j]
        if how == "mean":
            dt = np.diff(taus)
            avg = 0.5 * (qvals[:, :-1] + qvals[:, 1:])
            inner = (avg * dt[None, :]).sum(axis=1)
            lower = qvals[:, 0] * taus[0]
            upper = qvals[:, -1] * (1.0 - taus[-1])
            return lower + inner + upper
        # mode: highest-density bin → midpoint of (q_i, q_{i+1})
        dq = np.diff(qvals, axis=1)
        dt = np.diff(taus)
        density = dt[None, :] / np.where(dq > 1e-12, dq, 1e-12)
        top = np.argmax(density, axis=1)
        rows = np.arange(qvals.shape[0])
        return 0.5 * (qvals[rows, top] + qvals[rows, top + 1])

    @classmethod
    def stitch(cls, folds, *, timestamps, provenance):
        all_rows = np.concatenate([rows for rows, _ in folds])
        order = np.argsort(all_rows, kind="stable")
        ids_sorted = all_rows[order]
        ts_sorted = timestamps[all_rows][order]
        taus_set = {tuple(d.taus.tolist()) for _, d in folds}
        if len(taus_set) > 1:
            raise ValueError(
                "quantile folds use different tau vectors; all folds must "
                "share the same quantile grid."
            )
        taus = folds[0][1].taus
        qvals = np.concatenate([d.qvals for _, d in folds], axis=0)[order]
        tail_policy = folds[0][1].tail_policy
        return cls.from_arrays(
            taus=taus, qvals=qvals, tail_policy=tail_policy,
            ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
        )
