"""BracketForecast — per-row bracket-backed distribution.

Storage is always 2-D with NaN padding for ragged rows; see the class
docstring for the exact (edges, probs) shape contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bracketlearn.forecast._helpers import _quantile_via_brentq
from bracketlearn.forecast._meta import Backing, ProvenanceMeta
from bracketlearn.forecast.base import DistributionForecast


@dataclass(frozen=True)
class BracketForecast(DistributionForecast):
    """Per-row bracket-backed distribution.

    Storage is always 2-D:
      ``edges``: (N, B_max + 1) — row i's bracket boundaries live in
        the first ``B_i + 1`` columns; trailing columns are NaN.
      ``probs``: (N, B_max)     — row i's bracket probabilities live
        in the first ``B_i`` columns; trailing columns are NaN.

    All math is per-row. Ragged-row support is via NaN padding: a row's
    valid prefix is everything before the first NaN in ``edges`` (and
    the matching one-shorter prefix in ``probs``). All accessor methods
    mask NaN-padded positions out and return finite results for valid
    rows.

    The ``shared_edges`` helper returns the 1-D edge vector iff every
    row has identical edges (and no NaN padding); otherwise it raises.
    Use it from legacy callers that still assume a shared ladder.
    """

    edges: np.ndarray            # (N, B+1) with NaN padding for ragged rows
    probs: np.ndarray            # (N, B) with NaN padding for ragged rows

    @classmethod
    def from_arrays(
        cls,
        *,
        edges: np.ndarray,           # 1-D (B+1,) broadcast to all rows, OR 2-D (N, B+1)
        probs: np.ndarray,           # (N, B)
        ids: np.ndarray,
        timestamps: np.ndarray,
        provenance: ProvenanceMeta,
    ) -> BracketForecast:
        edges_in = np.asarray(edges, dtype=float)
        probs = np.asarray(probs, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if probs.ndim != 2:
            raise ValueError(f"probs must be 2-D (N, B); got shape {probs.shape}")
        if probs.shape[0] != ids.shape[0]:
            raise ValueError(f"N mismatch: probs N={probs.shape[0]} ids N={ids.shape[0]}")
        N = probs.shape[0]
        if edges_in.ndim == 1:
            if edges_in.shape[0] != probs.shape[1] + 1:
                raise ValueError(
                    f"probs shape {probs.shape} incompatible with shared edges {edges_in.shape}"
                )
            if np.any(np.diff(edges_in) <= 0):
                raise ValueError("edges must be monotone strictly increasing")
            edges = np.broadcast_to(edges_in[None, :], (N, edges_in.shape[0])).copy()
        elif edges_in.ndim == 2:
            if edges_in.shape[0] != N:
                raise ValueError(f"edges N={edges_in.shape[0]} != probs N={N}")
            if edges_in.shape[1] != probs.shape[1] + 1:
                raise ValueError(
                    f"edges shape {edges_in.shape} incompatible with probs {probs.shape}"
                )
            # Per-row monotonicity check, NaN-tolerant: for each row, the
            # finite prefix must be strictly increasing.
            edge_nan = np.isnan(edges_in)
            for i in range(N):
                row = edges_in[i]
                valid_len = int((~edge_nan[i]).sum())
                if valid_len < 2:
                    raise ValueError(
                        f"row {i}: edges must have ≥2 finite entries; got {valid_len}"
                    )
                prefix = row[:valid_len]
                if np.any(np.diff(prefix) <= 0):
                    raise ValueError(
                        f"row {i}: finite edge prefix must be monotone strictly increasing"
                    )
            edges = edges_in
        else:
            raise ValueError(f"edges must be 1-D or 2-D; got shape {edges_in.shape}")
        # Validate probs row-wise. NaN positions in probs must align with
        # the NaN-padded edge tail (row's B_i = valid_edges - 1).
        prob_nan = np.isnan(probs)
        edge_nan = np.isnan(edges)
        for i in range(N):
            valid_e = int((~edge_nan[i]).sum())
            valid_p = int((~prob_nan[i]).sum())
            if valid_p != valid_e - 1:
                raise ValueError(
                    f"row {i}: {valid_p} finite probs but {valid_e} finite edges — "
                    f"expected probs = edges - 1"
                )
            p_row = probs[i, :valid_p]
            if np.any(p_row < 0):
                raise ValueError(f"row {i}: probs must be nonnegative")
            if not np.isclose(p_row.sum(), 1.0, atol=1e-6):
                raise ValueError(
                    f"row {i}: probs must sum to 1; got {float(p_row.sum()):.6g}"
                )
        return cls(
            ids=ids, timestamps=timestamps, provenance=provenance,
            edges=edges, probs=probs,
        )

    @property
    def backing(self) -> Backing:
        return Backing.BRACKET

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"edges": self.edges, "probs": self.probs}

    @property
    def tail_policy(self):
        return None

    @property
    def tail_support(self) -> str:
        return "bounded"

    def shared_edges(self) -> np.ndarray:
        """Return the 1-D edge vector if every row shares the same edges.

        Raises if rows have ragged lengths (any NaN padding) or numerically
        distinct edge vectors. Use from legacy callers that still assume a
        shared ladder; new code should consume ``self.edges`` (2-D) directly.
        """
        edges = self.edges
        if np.isnan(edges).any():
            raise ValueError(
                "shared_edges: at least one row has NaN-padded edges — rows "
                "are ragged. Update caller to consume per-row edges."
            )
        first = edges[0]
        if not np.allclose(edges, first[None, :]):
            raise ValueError(
                "shared_edges: rows have distinct edge vectors. Update caller "
                "to consume per-row edges."
            )
        return first.copy()

    # ---------- internal helpers ----------

    def _row_valid_B(self) -> np.ndarray:
        """Per-row number of valid bins B_i. (N,) ints."""
        # B_i = count(non-NaN probs in row i) = count(non-NaN edges in row i) - 1.
        return (~np.isnan(self.probs)).sum(axis=1).astype(int)

    def _row_cum(self) -> np.ndarray:
        """(N, B_max + 1) cumulative probs. NaN-tolerant via nan-to-zero
        on probs; the cumulative tail beyond a row's valid prefix sits at
        the row's full mass (= 1) which is the right value for any query
        ≥ that row's right edge."""
        N, B_max = self.probs.shape
        p_clean = np.nan_to_num(self.probs, nan=0.0)
        cum = np.concatenate([np.zeros((N, 1)), np.cumsum(p_clean, axis=1)], axis=1)
        return cum

    def _per_row_search(self, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For a (N, M) query array, return (k_clipped, below, above) per
        (row, col). Uses each row's own edges (NaN-aware)."""
        N, M = vals.shape
        edges = self.edges
        B_per_row = self._row_valid_B()                # (N,)
        # left/right valid edge per row.
        left_edge = edges[:, 0]                         # (N,)
        right_edge = edges[np.arange(N), B_per_row]     # (N,) — edges[i, B_i]
        below = vals <= left_edge[:, None]
        above = vals >= right_edge[:, None]
        k = np.empty((N, M), dtype=int)
        # Per-row searchsorted on each row's finite prefix.
        for i in range(N):
            B_i = int(B_per_row[i])
            e_i = edges[i, :B_i + 1]
            ki = np.searchsorted(e_i, vals[i], side="right") - 1
            k[i] = np.clip(ki, 0, B_i - 1)
        return k, below, above

    # ---------- math ----------

    def cdf(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        N = self.ids.shape[0]
        M = x_arr.shape[0]
        # Broadcast x across all rows so each row evaluates the same M points.
        vals = np.broadcast_to(x_arr[None, :], (N, M)).copy()
        return self._cdf_2d(vals)[:, 0] if scalar else self._cdf_2d(vals)

    def _cdf_2d(self, vals: np.ndarray) -> np.ndarray:
        """Per-row CDF at vals[i, :]. Shape in/out: (N, M)."""
        N, M = vals.shape
        edges = self.edges
        probs_clean = np.nan_to_num(self.probs, nan=0.0)
        cum = self._row_cum()                           # (N, B_max+1)
        k, below, above = self._per_row_search(vals)    # (N, M) each
        # Gather per-row edge[k] and edge[k+1] and prob[k].
        lo = np.take_along_axis(edges, k, axis=1)       # (N, M)
        hi = np.take_along_axis(edges, k + 1, axis=1)   # (N, M)
        p_k = np.take_along_axis(probs_clean, k, axis=1)
        c_k = np.take_along_axis(cum, k, axis=1)
        widths = hi - lo
        safe_w = np.where(widths > 0, widths, 1.0)
        frac = np.where(widths > 0, (vals - lo) / safe_w, 0.0)
        out = c_k + frac * p_k
        out = np.where(below, 0.0, out)
        out = np.where(above, 1.0, out)
        return out

    def cdf_at(self, y):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        N = self.ids.shape[0]
        if y_arr.shape[0] != N:
            raise ValueError(f"cdf_at: y has {y_arr.shape[0]} rows, dist has {N}")
        vals = y_arr.reshape(N, 1)
        return self._cdf_2d(vals)[:, 0]

    def cdf_at_grid(self, y):
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError(f"cdf_at_grid: y must be 2-D (N, M); got shape {y_arr.shape}")
        N = self.ids.shape[0]
        if y_arr.shape[0] != N:
            raise ValueError(f"cdf_at_grid: y has {y_arr.shape[0]} rows, dist has {N}")
        nan_mask = np.isnan(y_arr)
        y_safe = np.where(nan_mask, 0.0, y_arr)
        out = self._cdf_2d(y_safe)
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        return out

    def ppf(self, tau):
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        scalar = np.isscalar(tau)
        if np.any((tau_arr < 0) | (tau_arr > 1)):
            raise ValueError("tau must be in [0, 1]")
        N = self.ids.shape[0]
        edges = self.edges
        cum = self._row_cum()
        B_per_row = self._row_valid_B()
        out = np.empty((N, tau_arr.shape[0]))
        for i in range(N):
            B_i = int(B_per_row[i])
            e_i = edges[i, :B_i + 1]
            cum_i = cum[i, :B_i + 1]
            for j, t in enumerate(tau_arr):
                if t <= 0:
                    out[i, j] = e_i[0]
                elif t >= 1:
                    out[i, j] = e_i[-1]
                else:
                    k = int(np.searchsorted(cum_i, t, side="right") - 1)
                    k = max(0, min(k, B_i - 1))
                    width_p = cum_i[k + 1] - cum_i[k]
                    if width_p <= 0:
                        out[i, j] = e_i[k]
                    else:
                        frac = (t - cum_i[k]) / width_p
                        out[i, j] = e_i[k] + frac * (e_i[k + 1] - e_i[k])
        return out[:, 0] if scalar else out

    def pdf(self, x, *, density_method=None):
        if density_method != "step":
            raise ValueError(
                "pdf on bracket backing requires density_method='step' "
                "(no silent KDE bandwidth)"
            )
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        scalar = np.isscalar(x)
        N = self.ids.shape[0]
        M = x_arr.shape[0]
        vals = np.broadcast_to(x_arr[None, :], (N, M)).copy()
        edges = self.edges
        probs_clean = np.nan_to_num(self.probs, nan=0.0)
        k, below, above = self._per_row_search(vals)
        lo = np.take_along_axis(edges, k, axis=1)
        hi = np.take_along_axis(edges, k + 1, axis=1)
        p_k = np.take_along_axis(probs_clean, k, axis=1)
        widths = hi - lo
        safe_w = np.where(widths > 0, widths, 1.0)
        density = np.where(widths > 0, p_k / safe_w, 0.0)
        # Outside the row's support → 0 (matches v0.2 behaviour).
        inside = ~(below | above)
        out = np.where(inside, density, 0.0)
        return out[:, 0] if scalar else out

    def mean(self):
        # Per-row midpoints weighted by probs; NaN-tolerant via nansum.
        mids = 0.5 * (self.edges[:, 1:] + self.edges[:, :-1])      # (N, B_max)
        return np.nansum(self.probs * mids, axis=1)

    def variance(self):
        mids = 0.5 * (self.edges[:, 1:] + self.edges[:, :-1])
        m = np.nansum(self.probs * mids, axis=1)
        ex2 = np.nansum(self.probs * (mids ** 2), axis=1)
        return ex2 - m ** 2

    def realized_bin(self, y: np.ndarray) -> np.ndarray:
        """Per-row index of the bracket containing realized y. (N,) ints.

        y below row's leftmost edge → 0; y at/above rightmost edge → B_i - 1.
        """
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        N = self.ids.shape[0]
        if y_arr.shape[0] != N:
            raise ValueError(f"realized_bin: y has {y_arr.shape[0]} rows, dist has {N}")
        edges = self.edges
        B_per_row = self._row_valid_B()
        out = np.empty(N, dtype=int)
        for i in range(N):
            B_i = int(B_per_row[i])
            e_i = edges[i, :B_i + 1]
            k = int(np.searchsorted(e_i, y_arr[i], side="right") - 1)
            out[i] = max(0, min(k, B_i - 1))
        return out

    def crps(self, y):
        from bracketlearn.score import crps_bracket
        return crps_bracket(self, y)

    def log_score(self, y):
        from bracketlearn.score import log_score_bracket
        return log_score_bracket(self, y)

    def to_point(self, *, how: str = "mean"):
        if how not in ("mean", "median", "mode"):
            raise ValueError(f"how={how!r} not in 'mean'/'median'/'mode'")
        mids = 0.5 * (self.edges[:, :-1] + self.edges[:, 1:])
        if how == "mean":
            return np.nansum(self.probs * mids, axis=1)
        if how == "mode":
            p_clean = np.nan_to_num(self.probs, nan=-np.inf)
            top = np.argmax(p_clean, axis=1)
            rows = np.arange(self.probs.shape[0])
            return mids[rows, top]
        return _quantile_via_brentq(self, 0.5)

    @classmethod
    def stitch(cls, folds, *, timestamps, provenance):
        all_rows = np.concatenate([rows for rows, _ in folds])
        order = np.argsort(all_rows, kind="stable")
        ids_sorted = all_rows[order]
        ts_sorted = timestamps[all_rows][order]
        edges = np.concatenate([d.edges for _, d in folds], axis=0)[order]
        probs = np.concatenate([d.probs for _, d in folds], axis=0)[order]
        return cls.from_arrays(
            edges=edges, probs=probs,
            ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
        )
