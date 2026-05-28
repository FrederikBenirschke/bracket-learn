"""Bracket-backed DistForecasters and dist-pool combiners.

CumulativeBinary, TailSpecialist, CDFBoostBracket, BracketClassifier
(bracket-emitting trainers). LinearPoolDist (convex combination of upstream
dists).
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


@dataclass
class CumulativeBinary(BaseEstimator):
    """Single LightGBM binary classifier on augmented features.

    Fits one classifier over (X, cutpoint) → 1[y ≤ cutpoint], then at
    predict time queries P(y ≤ k) for each cutpoint k in the row's own
    grid and emits a per-row bracket-backed dist.

    v0.3 — per-row brackets
    -----------------------
    Each market/event has its own cutpoint grid (the interior bracket
    edges) and its own outer-edge pair (left/right boundaries
    absorbing tail mass). Both are passed as id-keyed dicts at
    construction:

      ``cutpoints_by_id``:    dict mapping row id → 1-D float array of
                              cutpoints (length K_i ≥ 1).
      ``outer_edges_by_id``:  dict mapping row id → (lo_i, hi_i) tuple
                              with ``lo_i < cutpoints[0]`` and
                              ``hi_i > cutpoints[-1]``.

    Both dicts must cover every id passed to fit/predict. Each row i
    contributes K_i augmented training examples to the LGBM model — the
    examples for different rows may have different cutpoint counts and
    spacings. Cutpoint values are passed as a feature, so the model
    naturally generalises across grids.

    The model itself is global (one LGBM); only the per-row augmentation
    differs. Predict-time emits a BracketForecast on each row's full
    ladder ``[lo_i, cutpoints_i..., hi_i]``.
    """

    cutpoints_by_id: dict[Any, np.ndarray]
    outer_edges_by_id: dict[Any, tuple[float, float]]
    n_estimators: int = 80
    learning_rate: float = 0.05
    num_leaves: int = 7
    min_child_samples: int = 100
    monotone: bool = True
    name: str = "CumulativeBinary"
    depends_on: tuple[str, ...] = ()
    model_: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cutpoints_by_id, dict) or not self.cutpoints_by_id:
            raise ValueError(
                "CumulativeBinary needs a non-empty cutpoints_by_id dict "
                "(id → 1-D cutpoint array)"
            )
        if not isinstance(self.outer_edges_by_id, dict) or not self.outer_edges_by_id:
            raise ValueError(
                "CumulativeBinary needs a non-empty outer_edges_by_id dict "
                "(id → (lo, hi) tuple)"
            )
        # Validate each entry.
        for k, cuts in self.cutpoints_by_id.items():
            cuts_arr = np.asarray(cuts, dtype=float)
            if cuts_arr.ndim != 1 or cuts_arr.size == 0:
                raise ValueError(
                    f"cutpoints_by_id[{k!r}] must be 1-D non-empty; got shape {cuts_arr.shape}"
                )
            if np.any(np.diff(cuts_arr) <= 0):
                raise ValueError(
                    f"cutpoints_by_id[{k!r}] must be strictly increasing"
                )
            if k not in self.outer_edges_by_id:
                raise ValueError(
                    f"cutpoints_by_id has id {k!r} but outer_edges_by_id does not"
                )
            lo, hi = self.outer_edges_by_id[k]
            if not (lo < cuts_arr[0]):
                raise ValueError(
                    f"outer_edges_by_id[{k!r}][0]={lo} must be < cutpoints[0]={cuts_arr[0]}"
                )
            if not (hi > cuts_arr[-1]):
                raise ValueError(
                    f"outer_edges_by_id[{k!r}][1]={hi} must be > cutpoints[-1]={cuts_arr[-1]}"
                )

    def _augment(
        self,
        X: np.ndarray,
        ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the per-row augmented design matrix. Returns
        ``(X_aug, row_blocks)`` where ``row_blocks[i]`` is the
        ``slice`` covering row i's augmented examples."""
        N = X.shape[0]
        per_row_cuts = []
        missing = []
        for k in ids:
            try:
                per_row_cuts.append(np.asarray(self.cutpoints_by_id[k], dtype=float))
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
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        if X.shape[0] != y.shape[0] or X.shape[0] != ids.shape[0]:
            raise ValueError(
                f"shape mismatch: X N={X.shape[0]} y N={y.shape[0]} ids N={ids.shape[0]}"
            )
        X_aug, offsets, per_row_cuts = self._augment(X, ids)
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
    ) -> DistributionForecast:
        if self.model_ is None:
            raise RuntimeError("CumulativeBinary.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        N = X.shape[0]
        X_aug, offsets, per_row_cuts = self._augment(X, ids)
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
            lo, hi = self.outer_edges_by_id[ids[i]]
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


# ---------------------------------------------------------------------------
# TailSpecialist — EMOS body + LightGBM tail classifiers (DistForecaster).
# ---------------------------------------------------------------------------


@dataclass
class TailSpecialist(BaseEstimator):
    """Gaussian body (from upstream EMOS μ̂/σ̂) + LightGBM tail classifiers,
    on per-row brackets.

    depends_on a parametric-normal upstream (typically named ``emos``)
    and a per-row bracket ladder via ``brackets_by_id`` (id → 1-D edge
    array). Fits two global binary classifiers — one for "y in row's
    first bracket" and one for "y in row's last bracket" — and at
    predict time replaces each row's first/last bin mass with the
    classifier outputs, rescaling the middle bins to (1 - p_lo - p_hi).

    v0.3 — per-row brackets
    -----------------------
    Each row's first/last bracket can have *different* boundaries
    (Kalshi-style daily-rotating ladders). The training-time tail
    indicators are therefore "y in row's first bracket" /
    "y in row's last bracket" — per-row searchsorted, not a fixed
    threshold.

    Two global classifiers are still appropriate because the row's
    bracket geometry varies but the upstream-feature relationship to
    "tail event" doesn't. (Per-bracket classifier ensembles would
    require ≥1 trainer per market — not what this trainer is for.)
    """

    brackets_by_id: dict[Any, np.ndarray]
    upstream: str = "emos"
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    name: str = "TailSpecialist"
    clf_lo_: Any = field(default=None, init=False)
    clf_hi_: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.depends_on = (self.upstream,)
        if not isinstance(self.brackets_by_id, dict) or not self.brackets_by_id:
            raise ValueError(
                "TailSpecialist needs a non-empty brackets_by_id dict "
                "(id → 1-D edge array)"
            )
        for k, e in self.brackets_by_id.items():
            e_arr = np.asarray(e, dtype=float)
            if e_arr.ndim != 1 or e_arr.size < 4:
                raise ValueError(
                    f"brackets_by_id[{k!r}]: ladder must have ≥3 brackets "
                    f"(≥4 edges); got shape {e_arr.shape}"
                )
            if np.any(np.diff(e_arr) <= 0):
                raise ValueError(
                    f"brackets_by_id[{k!r}]: edges must be strictly increasing"
                )

    def _row_edges(self, ids: np.ndarray) -> list[np.ndarray]:
        try:
            return [np.asarray(self.brackets_by_id[k], dtype=float) for k in ids]
        except KeyError as e:
            raise KeyError(f"TailSpecialist: brackets_by_id missing id {e!r}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        if not deps_oof or self.upstream not in deps_oof:
            raise ValueError(
                f"TailSpecialist.fit needs deps_oof[{self.upstream!r}]"
            )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        if X.shape[0] != y.shape[0] or X.shape[0] != ids.shape[0]:
            raise ValueError(
                f"shape mismatch: X N={X.shape[0]} y N={y.shape[0]} ids N={ids.shape[0]}"
            )
        per_row_edges = self._row_edges(ids)
        # Per-row tail indicators: "y in row's first bin" / "y in row's last bin".
        y_lo = np.array([float(y[i] < per_row_edges[i][1]) for i in range(y.size)], dtype=int)
        y_hi = np.array([float(y[i] >= per_row_edges[i][-2]) for i in range(y.size)], dtype=int)
        if y_lo.sum() < 5 or y_hi.sum() < 5:
            raise RuntimeError(
                f"TailSpecialist: too few tail positives "
                f"(lo={int(y_lo.sum())}, hi={int(y_hi.sum())})"
            )
        common = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            objective="binary",
            verbose=-1,
        )
        if sample_weight is None:
            common["class_weight"] = "balanced"
        self.clf_lo_ = lgb.LGBMClassifier(**common)
        self.clf_hi_ = lgb.LGBMClassifier(**common)
        if sample_weight is not None:
            self.clf_lo_.fit(X, y_lo, sample_weight=sample_weight)
            self.clf_hi_.fit(X, y_hi, sample_weight=sample_weight)
        else:
            self.clf_lo_.fit(X, y_lo)
            self.clf_hi_.fit(X, y_hi)
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        if self.clf_lo_ is None:
            raise RuntimeError("TailSpecialist.predict_dist called before fit")
        if not deps_oof or self.upstream not in deps_oof:
            raise ValueError(
                f"TailSpecialist.predict_dist needs deps_oof[{self.upstream!r}]"
            )
        upstream = deps_oof[self.upstream]
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        per_row_edges = self._row_edges(ids)
        # Discretise upstream on each row's grid via integrate().
        body = upstream.integrate(per_row_edges)            # BracketForecast
        body_probs = body.probs                              # (N, B_max), NaN-padded
        body_edges = body.edges                              # (N, B_max+1)
        B_per_row = (~np.isnan(body_probs)).sum(axis=1).astype(int)
        # Tail probs from classifiers.
        p_lo = np.clip(self.clf_lo_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        p_hi = np.clip(self.clf_hi_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        # Disagreement check vs upstream body's first/last bin.
        upstream_p_lo = body_probs[np.arange(N), 0]
        rows_idx = np.arange(N)
        upstream_p_hi = body_probs[rows_idx, B_per_row - 1]
        max_disagreement = float(np.maximum(
            np.abs(p_lo - upstream_p_lo), np.abs(p_hi - upstream_p_hi),
        ).max())
        if max_disagreement > 0.5:
            import warnings
            warnings.warn(
                f"TailSpecialist: classifier tail probabilities disagree "
                f"with upstream EMOS by up to {max_disagreement:.2f} on "
                f"the outer bins. The classifier outputs *replace* the "
                f"upstream's edge-bin mass — large disagreement on a "
                f"narrow ladder usually means the EMOS body is dominating "
                f"the tails. Consider widening the ladder.",
                UserWarning, stacklevel=2,
            )
        # Per-row: replace bins [0] and [B_i-1], rescale [1 .. B_i-2] to
        # (1 - p_lo - p_hi).
        out_probs = np.full_like(body_probs, np.nan)
        for i in range(N):
            B_i = int(B_per_row[i])
            if B_i < 3:
                raise ValueError(
                    f"TailSpecialist row {i}: ladder has only {B_i} bins; need ≥3"
                )
            row = body_probs[i, :B_i].copy()
            inner = row[1:-1]
            inner_sum = float(inner.sum())
            if inner_sum <= 0:
                raise ValueError(
                    f"TailSpecialist row {i}: upstream has zero body mass in "
                    f"inner bins [1..{B_i-2}]. Refusing to redistribute uniformly."
                )
            body_total = max(0.0, 1.0 - p_lo[i] - p_hi[i])
            inner_scaled = inner * (body_total / inner_sum)
            new_row = np.concatenate([[p_lo[i]], inner_scaled, [p_hi[i]]])
            s = new_row.sum()
            if s <= 0:
                raise ValueError(
                    f"TailSpecialist row {i}: row sum non-positive after "
                    f"renormalisation — should be unreachable; investigate."
                )
            new_row = new_row / s
            out_probs[i, :B_i] = new_row
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_brackets(
            edges=body_edges, probs=out_probs,
            ids=ids, timestamps=timestamps,
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# LinearPoolDist — convex combination of upstream DistributionForecasts.
# ---------------------------------------------------------------------------


@dataclass
class LinearPoolDist(BaseEstimator):
    """Linear (mixture) opinion pool over K upstream dists:

        F(y | x) = Σ_k w_k · F_k(y | x),    w_k ≥ 0,  Σ w_k = 1

    Weights are GLOBAL (not per-row) and fit by minimising weighted-empirical
    CRPS on OOF. Per-component samples drawn from a fixed mid-rank τ grid
    via ppf — so each upstream backing must support ppf.

    Output backing: quantile, evaluated at a 99-point τ grid by inverting
    the weighted empirical CDF of stacked component samples. Tail policy:
    clip.

    For Gaussian-only upstream a closed-form mixture-CRPS exists (Grimit
    et al., 2006) — left as a v0.2 optimisation.
    """

    deps: tuple[str, ...]
    n_samples: int = 200
    name: str = "LinearPoolDist"
    weights_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if len(self.deps) < 2:
            raise ValueError(
                f"LinearPoolDist needs ≥2 upstream deps; got {self.deps}"
            )
        self.depends_on = tuple(self.deps)

    def _sample_grid(self) -> np.ndarray:
        # Mid-rank τ grid in (0, 1); excludes endpoints so parametric-normal
        # tails don't blow up to ±inf.
        return (np.arange(self.n_samples) + 0.5) / self.n_samples

    def _component_samples(self, deps_oof: dict[str, Any]) -> np.ndarray:
        """Return (K, N, n_samples) sample tensor from upstream ppfs."""
        taus = self._sample_grid()
        cols = [deps_oof[name].ppf(taus) for name in self.depends_on]
        return np.stack(cols, axis=0)

    @staticmethod
    def _weighted_crps(
        stacked: np.ndarray,                       # (N, M) — M = K·S
        sample_w: np.ndarray,                      # (M,) sums to 1
        y: np.ndarray,                             # (N,)
    ) -> np.ndarray:
        """Per-row weighted-empirical CRPS.

        CRPS = Σ_j w_j |x_j - y|  -  0.5 · Σ_{j,k} w_j w_k |x_j - x_k|

        Vectorised pairwise term via sorted-sample identity:
            0.5 · Σ_{j,k} w_j w_k |x_j - x_k|
              = Σ_j w_j (x_j · cum_w_j  -  cum_wx_j)
        where cum_w / cum_wx are cumulative sums over x-sorted samples.
        """
        N, M = stacked.shape
        term1 = (sample_w[None, :] * np.abs(stacked - y[:, None])).sum(axis=1)
        order = np.argsort(stacked, axis=1)
        s_sorted = np.take_along_axis(stacked, order, axis=1)
        w_sorted = sample_w[order]
        cum_w = np.cumsum(w_sorted, axis=1)
        cum_wx = np.cumsum(w_sorted * s_sorted, axis=1)
        pairwise = (w_sorted * (s_sorted * cum_w - cum_wx)).sum(axis=1)
        return term1 - pairwise

    def _objective(
        self,
        w: np.ndarray,
        comp_samples: np.ndarray,                  # (K, N, S)
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        K, N, S = comp_samples.shape
        stacked = comp_samples.transpose(1, 0, 2).reshape(N, K * S)
        sample_w = np.repeat(w, S) / S
        crps = self._weighted_crps(stacked, sample_w, y)
        if sample_weight is None:
            return float(crps.mean())
        sw = np.asarray(sample_weight, dtype=float)
        return float((sw * crps).sum() / sw.sum())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        from scipy.optimize import minimize

        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"LinearPoolDist.fit needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        y = np.asarray(y, dtype=float)
        comp_samples = self._component_samples(deps_oof)
        K = comp_samples.shape[0]
        w0 = np.full(K, 1.0 / K)
        bounds = [(0.0, 1.0)] * K
        constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        res = minimize(
            self._objective, w0,
            args=(comp_samples, y, sample_weight),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 200},
        )
        if not res.success:
            raise RuntimeError(
                f"LinearPoolDist: weight optimisation failed: {res.message}"
            )
        w_fit = np.clip(res.x, 0.0, 1.0)
        self.weights_ = w_fit / w_fit.sum()
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        from bracketlearn.forecast import TailPolicy, TailRule

        if self.weights_ is None:
            raise RuntimeError("LinearPoolDist.predict_dist called before fit")
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"LinearPoolDist.predict_dist needs deps_oof for {self.depends_on}"
            )
        comp_samples = self._component_samples(deps_oof)        # (K, N, S)
        K, N, S = comp_samples.shape
        stacked = comp_samples.transpose(1, 0, 2).reshape(N, K * S)
        sample_w = np.repeat(self.weights_, S) / S

        taus_out = np.linspace(0.01, 0.99, 99)
        qvals = np.empty((N, taus_out.size))
        for i in range(N):
            order = np.argsort(stacked[i])
            s_sorted = stacked[i][order]
            w_sorted = sample_w[order]
            cum = np.cumsum(w_sorted)
            qvals[i] = np.interp(taus_out, cum, s_sorted)
        qvals = np.maximum.accumulate(qvals, axis=1)            # isotonic-repair

        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_quantiles(
            taus=taus_out, qvals=qvals,
            tail_policy=TailPolicy.same(TailRule.clip()),
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# CDFBoostBracket — B LightGBM heads on upstream-CDF features → bracket dist.
# ---------------------------------------------------------------------------


@dataclass
class CDFBoostBracket(BaseEstimator):
    """B LightGBM binary classifiers over upstream-CDF features.

    Construction
        - ``edges`` (B+1,): bracket ladder. B = ``len(edges) - 1`` bins.
        - ``deps``: K upstream DistForecaster names.

    Feature matrix per row (passed to all B heads): the CDF of each upstream
    dist evaluated at every ladder edge → shape ``(K * (B+1),)``. Optionally
    concat raw X with ``include_raw_X=True`` (off by default — keeps the
    "dist features only" framing clean).

    Training: for each bin b, classifier_b predicts ``y_b = 1[edges[b] <= y < edges[b+1]]``.
    Outputs (N, B) probabilities, row-renormalised → bracket-backed dist.

    Why this rather than linear stacking on upstream µ:
      - sees the full CDF shape, not a point summary
      - tree splits can model conditional "trust schedules" across regimes
      - output is bracket-backed: natural fit for laddered contract pricing

    Compare with:
      - LinearPoolDist:    convex combination, global weights, full-dist mixture
      - DistAsFeatures + NGBoostNormal:  Gaussian output, dist-summary features
      - CumulativeBinary:  single classifier with cutpoint augmentation
    """

    deps: tuple[str, ...]
    brackets_by_id: dict[Any, np.ndarray]
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    include_raw_X: bool = False
    name: str = "CDFBoostBracket"
    clfs_: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.deps:
            raise ValueError("CDFBoostBracket requires at least one upstream dep")
        if not isinstance(self.brackets_by_id, dict) or not self.brackets_by_id:
            raise ValueError(
                "CDFBoostBracket needs a non-empty brackets_by_id dict "
                "(id → 1-D edge array)"
            )
        # Uniform-B requirement: all rows must share the same bin count
        # so that B head classifiers can be trained. Edge *values* may
        # differ — only B is fixed.
        Bs = set()
        for k, e in self.brackets_by_id.items():
            e_arr = np.asarray(e, dtype=float)
            if e_arr.ndim != 1 or e_arr.size < 3:
                raise ValueError(
                    f"brackets_by_id[{k!r}]: ladder must have ≥2 bins (≥3 edges); "
                    f"got shape {e_arr.shape}"
                )
            if np.any(np.diff(e_arr) <= 0):
                raise ValueError(
                    f"brackets_by_id[{k!r}]: edges must be strictly increasing"
                )
            Bs.add(e_arr.size - 1)
        if len(Bs) > 1:
            raise ValueError(
                f"CDFBoostBracket requires uniform bin count across rows; "
                f"saw {sorted(Bs)}. Use per-row trainers like TailSpecialist or "
                f"CumulativeBinary for ragged B."
            )
        self._B = next(iter(Bs))
        self.depends_on = tuple(self.deps)

    def _featurize(
        self,
        X: np.ndarray | None,
        ids: np.ndarray,
        deps_oof: dict[str, Any],
    ) -> np.ndarray:
        # Per-row CDF of each upstream dist evaluated at the row's own
        # edges. cdf_at_grid returns (N, B+1) for a (N, B+1) edge array.
        per_row_edges = np.stack(
            [np.asarray(self.brackets_by_id[k], dtype=float) for k in ids], axis=0,
        )                                                # (N, B+1)
        cols = [deps_oof[name].cdf_at_grid(per_row_edges) for name in self.depends_on]
        Z = np.column_stack(cols)                        # (N, K * (B+1))
        if self.include_raw_X:
            if X is None:
                raise ValueError(
                    "CDFBoostBracket.include_raw_X=True but X was None"
                )
            X = np.asarray(X, dtype=float)
            Z = np.column_stack([X, Z])
        return Z

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"CDFBoostBracket.fit needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        Z = self._featurize(X, ids, deps_oof)
        B = self._B
        # Per-row bin assignment of y under each row's own edges.
        N = y.shape[0]
        bin_idx = np.empty(N, dtype=int)
        for i in range(N):
            e_i = np.asarray(self.brackets_by_id[ids[i]], dtype=float)
            k = int(np.searchsorted(e_i, y[i], side="right") - 1)
            bin_idx[i] = max(0, min(k, B - 1))

        common = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            objective="binary",
            verbose=-1,
        )
        self.clfs_ = []
        for b in range(B):
            y_b = (bin_idx == b).astype(int)
            if y_b.sum() < 2:
                base_rate = float(y_b.mean())
                self.clfs_.append(("const", base_rate))
                continue
            clf = lgb.LGBMClassifier(**common, class_weight="balanced")
            if sample_weight is not None:
                clf.fit(Z, y_b, sample_weight=sample_weight)
            else:
                clf.fit(Z, y_b)
            self.clfs_.append(("model", clf))
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        if not self.clfs_:
            raise RuntimeError("CDFBoostBracket.predict_dist called before fit")
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"CDFBoostBracket.predict_dist needs deps_oof for {self.depends_on}"
            )
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        Z = self._featurize(X, ids, deps_oof)
        N = Z.shape[0]
        B = len(self.clfs_)
        probs = np.empty((N, B))
        for b, (kind, model) in enumerate(self.clfs_):
            if kind == "const":
                probs[:, b] = model
            else:
                probs[:, b] = model.predict_proba(Z)[:, 1]
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise RuntimeError(
                "CDFBoostBracket: all-zero row in predict_proba "
                "(no head fired); check upstream dist coverage"
            )
        probs = probs / row_sum
        # Per-row edges from brackets_by_id (uniform B across rows by
        # construction, so the stacked array has no NaN padding).
        per_row_edges = np.stack(
            [np.asarray(self.brackets_by_id[k], dtype=float) for k in ids], axis=0,
        )
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_brackets(
            edges=per_row_edges, probs=probs,
            ids=ids, timestamps=timestamps,
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# BracketClassifier — one classifier on (X, lo, hi) → P(y ∈ [lo, hi)).
# ---------------------------------------------------------------------------


@dataclass
class BracketClassifier(BaseEstimator):
    """Single binary classifier with bracket bounds as features.

    For each (row i, bracket b) pair, build one augmented training
    example with features ``[X_i, lo_b, hi_b]`` and target
    ``1[y_i ∈ [lo_b, hi_b))`` (last bracket is closed on the right).
    Any sklearn-style classifier with ``fit`` + ``predict_proba`` works
    — Logistic, GradientBoosting, RandomForest, LightGBM, MLP, …

    At predict time the classifier scores every bracket for every row;
    row-wise probabilities are clipped to (ε, 1-ε) and renormalised to
    sum to 1, producing a bracket-backed dist.

    Why this and not CumulativeBinary or CDFBoostBracket
    -----------------------------------------------------
    - ``CumulativeBinary`` models P(y ≤ k) and differences the CDF.
      Naturally monotone-constrainable → locks the estimator to LGBM.
      Edge feature: a single cutpoint scalar.
    - ``CDFBoostBracket`` trains B separate classifier heads on
      upstream-CDF features. Requires uniform B across rows and an
      upstream dist forecaster as a dep.
    - ``BracketClassifier`` trains ONE classifier on bracket-conditional
      bernoulli targets with the bracket bounds as features. Ragged B
      across rows is fine. No upstream dep required. Any predict_proba
      classifier works. Closest analogue: a sklearn classifier you'd
      use to predict "did the Kalshi 68–69° bracket resolve YES".

    The output is bracket-backed (per-row ladder from ``brackets_by_id``).
    No tail policy: the per-row ladder is the support.
    """

    estimator: Any
    brackets_by_id: dict[Any, np.ndarray]
    name: str = "BracketClassifier"
    depends_on: tuple[str, ...] = ()
    clip_eps: float = 1e-6
    model_: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not hasattr(self.estimator, "fit"):
            raise ValueError(
                "BracketClassifier: estimator must have a .fit() method"
            )
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                "BracketClassifier: estimator must have a .predict_proba() method; "
                f"got {type(self.estimator).__name__}. Use a classifier such as "
                "sklearn.linear_model.LogisticRegression, "
                "sklearn.ensemble.GradientBoostingClassifier, "
                "or lightgbm.LGBMClassifier."
            )
        if not isinstance(self.brackets_by_id, dict) or not self.brackets_by_id:
            raise ValueError(
                "BracketClassifier needs a non-empty brackets_by_id dict "
                "(id → 1-D edge array)"
            )
        for k, e in self.brackets_by_id.items():
            e_arr = np.asarray(e, dtype=float)
            if e_arr.ndim != 1 or e_arr.size < 3:
                raise ValueError(
                    f"brackets_by_id[{k!r}]: ladder must have ≥2 bins "
                    f"(≥3 edges); got shape {e_arr.shape}"
                )
            if np.any(np.diff(e_arr) <= 0):
                raise ValueError(
                    f"brackets_by_id[{k!r}]: edges must be strictly increasing"
                )
        if not (0.0 < self.clip_eps < 0.5):
            raise ValueError(
                f"clip_eps must be in (0, 0.5); got {self.clip_eps}"
            )

    def _augment(
        self,
        X: np.ndarray,
        ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Build the per-bracket augmented design matrix.

        Returns ``(X_aug, offsets, per_row_edges)`` where
        ``offsets[i] : offsets[i+1]`` covers row i's B_i augmented
        rows, and ``per_row_edges[i]`` is the row's edge array
        (length B_i + 1).
        """
        N = X.shape[0]
        per_row_edges: list[np.ndarray] = []
        missing: list[Any] = []
        for k in ids:
            try:
                per_row_edges.append(
                    np.asarray(self.brackets_by_id[k], dtype=float),
                )
            except KeyError:
                missing.append(k)
        if missing:
            raise KeyError(
                f"BracketClassifier: brackets_by_id missing {len(missing)} "
                f"id(s); first: {missing[:3]}"
            )
        B_per_row = np.array([e.size - 1 for e in per_row_edges], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(B_per_row)])
        M = int(offsets[-1])
        n_feat = X.shape[1]
        X_aug = np.empty((M, n_feat + 2), dtype=float)
        for i in range(N):
            sl = slice(int(offsets[i]), int(offsets[i + 1]))
            X_aug[sl, :n_feat] = X[i]
            e_i = per_row_edges[i]
            X_aug[sl, n_feat] = e_i[:-1]      # lo
            X_aug[sl, n_feat + 1] = e_i[1:]   # hi
        return X_aug, offsets, per_row_edges

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        if X.shape[0] != y.shape[0] or X.shape[0] != ids.shape[0]:
            raise ValueError(
                f"shape mismatch: X N={X.shape[0]} y N={y.shape[0]} "
                f"ids N={ids.shape[0]}"
            )
        X_aug, offsets, per_row_edges = self._augment(X, ids)
        N = X.shape[0]
        y_aug = np.zeros(X_aug.shape[0], dtype=int)
        for i in range(N):
            e_i = per_row_edges[i]
            # Bin containing y_i (right-side searchsorted minus one).
            # k ∈ {-1, …, B_i}; only 0 ≤ k < B_i fires a positive label.
            k = int(np.searchsorted(e_i, y[i], side="right") - 1)
            if 0 <= k < e_i.size - 1:
                y_aug[int(offsets[i]) + k] = 1
        # Loud rail: if no positives, the classifier can't fit a non-degenerate
        # boundary and predict_proba may return single-column output.
        if y_aug.sum() == 0:
            raise RuntimeError(
                "BracketClassifier: no row's y landed in any of its "
                "configured brackets — every augmented target is 0. "
                "Check brackets_by_id covers the y range."
            )
        from bracketlearn.trainers._common import _estimator_accepts_sample_weight

        self.model_ = self.estimator
        if sample_weight is not None and _estimator_accepts_sample_weight(self.model_):
            sw = np.asarray(sample_weight, dtype=float)
            sw_aug = np.empty(X_aug.shape[0], dtype=float)
            for i in range(N):
                sl = slice(int(offsets[i]), int(offsets[i + 1]))
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
    ) -> DistributionForecast:
        if self.model_ is None:
            raise RuntimeError("BracketClassifier.predict_dist called before fit")
        X = np.asarray(X, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        N = X.shape[0]
        X_aug, offsets, per_row_edges = self._augment(X, ids)
        proba_aug = self.model_.predict_proba(X_aug)
        if proba_aug.shape[1] != 2:
            raise RuntimeError(
                f"BracketClassifier: estimator.predict_proba returned "
                f"{proba_aug.shape[1]} columns, expected 2 (binary). "
                f"This usually means the training labels were single-class."
            )
        p_aug = np.clip(proba_aug[:, 1], self.clip_eps, 1.0 - self.clip_eps)
        B_per_row = np.array(
            [e.size - 1 for e in per_row_edges], dtype=int,
        )
        B_max = int(B_per_row.max())
        edges_out = np.full((N, B_max + 1), np.nan, dtype=float)
        probs_out = np.full((N, B_max), np.nan, dtype=float)
        for i in range(N):
            sl = slice(int(offsets[i]), int(offsets[i + 1]))
            p_row = p_aug[sl]
            s = float(p_row.sum())
            if s <= 0:
                raise RuntimeError(
                    f"BracketClassifier row {i}: clipped probabilities "
                    f"summed to {s}; should be unreachable given clip_eps > 0."
                )
            p_row = p_row / s
            B_i = int(B_per_row[i])
            probs_out[i, :B_i] = p_row
            edges_out[i, : B_i + 1] = per_row_edges[i]
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_brackets(
            edges=edges_out, probs=probs_out,
            ids=ids, timestamps=timestamps,
            provenance=prov,
        )
