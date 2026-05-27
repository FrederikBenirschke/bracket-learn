"""Bracket-backed DistForecasters and dist-pool combiners.

CumulativeBinary, TailSpecialist, CDFBoostBracket (bracket-emitting trainers).
LinearPoolDist (convex combination of upstream dists).
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

    Fits one classifier over (X, cutpoint) → 1[y ≤ cutpoint] using a fixed grid
    of cutpoints (derived from the bracket ladder); at predict time, queries
    P(y ≤ k) for each k in the grid and emits a quantile-backed dist where
    taus = the cdf values at the configured cutpoints.

    Constructed with a fixed `cutpoints` array — typically the interior
    edges of the eval bracket ladder. Caller MUST also pass
    ``outer_edges=(low, high)`` defining the bracket boundaries below
    ``cutpoints[0]`` and above ``cutpoints[-1]``. The two outer bins
    absorb the tail mass under TailRule.clip semantics; without explicit
    outer edges, downstream mean/variance would be biased by an invented
    pad.
    """

    cutpoints: np.ndarray
    outer_edges: tuple[float, float]
    n_estimators: int = 80
    learning_rate: float = 0.05
    num_leaves: int = 7
    min_child_samples: int = 100
    monotone: bool = True
    name: str = "CumulativeBinary"
    depends_on: tuple[str, ...] = ()
    model_: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        lo, hi = self.outer_edges
        cuts = np.asarray(self.cutpoints, dtype=float)
        if cuts.size == 0:
            return  # fit() will raise on this; defer.
        if not (lo < cuts[0]):
            raise ValueError(
                f"outer_edges[0]={lo} must be strictly less than cutpoints[0]={cuts[0]}"
            )
        if not (hi > cuts[-1]):
            raise ValueError(
                f"outer_edges[1]={hi} must be strictly greater than cutpoints[-1]={cuts[-1]}"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        cuts = np.asarray(self.cutpoints, dtype=float)
        if cuts.size == 0:
            raise ValueError("CumulativeBinary requires at least one cutpoint")
        N = X.shape[0]
        # Build augmented training set: each train row × each cutpoint.
        X_aug = np.repeat(X, cuts.size, axis=0)
        cut_col = np.tile(cuts, N).reshape(-1, 1)
        X_aug = np.hstack([X_aug, cut_col])
        y_aug = (np.repeat(y, cuts.size) <= np.tile(cuts, N)).astype(int)
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
            sw_aug = np.repeat(sample_weight, cuts.size)
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
        cuts = np.asarray(self.cutpoints, dtype=float)
        N = X.shape[0]
        # Build query: each test row × each cutpoint.
        X_aug = np.repeat(X, cuts.size, axis=0)
        cut_col = np.tile(cuts, N).reshape(-1, 1)
        X_aug = np.hstack([X_aug, cut_col])
        proba = self.model_.predict_proba(X_aug)[:, 1].reshape(N, cuts.size)
        # Isotonic repair across rows in one pass (non-monotonicity is rare
        # with monotone=True but the LGBM constraint isn't strict for binary).
        proba = np.maximum.accumulate(proba, axis=1)
        # taus = proba values at the cutpoints (per row), but quantile-backed
        # storage assumes shared taus across rows. Reverse the roles: store
        # cutpoints as qvals, and use proba as per-row taus — but that breaks
        # the shared-tau invariant too. Instead, define a *fixed tau grid*
        # using rank of cutpoints (e.g. (1..K)/(K+1)) and let qvals = cutpoints.
        # Simpler: emit a bracket-backed dist on edges = [-inf-equivalent,
        # cuts, +inf-equivalent], probs = diff(0, proba_at_cuts, 1).
        #
        # Emit a bracket dist over [outer_edges[0], cuts..., outer_edges[1]].
        # The two outer bins absorb tail mass under TailRule.clip. The outer
        # edges are explicit constructor args (no invented pad).
        lo, hi = self.outer_edges
        edges = np.concatenate([[lo], cuts, [hi]])
        # CDF at the inner edges = classifier proba; at the outer edges = 0 and 1.
        cdf_at_edges = np.column_stack([
            np.zeros(N),
            proba,
            np.ones(N),
        ])
        probs = bracket_probs_from_cdf_at_edges(
            cdf_at_edges, source="CumulativeBinary.predict_dist",
        )
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# TailSpecialist — EMOS body + LightGBM tail classifiers (DistForecaster).
# ---------------------------------------------------------------------------


@dataclass
class TailSpecialist(BaseEstimator):
    """Gaussian body (from upstream EMOS μ̂/σ̂) + LightGBM tail classifiers.

    depends_on a parametric-normal upstream (typically named 'emos') and a ladder
    (edges) — fits two binary classifiers for the first/last bracket
    indicators, then rescales the Gaussian body probs to (1 - p_lo - p_hi).
    """

    edges: np.ndarray
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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        if not deps_oof or self.upstream not in deps_oof:
            raise ValueError(
                f"TailSpecialist.fit needs deps_oof[{self.upstream!r}]"
            )
        edges = np.asarray(self.edges, dtype=float)
        if edges.size < 4:
            raise ValueError(
                f"TailSpecialist needs ladder with ≥3 brackets (4 edges); got {edges.size}"
            )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # First-bracket indicator: y < edges[1]. Last-bracket: y >= edges[-2].
        y_lo = (y < edges[1]).astype(int)
        y_hi = (y >= edges[-2]).astype(int)
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
        # class_weight="balanced" only when the caller does NOT supply
        # sample_weight (don't silently multiply user weights
        # by sklearn's balanced inverse-frequency weights).
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
        # Discretise EMOS dist onto ladder.
        edges = np.asarray(self.edges, dtype=float)
        cdf_hi = upstream.cdf(edges[1:])
        cdf_lo = upstream.cdf(edges[:-1])
        body_probs = np.clip(cdf_hi - cdf_lo, 0.0, 1.0)
        N, B = body_probs.shape
        # Tail probs from classifiers.
        X = np.asarray(X, dtype=float)
        p_lo = np.clip(self.clf_lo_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        p_hi = np.clip(self.clf_hi_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        # Sanity check: when the ladder is narrow, the upstream EMOS
        # may put substantial mass in body bins 0 and B-1 (the bins
        # the tail classifiers are about to replace). Discarding that
        # mass is the *point* of TailSpecialist — but if the classifier
        # disagrees with the upstream by more than `tail_disagreement_tol`,
        # the user is probably running on a ladder too narrow for this
        # trainer's design. Warn loudly so it's not silent.
        upstream_p_lo = body_probs[:, 0]
        upstream_p_hi = body_probs[:, -1]
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
        # Rescale inner bins (1..B-1, i.e. all but first and last) to
        # (1 - p_lo - p_hi). Refuse to silently fabricate a
        # uniform body when the upstream returns zero inner mass —
        # that means the upstream is degenerate and the caller needs
        # to know.
        inner = body_probs[:, 1:-1]
        inner_sum = inner.sum(axis=1, keepdims=True)
        if np.any(inner_sum <= 0):
            n_bad = int((inner_sum.ravel() <= 0).sum())
            raise ValueError(
                f"TailSpecialist.predict_dist: {n_bad}/{N} rows have zero "
                f"upstream body mass in bins [1..B-2]. Refusing to "
                f"redistribute uniformly."
            )
        body_total = np.maximum(0.0, 1.0 - p_lo - p_hi)[:, None]
        inner_scaled = inner * (body_total / inner_sum)
        probs = np.concatenate(
            [p_lo[:, None], inner_scaled, p_hi[:, None]], axis=1,
        )
        # Final renorm against numerical drift (clip+scale can leave
        # rounding errors at the 1e-15 level). Any row sum at or below
        # zero indicates a logic error, not drift — raise.
        row_sum = probs.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError(
                "TailSpecialist.predict_dist: row sum is non-positive after "
                "renormalisation — should be unreachable; investigate."
            )
        probs = probs / row_sum
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
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
    edges: np.ndarray
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
        edges = np.asarray(self.edges, dtype=float)
        if edges.ndim != 1 or edges.size < 3:
            raise ValueError(
                f"edges must be 1-D with ≥3 entries (≥2 bins); got shape {edges.shape}"
            )
        if np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing")
        self.edges = edges
        self.depends_on = tuple(self.deps)

    def _featurize(
        self,
        X: np.ndarray | None,
        deps_oof: dict[str, Any],
    ) -> np.ndarray:
        cols = [deps_oof[name].cdf(self.edges) for name in self.depends_on]
        Z = np.column_stack(cols)
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
        Z = self._featurize(X, deps_oof)
        B = self.edges.size - 1
        # Bin assignment: index of bin each y falls into. y outside [edges[0],
        # edges[-1]] is clipped to the nearest bin (if you want
        # to forbid out-of-range y, do it at the caller).
        bin_idx = np.searchsorted(self.edges, y, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, B - 1)

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
                # Degenerate bin: no positives. Store sentinel = base rate.
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
        Z = self._featurize(X, deps_oof)
        N = Z.shape[0]
        B = len(self.clfs_)
        probs = np.empty((N, B))
        for b, (kind, model) in enumerate(self.clfs_):
            if kind == "const":
                probs[:, b] = model
            else:
                probs[:, b] = model.predict_proba(Z)[:, 1]
        # Row-renorm (heads are independent, so sums won't be 1).
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise RuntimeError(
                "CDFBoostBracket: all-zero row in predict_proba "
                "(no head fired); check upstream dist coverage"
            )
        probs = probs / row_sum
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_brackets(
            edges=self.edges, probs=probs,
            ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )
