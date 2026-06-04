"""Combiner / meta trainers — forecasts built from upstream forecasts.

Every trainer here consumes the out-of-fold distributions of one or more
upstream models (received positionally via ``upstream=[...]`` under a
``Stacker``) and emits a combined forecast. They are gathered in one module
because "combine upstreams" is a single concept; the split by output backing
(parametric vs bracket) is incidental.

- ``StackedParametric`` / ``BMAStacking`` — parametric meta-learners over
  upstream (μ, σ): OLS-of-μ and Bayesian model averaging.
- ``DistAsFeatures`` — generic bridge: upstream dists become a feature matrix
  for any downstream trainer.
- ``BracketStacking`` — learned per-bracket combination of upstream bracket
  probabilities.
- ``LinearPoolDist`` — convex (linear) opinion pool of upstream dists.
- ``TailSpecialist`` — Gaussian body from an upstream EMOS + LightGBM tail
  classifiers, on per-row brackets.
- ``CDFBoostBracket`` — gradient-boosted CDF correction on a bracket grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
from scipy.optimize import minimize

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    BracketForecast,
    DistributionForecast,
    MixtureNormalForecast,
    NormalForecast,
    PointForecast,
    ProvenanceMeta,
    StudentTForecast,
)
from bracketlearn.trainers._compose_util import resolve_upstream, upstream_label

# Subclasses that count as "parametric upstream" for the stacking trainers.
_PARAMETRIC_BACKINGS = (NormalForecast, StudentTForecast, MixtureNormalForecast)

# Euler-Mascheroni constant. Used by StackedParametric(sigma_method=
# 'geometric_mean_upstream') to debias E[log Z²] under Gaussian residuals:
# for Z ~ N(0, 1), E[log Z²] = −γ_E − log 2.
_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# DistAsFeatures — generic bridge: upstream dists → feature matrix → any trainer.
# ---------------------------------------------------------------------------


_DIST_FEATURE_TAUS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class DistAsFeatures(BaseEstimator):
    """Materialise K upstream distributions into a feature matrix and hand it
    to a downstream forecaster.

    Per-row features extracted from each upstream dist:

        - quantiles at ``feature_taus`` (default 5 quantiles)
        - mean (if ``include_mean=True``)
        - variance (if ``include_variance=True``)
        - CDF at ``tail_cutpoints`` (tail-mass features)

    Total per row: ``K * (len(feature_taus) + include_mean + include_variance + len(tail_cutpoints))``.

    The downstream forecaster sees ONLY dist-derived features, not raw X.
    If you also want raw X, build a separate node — keeping this class
    single-purpose is intentional.

    Upstream forecasts arrive **positionally** via ``upstream=[dist, ...]``
    (the `Stacker` contract); ``DistAsFeatures`` featurizes them in that order.

    Requires each upstream backing to support ``ppf`` for the requested
    ``feature_taus``. v0.1 ppf coverage: parametric-normal, mixture-normal,
    quantile, bracket.
    """

    downstream: Any = None
    feature_taus: tuple[float, ...] = _DIST_FEATURE_TAUS
    tail_cutpoints: tuple[float, ...] = ()
    include_mean: bool = True
    include_variance: bool = True
    name: str = "DistAsFeatures"
    _n_features_: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.downstream is None:
            raise ValueError("DistAsFeatures requires a downstream forecaster")

    def _featurize(self, ups: list[Any]) -> np.ndarray:
        taus = np.asarray(self.feature_taus, dtype=float)
        cuts = np.asarray(self.tail_cutpoints, dtype=float)
        cols: list[np.ndarray] = []
        for d in ups:
            cols.append(d.ppf(taus))                  # (N, len(taus))
            if self.include_mean:
                cols.append(d.mean()[:, None])
            if self.include_variance:
                cols.append(d.variance()[:, None])
            if cuts.size:
                cols.append(d.cdf(cuts))              # (N, len(cuts))
        return np.column_stack(cols)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        upstream: list[Any] | None = None,
    ) -> Self:
        ups = resolve_upstream(upstream, where="DistAsFeatures.fit")
        Z = self._featurize(ups)
        # Forward sample_weight only if downstream accepts it; matches the
        # SklearnPoint convention.
        try:
            self.downstream.fit(Z, y, sample_weight=sample_weight)
        except TypeError:
            self.downstream.fit(Z, y)
        self._n_features_ = Z.shape[1]
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        upstream: list[Any] | None = None,
    ) -> PointForecast:
        Z = self._predict_features(upstream)
        return self.downstream.predict(Z, ids=ids, timestamps=timestamps)

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        Z = self._predict_features(upstream)
        return self.downstream.predict_dist(Z, ids=ids, timestamps=timestamps)

    def _predict_features(self, upstream: list[Any] | None) -> np.ndarray:
        ups = resolve_upstream(upstream, where="DistAsFeatures.predict")
        Z = self._featurize(ups)
        if Z.shape[1] != self._n_features_:
            raise RuntimeError(
                f"DistAsFeatures: train had {self._n_features_} features; "
                f"predict produced {Z.shape[1]}"
            )
        return Z


# ---------------------------------------------------------------------------
# BracketStacking — multiclass classifier over concatenated bracket-prob deps.
# ---------------------------------------------------------------------------


@dataclass
class BracketStacking(BaseEstimator):
    """Meta-learner: ``estimator`` over concatenated bracket-prob vectors.

    Counterpart to ``BMAStacking`` for BRACKET-form upstreams. Each
    upstream's per-row probability vector ``(N, K)`` is concatenated
    along columns; the resulting ``(N, K * len(deps))`` feature matrix
    is fed to a multiclass classifier whose label is the row's realized
    bracket index (0..K-1). At predict time the classifier's
    ``predict_proba`` becomes the row's bracket distribution.

    Why this and not ``BMAStacking``: BMA produces convex weight
    combinations on the simplex, which can only interpolate between
    upstreams. A LightGBM (or any non-linear) multiclass head learns
    *regime-conditional* interactions — "trust EMOS when forecasts
    disagree, market when they cluster" — that a convex pool cannot
    express. Empirically this matters: stacking a LightGBM head over
    bracket probs typically beats convex pooling by 20-40% on logloss.

    Why not ``DistAsFeatures``: that primitive extracts a fixed feature
    set (quantiles, mean, var) from each upstream, then runs a downstream
    *point* or *dist* forecaster on those features. It loses the
    bracket-prob shape information — the raw per-bin probabilities
    are not in its feature set. BracketStacking preserves the full
    bracket-prob shape across all upstreams and lets the classifier
    learn directly on those vectors.

    Contract:

    * All upstreams must be ``BracketForecast`` with matching per-row edges
      (same K, same boundaries). Rows where upstreams disagree on edges are
      caller-resolved — typically by filtering to the modal K and dropping
      non-conforming rows.
    * ``estimator`` must be sklearn-compatible with ``predict_proba``.
      ``num_class`` is auto-set from observed K when the estimator
      accepts that parameter (LightGBM, sklearn classifiers); otherwise
      caller pre-configures it.

    Predict-time edges are taken from the first upstream — since the contract
    requires all upstreams share edges, any one is canonical.
    """

    estimator: Any = None
    name: str = "BracketStacking"
    K_: int | None = field(default=None, init=False)
    edges_template_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.estimator is None:
            raise ValueError("BracketStacking requires an estimator")

    def _assemble(
        self, ups: list[Any], N: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Concatenate per-row prob vectors across all upstreams.

        Returns ``(Z, K, edges_ref)``: ``Z`` is ``(N, K * len(ups))``
        feature matrix; ``edges_ref`` is the (N, K+1) edge array from
        the first upstream (all must agree on edges).
        """
        cols: list[np.ndarray] = []
        K: int | None = None
        edges_ref: np.ndarray | None = None
        for i, d in enumerate(ups):
            label = upstream_label(i)
            if not isinstance(d, BracketForecast):
                raise NotImplementedError(
                    f"BracketStacking expects bracket-backed upstream; "
                    f"{label} is {type(d).__name__}"
                )
            probs = np.asarray(d.probs, dtype=float)
            if probs.shape[0] != N:
                raise ValueError(
                    f"BracketStacking: upstream {label} has N={probs.shape[0]} rows, "
                    f"expected N={N}"
                )
            if K is None:
                K = int(probs.shape[1])
                edges_ref = np.asarray(d.edges, dtype=float)
            elif int(probs.shape[1]) != K:
                raise ValueError(
                    f"BracketStacking: upstream {label} has K={probs.shape[1]} bins, "
                    f"expected K={K} (all upstreams must share bracket count)"
                )
            cols.append(probs)
        assert K is not None and edges_ref is not None
        return np.column_stack(cols), K, edges_ref

    def _validate_ids(
        self,
        ups: list[Any],
        caller_ids: np.ndarray | None,
    ) -> None:
        """Match BMAStacking's ids-alignment contract."""
        upstream_ids = None
        for i, d in enumerate(ups):
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"BracketStacking: upstream {upstream_label(i)}.ids "
                    "does not match the first upstream's ids — rows would be misaligned"
                )
        if (
            caller_ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(caller_ids), upstream_ids)
        ):
            raise ValueError(
                "BracketStacking: caller's ids do not match upstream ids — "
                "rows would be misaligned"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        upstream: list[Any] | None = None,
        labels: np.ndarray | None = None,
    ) -> Self:
        """Fit the multiclass head.

        ``labels`` (optional) overrides the default ``realized_bin(y)``
        derivation. Use it when the upstream's edges don't reflect
        the true bin assignment — e.g. Kalshi overlapping brackets,
        where multiple brackets contain y and the caller has its own
        "first match" tie-breaker. When omitted, the first upstream's
        ``realized_bin(y)`` provides the labels and rows with
        non-finite y are dropped.
        """
        ups = resolve_upstream(upstream, where="BracketStacking.fit")
        y = np.asarray(y, dtype=float)
        N = y.shape[0]
        self._validate_ids(ups, ids)
        Z, K, edges_ref = self._assemble(ups, N)
        if labels is not None:
            labels_arr = np.asarray(labels, dtype=int)
            if labels_arr.shape != (N,):
                raise ValueError(
                    f"BracketStacking: labels must be (N,)={N}, got {labels_arr.shape}"
                )
            if np.any(labels_arr < 0) or np.any(labels_arr >= K):
                raise ValueError(
                    f"BracketStacking: labels must be in [0, K={K}); "
                    f"got min={int(labels_arr.min())}, max={int(labels_arr.max())}"
                )
            valid = np.ones(N, dtype=bool)
        else:
            labels_arr = ups[0].realized_bin(y).astype(int)
            # realized_bin already clips to [0, K-1] — no negative labels possible.
            # Filter out rows with non-finite y (would have produced 0-clip silently).
            valid = np.isfinite(y)
        if int(valid.sum()) < K * 2:
            raise RuntimeError(
                f"BracketStacking.fit: only {int(valid.sum())} valid rows for "
                f"{K}-class multiclass — too few (need ≥ 2*K)"
            )
        # Auto-set num_class if estimator accepts it; LightGBM needs this
        # at construction for multiclass — but it also accepts a re-fit
        # with num_class=K via set_params.
        try:
            self.estimator.set_params(num_class=K)
        except (ValueError, AttributeError):
            pass
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape != (N,):
                raise ValueError(
                    f"BracketStacking: sample_weight must be (N,)={N}, got {sw.shape}"
                )
            fit_kwargs["sample_weight"] = sw[valid]
        self.estimator.fit(Z[valid], labels_arr[valid], **fit_kwargs)
        self.K_ = K
        self.edges_template_ = edges_ref
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        upstream: list[Any] | None = None,
    ) -> BracketForecast:
        if self.K_ is None:
            raise RuntimeError("BracketStacking.predict_dist called before fit")
        ups = resolve_upstream(upstream, where="BracketStacking.predict_dist")
        N = len(ids)
        self._validate_ids(ups, np.asarray(ids))
        Z, K, edges_ref = self._assemble(ups, N)
        if K != self.K_:
            raise ValueError(
                f"BracketStacking: predict K={K} != train K={self.K_}"
            )
        proba = np.asarray(
            self.estimator.predict_proba(Z), dtype=float
        )
        if proba.shape != (N, K):
            raise RuntimeError(
                f"BracketStacking: estimator.predict_proba returned shape "
                f"{proba.shape}, expected ({N}, {K})"
            )
        # Re-normalize against float rounding so BracketForecast.from_arrays
        # accepts (it requires per-row sum within 1e-6 of 1).
        proba_sum = proba.sum(axis=1, keepdims=True)
        if np.any(proba_sum <= 0):
            raise RuntimeError(
                "BracketStacking: estimator returned a row with zero total "
                "probability — predict_proba contract violated"
            )
        proba = proba / proba_sum
        return BracketForecast.from_arrays(
            edges=edges_ref,
            probs=proba,
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=ProvenanceMeta.placeholder(self.name),
        )

# ---------------------------------------------------------------------------
# StackedParametric — DistForecaster meta-learner over upstream μ (and
# optionally σ), received positionally via ``upstream=[...]`` under a
# ``Stacker``.
# ---------------------------------------------------------------------------


@dataclass
class StackedParametric(BaseEstimator):
    """Meta-learner over upstream forecasters' parametric outputs.

    Defaults reproduce v0.1 ``StackedParametric`` behaviour exactly: OLS over
    upstream μ with intercept (unconstrained), constant σ̂ from residual
    std, Gaussian output. The optional knobs below widen the surface.

    ``weight_constraint``:
        * ``"unconstrained"`` (default) — OLS with intercept; μ-weights
          take any sign and any magnitude.
        * ``"convex"`` — Σ wₖ = 1, wₖ ≥ 0 via SLSQP (classic Breiman
          1996 stacking). Intercept stays free so it can absorb any
          common bias in the upstream μ scale.

    ``sigma_method``:
        * ``"constant"`` (default) — σ̂ = std(in-sample residuals);
          single scalar applied to every row.
        * ``"geometric_mean_upstream"`` — per-row dispersion modelled as
          σ̂(x) = exp(α + Σ wⱼ · log σⱼ(x)). Fit by OLS regressing the
          bias-corrected target ``0.5·(log(resid² + ε) + γ_E + log 2)``
          on per-upstream log σⱼ(x), where the additive constant
          ``γ_E + log 2`` debiases E[log Z²] for Gaussian residuals.
          Requires every upstream to expose a positive σ in
          ``params['sigma']`` (i.e. parametric Normal / Student-t).
          ε is a small floor on resid² so resid=0 rows do not produce
          −∞ targets.

    ``dist_family``:
        * ``"normal"`` (default) — N(μ̂, σ̂²).
        * ``"student_t"`` — t_ν(μ̂, scale) with ν = ``student_t_df``.
          The fitted σ̂ is interpreted as the standard deviation of
          residuals (matches the residual-fit semantics); it is
          converted to the t-distribution *scale* parameter via
          ``scale = σ̂ · sqrt((ν − 2) / ν)`` so the forecast variance
          equals σ̂² regardless of ν.

    Upstream forecasts arrive **positionally** via ``upstream=[dist, ...]``
    (the ``Stacker`` contract) — this reads ``.params['mu']`` (and
    ``['sigma']`` when ``sigma_method='geometric_mean_upstream'``) from each,
    in declared order.
    """

    name: str = "StackedParametric"
    weight_constraint: Literal["unconstrained", "convex"] = "unconstrained"
    sigma_method: Literal["constant", "geometric_mean_upstream"] = "constant"
    dist_family: Literal["normal", "student_t"] = "normal"
    student_t_df: float = 5.0
    weights_: np.ndarray | None = field(default=None, init=False)
    intercept_: float | None = field(default=None, init=False)
    sigma_: float | None = field(default=None, init=False)
    sigma_alpha_: float | None = field(default=None, init=False)
    sigma_log_weights_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.weight_constraint not in ("unconstrained", "convex"):
            raise ValueError(
                f"StackedParametric.weight_constraint must be 'unconstrained' or "
                f"'convex'; got {self.weight_constraint!r}"
            )
        if self.sigma_method not in ("constant", "geometric_mean_upstream"):
            raise ValueError(
                f"StackedParametric.sigma_method must be 'constant' or "
                f"'geometric_mean_upstream'; got {self.sigma_method!r}"
            )
        if self.dist_family not in ("normal", "student_t"):
            raise ValueError(
                f"StackedParametric.dist_family must be 'normal' or 'student_t'; "
                f"got {self.dist_family!r}"
            )
        if self.dist_family == "student_t" and self.student_t_df <= 2.0:
            raise ValueError(
                f"StackedParametric(dist_family='student_t'): student_t_df must be > 2 "
                f"for finite variance; got {self.student_t_df}"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        upstream: list[Any] | None = None,
    ) -> Self:
        ups = resolve_upstream(upstream, where="StackedParametric.fit")
        y = np.asarray(y, dtype=float)
        # Stack upstream μ predictions row-aligned. We REQUIRE that each
        # upstream dist's .ids matches our (X, y) row order (no silent
        # misalignment). If the caller passes ids, we check
        # them; if not, we still require all upstream dists to agree on
        # their own ids vectors (else the meta-learner builds rows from
        # mis-zipped predictions).
        upstream_ids = None
        for i, d in enumerate(ups):
            label = upstream_label(i)
            if not isinstance(d, _PARAMETRIC_BACKINGS):
                raise NotImplementedError(
                    f"StackedParametric expects parametric upstream; "
                    f"{label} is {type(d).__name__}"
                )
            if d.params["mu"].shape[0] != y.shape[0]:
                raise ValueError(
                    f"StackedParametric.fit: upstream {label} has N={d.params['mu'].shape[0]} "
                    f"but y has N={y.shape[0]}"
                )
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"StackedParametric.fit: upstream {label}.ids does not match the "
                    f"first upstream's ids — meta-learner rows would be misaligned"
                )
        if (
            ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(ids), upstream_ids)
        ):
            raise ValueError(
                "StackedParametric.fit: caller's ids do not match upstream ids — "
                "rows would be misaligned"
            )
        cols = [d.params["mu"] for d in ups]
        Z = np.column_stack(cols)  # (N, K)
        if self.weight_constraint == "unconstrained":
            self._fit_mu_unconstrained(Z, y, sample_weight)
        else:
            self._fit_mu_convex(Z, y, sample_weight)
        resid = y - (self.intercept_ + Z @ self.weights_)
        # Refuse degenerate residuals before either σ branch. v0.1 floored
        # σ̂≤0 to 1e-3, which produced near-deterministic forecasts and
        # masked upstream-μ-vs-y collinearity (data leak). The same
        # collinearity poisons the geometric σ fit (target → −∞), so we
        # guard once on the residual std and let both σ branches proceed.
        y_scale = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
        resid_std = float(np.std(resid, ddof=1))
        if resid_std <= max(1e-9 * max(y_scale, 1.0), 1e-12):
            raise ValueError(
                f"StackedParametric.fit: residual std is degenerate "
                f"(resid_std={resid_std:.3g}, y_scale={y_scale:.3g}); "
                f"meta-learner perfectly fits training y, which means either "
                f"upstream μ collinearity with y (data leak) or N is too small. "
                f"Refusing to substitute a 1e-3 floor."
            )
        if self.sigma_method == "constant":
            self.sigma_ = resid_std
        else:
            self._fit_sigma_geometric_upstream(
                resid, y_scale, sample_weight, ups,
            )
        return self

    def _fit_mu_unconstrained(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
        # OLS with intercept (weighted if sample_weight given).
        A = np.column_stack([np.ones(Z.shape[0]), Z])
        if sample_weight is None:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            sw = np.sqrt(np.asarray(sample_weight, dtype=float))
            sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        self.intercept_ = float(sol[0])
        self.weights_ = sol[1:]

    def _fit_mu_convex(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
        K = Z.shape[1]
        if sample_weight is None:
            def loss(params: np.ndarray) -> float:
                resid = y - params[0] - Z @ params[1:]
                return float(np.mean(resid ** 2))
        else:
            sw = np.asarray(sample_weight, dtype=float)
            sw_sum = float(sw.sum())

            def loss(params: np.ndarray) -> float:
                resid = y - params[0] - Z @ params[1:]
                return float((sw * resid ** 2).sum() / sw_sum)

        x0 = np.concatenate(
            [[float(np.mean(y) - np.mean(Z))], np.full(K, 1.0 / K)],
        )
        bounds = [(None, None)] + [(0.0, 1.0)] * K
        constraints = {
            "type": "eq",
            "fun": lambda p: float(np.sum(p[1:]) - 1.0),
        }
        res = minimize(
            loss, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if not res.success:
            raise RuntimeError(
                f"StackedParametric(weight_constraint='convex'): SLSQP failed: "
                f"{res.message}"
            )
        self.intercept_ = float(res.x[0])
        self.weights_ = np.asarray(res.x[1:], dtype=float)

    def _fit_sigma_geometric_upstream(
        self,
        resid: np.ndarray,
        y_scale: float,
        sample_weight: np.ndarray | None,
        ups: list[Any],
    ) -> None:
        cols_sigma = self._collect_upstream_sigma(ups, where="fit")
        Z_log = np.column_stack([np.log(c) for c in cols_sigma])
        # Floor on resid² protects exact-zero rows from −∞ targets while
        # staying well below typical residual variance.
        eps = (1e-6 * max(y_scale, 1.0)) ** 2
        target = 0.5 * (np.log(resid ** 2 + eps) + _EULER_GAMMA + math.log(2.0))
        A = np.column_stack([np.ones(Z_log.shape[0]), Z_log])
        if sample_weight is None:
            sol, *_ = np.linalg.lstsq(A, target, rcond=None)
        else:
            sw = np.sqrt(np.asarray(sample_weight, dtype=float))
            sol, *_ = np.linalg.lstsq(A * sw[:, None], target * sw, rcond=None)
        self.sigma_alpha_ = float(sol[0])
        self.sigma_log_weights_ = np.asarray(sol[1:], dtype=float)

    def _collect_upstream_sigma(
        self,
        ups: list[Any],
        *,
        where: str,
    ) -> list[np.ndarray]:
        cols: list[np.ndarray] = []
        for i, d in enumerate(ups):
            label = upstream_label(i)
            if "sigma" not in d.params:
                raise ValueError(
                    f"StackedParametric({where}, sigma_method='geometric_mean_upstream'): "
                    f"upstream {label} has no σ in params "
                    f"(type={type(d).__name__}); either pick "
                    f"sigma_method='constant' or feed parametric upstreams"
                )
            s = np.asarray(d.params["sigma"], dtype=float)
            if np.any(s <= 0):
                n_bad = int(np.sum(s <= 0))
                raise ValueError(
                    f"StackedParametric({where}): upstream {label} has {n_bad} "
                    f"non-positive σ values; cannot take log for "
                    f"geometric_mean_upstream"
                )
            cols.append(s)
        return cols

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        # At predict time, the driver must have re-run the upstream stages
        # on the current X; it passes their dist positionally.
        ups = resolve_upstream(upstream, where="StackedParametric.predict_dist")
        # Row-alignment check: each upstream's ids must match the caller's ids
        # exactly (no silent misalignment).
        ids_arr = np.asarray(ids)
        for i, d in enumerate(ups):
            if not np.array_equal(d.ids, ids_arr):
                raise ValueError(
                    f"StackedParametric.predict_dist: upstream "
                    f"{upstream_label(i)}.ids does not "
                    f"match caller ids — rows would be misaligned"
                )
        cols = [d.params["mu"] for d in ups]
        Z = np.column_stack(cols)
        mu = self.intercept_ + Z @ self.weights_
        if self.sigma_method == "constant":
            sigma_std = np.full_like(mu, self.sigma_)
        else:
            cols_sigma = self._collect_upstream_sigma(ups, where="predict_dist")
            Z_log = np.column_stack([np.log(c) for c in cols_sigma])
            sigma_std = np.exp(self.sigma_alpha_ + Z_log @ self.sigma_log_weights_)
        ids_arr = np.asarray(ids)
        ts_arr = np.asarray(timestamps)
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        if self.dist_family == "normal":
            return DistributionForecast.from_normal(
                mu, sigma_std, ids=ids_arr, timestamps=ts_arr, provenance=prov,
            )
        # student_t: σ̂ is residual std; convert to t-scale so variance == σ̂².
        nu = self.student_t_df
        scale = sigma_std * math.sqrt((nu - 2.0) / nu)
        df_arr = np.full_like(mu, nu)
        return DistributionForecast.from_student_t(
            mu, scale, df_arr, ids=ids_arr, timestamps=ts_arr, provenance=prov,
        )


# ---------------------------------------------------------------------------
# BMAStacking — Bayesian model averaging meta-learner. Mixture-of-Normals output.
# ---------------------------------------------------------------------------


@dataclass
class BMAStacking(BaseEstimator):
    """Bayesian model averaging meta-learner. DistForecaster over upstreams.

    Replaces ``StackedParametric``'s OLS-of-μ with a posterior over the mixture
    weight vector ``w`` on the K-simplex, and emits a true
    ``MixtureNormalForecast`` instead of a Normal collapsed onto a
    constant residual σ̂.

    Model::

        y_i | w ~ Σ_k w_k · N(y_i; μ_{k,i}, σ_{k,i})
        w        ~ Dir(α_0, …, α_0)                  (symmetric concentration)

    where each upstream forecaster k contributes (μ_{k,i}, σ_{k,i}) per
    row i from its OOF ``DistributionForecast``. For non-Normal
    parametric upstreams (Student-t, MixtureNormal) we use the marginal
    moments — μ = ``dist.mean()``, σ = √``dist.variance()`` — i.e. the
    standard moment-matching BMA approximation.

    Fit (EM with Dirichlet prior):

    * E-step: γ_{ik} = w_k · L_{ik} / Σ_j w_j L_{ij}, where
      L_{ik} = N(y_i; μ_{k,i}, σ_{k,i}).
    * M-step: α_n_k = α_0 + Σ_i s_i · γ_{ik}, then
      w_k = α_n_k / Σ_j α_n_j (posterior mean).

    s_i = sample_weight_i (1 if unweighted). Iterates until
    ‖w_new − w‖∞ < ``tol`` or ``max_iter`` is reached — non-convergence
    raises (Rule #0.5; partial weights would silently misweight tails).

    Predict at new x*: the pipeline re-runs upstreams on the inference
    rows; each contributes (μ_{k,*}, σ_{k,*}). Output is
    ``MixtureNormalForecast`` with weights = posterior mean w broadcast
    to (N, K).

    Why this beats ``StackedParametric``:

    * Per-row output σ — the mixture's standard deviation grows wherever
      upstream μ̂'s disagree on that row. ``StackedParametric``'s σ̂ is one scalar
      from training residuals.
    * No σ̂ → 0 collapse (the v0.1 ``StackedParametric`` pathology). The mixture
      σ is bounded below by min_k σ_{k,i}.
    * Weights live on the simplex (no extrapolation pathology from
      unconstrained OLS coefficients).
    """

    alpha_prior: float = 1.0
    max_iter: int = 500
    tol: float = 1e-6
    name: str = "BMAStacking"
    weights_: np.ndarray | None = field(default=None, init=False)
    alpha_n_: np.ndarray | None = field(default=None, init=False)
    n_iter_: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.alpha_prior <= 0:
            raise ValueError(
                f"BMAStacking: alpha_prior must be > 0 (got {self.alpha_prior}); "
                "improper Dirichlet priors not supported."
            )

    @staticmethod
    def _upstream_moments(dist: Any, N: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract per-row (μ, σ) from any parametric upstream via the
        DistributionForecast moment API. Reject non-parametric backings."""
        if not isinstance(dist, _PARAMETRIC_BACKINGS):
            raise NotImplementedError(
                f"BMAStacking expects parametric upstream; "
                f"{name} is {type(dist).__name__}"
            )
        mu = np.asarray(dist.mean(), dtype=float)
        var = np.asarray(dist.variance(), dtype=float)
        if mu.shape != (N,) or var.shape != (N,):
            raise ValueError(
                f"BMAStacking: upstream {name} returned mean/variance with "
                f"shape {mu.shape}/{var.shape}, expected ({N},)"
            )
        if np.any(var <= 0):
            raise ValueError(
                f"BMAStacking: upstream {name} has non-positive variance "
                "on some rows — likelihood would be undefined."
            )
        return mu, np.sqrt(var)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        upstream: list[Any] | None = None,
    ) -> Self:
        ups = resolve_upstream(upstream, where="BMAStacking.fit")
        y = np.asarray(y, dtype=float)
        N = y.shape[0]
        # Row-alignment guard. Same contract as StackedParametric — upstream ids
        # must agree with each other and with the caller's ids (if given).
        upstream_ids = None
        for i, d in enumerate(ups):
            if upstream_ids is None:
                upstream_ids = d.ids
            elif not np.array_equal(upstream_ids, d.ids):
                raise ValueError(
                    f"BMAStacking.fit: upstream {upstream_label(i)}.ids "
                    "does not match the first upstream's ids — mixture rows would be misaligned"
                )
        if (
            ids is not None
            and upstream_ids is not None
            and not np.array_equal(np.asarray(ids), upstream_ids)
        ):
            raise ValueError(
                "BMAStacking.fit: caller's ids do not match upstream ids — "
                "rows would be misaligned"
            )
        # Collect per-row moments.
        K = len(ups)
        mu = np.empty((N, K))
        sigma = np.empty((N, K))
        for j, d in enumerate(ups):
            mu[:, j], sigma[:, j] = self._upstream_moments(
                d, N, upstream_label(j),
            )
        # Likelihood matrix L_ik = N(y_i; μ_{k,i}, σ_{k,i}).
        z = (y[:, None] - mu) / sigma
        log_L = -0.5 * z ** 2 - np.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        # Numerical-stability trick: subtract per-row max before exp.
        log_L_max = log_L.max(axis=1, keepdims=True)
        L = np.exp(log_L - log_L_max)  # (N, K), per-row max == 1
        s = (
            np.asarray(sample_weight, dtype=float)
            if sample_weight is not None
            else np.ones(N)
        )
        if s.shape != (N,) or np.any(s < 0):
            raise ValueError(
                "BMAStacking: sample_weight must be 1-D, same length as y, non-negative."
            )
        # EM loop. Convergence on Δ weighted-log-likelihood (the EM objective)
        # rather than Δw — when upstreams are near-duplicates, w drifts
        # linearly toward the fixed point with no real change in the
        # objective, and tol-on-Δw spuriously fails. Δll is the proper
        # criterion (and Σ Δll = 0 implies w is stationary).
        w = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        delta_ll = np.inf
        # Constant offset from the per-row log_L_max subtraction (cancels in
        # γ but matters for the true log-likelihood reported below).
        offset = float((s * log_L_max[:, 0]).sum())
        for it in range(self.max_iter):
            num = w[None, :] * L            # (N, K)
            denom = num.sum(axis=1, keepdims=True)
            if np.any(denom <= 0):
                raise ValueError(
                    "BMAStacking.fit: row likelihood is zero under all components — "
                    "upstream μ̂'s sit too far from y on some rows. Check upstream "
                    "fit or widen σ_floor on upstreams."
                )
            # Marginal log-likelihood under the current mixture weights.
            ll = float((s * np.log(denom[:, 0])).sum()) + offset
            delta_ll = ll - prev_ll
            if it > 0 and abs(delta_ll) < self.tol * max(abs(ll), 1.0):
                self.n_iter_ = it + 1
                break
            gamma = num / denom              # (N, K), rows sum to 1
            alpha_n = float(self.alpha_prior) + (s[:, None] * gamma).sum(axis=0)
            w = alpha_n / alpha_n.sum()
            prev_ll = ll
        else:
            raise RuntimeError(
                f"BMAStacking.fit: EM did not converge in {self.max_iter} "
                f"iterations (last Δlog-lik = {delta_ll:.3g}). "
                "Raise max_iter or check upstream quality."
            )
        self.weights_ = w
        self.alpha_n_ = alpha_n
        return self

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        if self.weights_ is None:
            raise RuntimeError("BMAStacking.predict_dist called before fit")
        ups = resolve_upstream(upstream, where="BMAStacking.predict_dist")
        ids_arr = np.asarray(ids)
        N = ids_arr.shape[0]
        for i, d in enumerate(ups):
            if not np.array_equal(d.ids, ids_arr):
                raise ValueError(
                    f"BMAStacking.predict_dist: upstream "
                    f"{upstream_label(i)}.ids does "
                    "not match caller ids — mixture rows would be misaligned"
                )
        K = len(ups)
        mu = np.empty((N, K))
        sigma = np.empty((N, K))
        for j, d in enumerate(ups):
            mu[:, j], sigma[:, j] = self._upstream_moments(
                d, N, upstream_label(j),
            )
        weights = np.broadcast_to(self.weights_, (N, K)).copy()
        prov = ProvenanceMeta.placeholder(self.name, sigma_source="native")
        return DistributionForecast.from_mixture_normal(
            weights=weights, mus=mu, sigmas=sigma,
            ids=ids_arr, timestamps=np.asarray(timestamps),
            provenance=prov,
        )



# ---------------------------------------------------------------------------
# TailSpecialist — EMOS body + LightGBM tail classifiers (DistForecaster).
# ---------------------------------------------------------------------------


@dataclass
class TailSpecialist(BaseEstimator):
    """Gaussian body (from upstream EMOS μ̂/σ̂) + LightGBM tail classifiers,
    on per-row brackets.

    Takes a single parametric-normal upstream (positionally, via the
    ``Stacker`` contract) and a per-row bracket ladder via ``brackets_by_id``
    (id → 1-D edge array). Fits two global binary classifiers — one for "y in row's
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
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    name: str = "TailSpecialist"
    clf_lo_: Any = field(default=None, init=False)
    clf_hi_: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
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
            raise KeyError(f"TailSpecialist: brackets_by_id missing id {e!r}") from e

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        sample_weight: np.ndarray | None = None,
        upstream: list[Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        # Validate an upstream is present (fit reads only X/y; the body
        # Gaussian is consumed at predict time).
        resolve_upstream(upstream, where="TailSpecialist.fit")
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
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        if self.clf_lo_ is None:
            raise RuntimeError("TailSpecialist.predict_dist called before fit")
        up_dist = resolve_upstream(
            upstream, where="TailSpecialist.predict_dist",
        )[0]
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        per_row_edges = self._row_edges(ids)
        # Discretise upstream on each row's grid via integrate().
        body = up_dist.integrate(per_row_edges)             # BracketForecast
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

    n_samples: int = 200
    name: str = "LinearPoolDist"
    weights_: np.ndarray | None = field(default=None, init=False)

    def _sample_grid(self) -> np.ndarray:
        # Mid-rank τ grid in (0, 1); excludes endpoints so parametric-normal
        # tails don't blow up to ±inf.
        return (np.arange(self.n_samples) + 0.5) / self.n_samples

    def _resolve(self, upstream: list[Any] | None, *, where: str) -> list[Any]:
        ups = resolve_upstream(upstream, where=where)
        if len(ups) < 2:
            raise ValueError(
                f"{where}: LinearPoolDist needs ≥2 upstreams; got {len(ups)}"
            )
        return ups

    def _component_samples(self, ups: list[Any]) -> np.ndarray:
        """Return (K, N, n_samples) sample tensor from upstream ppfs."""
        taus = self._sample_grid()
        cols = [d.ppf(taus) for d in ups]
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
        upstream: list[Any] | None = None,
    ) -> Self:
        from scipy.optimize import minimize

        ups = self._resolve(upstream, where="LinearPoolDist.fit")
        y = np.asarray(y, dtype=float)
        comp_samples = self._component_samples(ups)
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
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        from bracketlearn.forecast import TailPolicy, TailRule

        if self.weights_ is None:
            raise RuntimeError("LinearPoolDist.predict_dist called before fit")
        ups = self._resolve(upstream, where="LinearPoolDist.predict_dist")
        comp_samples = self._component_samples(ups)        # (K, N, S)
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
        - ``brackets_by_id``: id → 1-D edge array (B = len(edges) - 1 bins,
          uniform across rows).
        - K upstream DistForecasters arrive positionally via ``upstream=[...]``.

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

    brackets_by_id: dict[Any, np.ndarray]
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    include_raw_X: bool = False
    name: str = "CDFBoostBracket"
    clfs_: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # The upstream count is resolved at fit from ``upstream=[...]``.
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

    def _featurize(
        self,
        X: np.ndarray | None,
        ids: np.ndarray,
        ups: list[Any],
    ) -> np.ndarray:
        # Per-row CDF of each upstream dist evaluated at the row's own
        # edges. cdf_at_grid returns (N, B+1) for a (N, B+1) edge array.
        per_row_edges = np.stack(
            [np.asarray(self.brackets_by_id[k], dtype=float) for k in ids], axis=0,
        )                                                # (N, B+1)
        cols = [d.cdf_at_grid(per_row_edges) for d in ups]
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
        upstream: list[Any] | None = None,
    ) -> Self:
        import lightgbm as lgb

        ups = resolve_upstream(upstream, where="CDFBoostBracket.fit")
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        Z = self._featurize(X, ids, ups)
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
        upstream: list[Any] | None = None,
    ) -> DistributionForecast:
        if not self.clfs_:
            raise RuntimeError("CDFBoostBracket.predict_dist called before fit")
        ups = resolve_upstream(upstream, where="CDFBoostBracket.predict_dist")
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        Z = self._featurize(X, ids, ups)
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
