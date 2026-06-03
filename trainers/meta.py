"""Meta / bridge trainers — upstream dists become features for any trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    BracketForecast,
    DistributionForecast,
    PointForecast,
)
from bracketlearn.forecast._meta import ProvenanceMeta
from bracketlearn.trainers._compose_util import resolve_upstream, upstream_label

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
