"""Meta / bridge trainers — upstream dists become features for any trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    DistributionForecast,
    PointForecast,
)

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

    The downstream's own ``depends_on`` is ignored; ``DistAsFeatures`` owns
    the dep contract via its ``deps`` argument.

    Requires each upstream backing to support ``ppf`` for the requested
    ``feature_taus``. v0.1 ppf coverage: parametric-normal, mixture-normal,
    quantile, bracket.
    """

    deps: tuple[str, ...]
    downstream: Any
    feature_taus: tuple[float, ...] = _DIST_FEATURE_TAUS
    tail_cutpoints: tuple[float, ...] = ()
    include_mean: bool = True
    include_variance: bool = True
    name: str = "DistAsFeatures"
    _n_features_: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.deps:
            raise ValueError("DistAsFeatures requires at least one upstream dep")
        self.depends_on = tuple(self.deps)

    def _featurize(self, deps_oof: dict[str, Any]) -> np.ndarray:
        taus = np.asarray(self.feature_taus, dtype=float)
        cuts = np.asarray(self.tail_cutpoints, dtype=float)
        cols: list[np.ndarray] = []
        for name in self.depends_on:
            d = deps_oof[name]
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
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"DistAsFeatures.fit needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        Z = self._featurize(deps_oof)
        # Forward sample_weight only if downstream accepts it; matches the
        # SklearnPoint convention.
        try:
            self.downstream.fit(Z, y, sample_weight=sample_weight, deps_oof=None)
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
        deps_oof: dict[str, Any] | None = None,
    ) -> PointForecast:
        Z = self._predict_features(deps_oof)
        return self.downstream.predict(Z, ids=ids, timestamps=timestamps)

    def predict_dist(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        deps_oof: dict[str, Any] | None = None,
    ) -> DistributionForecast:
        Z = self._predict_features(deps_oof)
        return self.downstream.predict_dist(Z, ids=ids, timestamps=timestamps)

    def _predict_features(self, deps_oof: dict[str, Any] | None) -> np.ndarray:
        if not deps_oof or set(self.depends_on) - set(deps_oof):
            raise ValueError(
                f"DistAsFeatures.predict needs deps_oof for {self.depends_on}; "
                f"got {list(deps_oof or [])}"
            )
        Z = self._featurize(deps_oof)
        if Z.shape[1] != self._n_features_:
            raise RuntimeError(
                f"DistAsFeatures: train had {self._n_features_} features; "
                f"predict produced {Z.shape[1]}"
            )
        return Z


