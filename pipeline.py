"""ForecastPipeline — orchestration (CV + OOF stitching + DAG injection).

sklearn-style API::

    pipeline = ForecastPipeline(
        steps=[
            ("ridge", LiftedForecaster(SklearnPoint(Ridge()), GlobalResidual())),
            ("emos",  CalibratedForecaster(EMOS(), Isotonic())),
            ("stack", StackedParametric(deps=("ridge", "emos"))),
        ],
        cv="expanding-window", n_folds=5,
    )
    result = pipeline.fit_predict(X, y, ids=ids, timestamps=ts)
    print(result.score(y, metrics=["crps", "log_score", "pit"]))

Each step is a plain `(name, forecaster)` tuple. Lifters and calibrators
live inside `LiftedForecaster` / `CalibratedForecaster` wrappers — no
special pipeline slots.

v0.1 vertical slice:

- Expanding-window CV with embargo.
- Per-fold OOF stitching for each registered forecaster.
- depends_on topo-sort: pipeline pre-computes upstream OOF and injects via
  deps_oof dict kwarg to downstream fit.
- LiftedForecaster gets base_oof from an inner half-split.
- CalibratedForecaster gets fit on a held-out calibration tail of each fold.
- fit_predict returns PipelineResult; user calls result.score(y) instead
  of manually aligning OOF coverage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from bracketlearn.base import BaseEstimator, clone
from bracketlearn.forecast import (
    DistributionForecast,
    PointForecast,
    ProvenanceMeta,
)
from bracketlearn.protocols import (
    Calibrator,
    Lifter,
    PointForecaster,
)

# ---------------------------------------------------------------------------
# Composite forecasters: combine simple Forecasters into richer ones.
#
# - LiftedForecaster:     PointForecaster + Lifter      → DistForecaster.
# - CalibratedForecaster: DistForecaster  + Calibrator  → DistForecaster.
#
# Both are flat wrappers — the pipeline keeps a `[(name, forecaster)]` list
# without special slots for lifters/calibrators.
# ---------------------------------------------------------------------------


class LiftedForecaster(BaseEstimator):
    """PointForecaster + Lifter, exposed as a DistForecaster.

    fit signature: ``fit(X, y, *, base_oof: PointForecast)``.
    Pipeline supplies base_oof from its fold structure. Standalone callers
    compute OOF themselves (cross_val_predict → PointForecast → .fit).

    No hidden inner CV. No secret pipeline-state coupling.
    """

    def __init__(
        self,
        base: PointForecaster,
        lifter: Lifter,
        *,
        name: str | None = None,
    ):
        self.base = base
        self.lifter = lifter
        self.name = name or f"{base.name}+{type(lifter).__name__}"
        self.depends_on = base.depends_on

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        base_oof: PointForecast,
        deps_oof: dict[str, Any] | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        self.base.fit(
            X, y,
            sample_weight=sample_weight,
            deps_oof=deps_oof,
        )
        if self.lifter.requires_X:
            self.lifter.fit(base_oof, y, X=X)
        else:
            self.lifter.fit(base_oof, y)
        return self

    def predict_dist(
        self,
        X: Any,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> DistributionForecast:
        point = self.base.predict(X, ids=ids, timestamps=timestamps)
        return self.lifter.lift(point)


class CalibratedForecaster(BaseEstimator):
    """Wraps a DistForecaster with a Calibrator. Pipeline fits the calibrator
    on a held-out tail of each training fold (see ForecastPipeline).

    Mirrors LiftedForecaster: the wrapped trainer stays a plain DistForecaster
    so the pipeline keeps a flat list of (name, forecaster) pairs.
    """

    def __init__(
        self,
        forecaster: Any,
        calibrator: Calibrator,
        *,
        name: str | None = None,
    ):
        self.forecaster = forecaster
        self.calibrator = calibrator
        self.name = name or f"{getattr(forecaster, 'name', type(forecaster).__name__)}+{type(calibrator).__name__}"
        self.depends_on = tuple(getattr(forecaster, "depends_on", ()))

    def fit(self, X: Any, y: np.ndarray, **kwargs: Any):
        self.forecaster.fit(X, y, **kwargs)
        return self

    def predict_dist(self, X: Any, **kwargs: Any) -> DistributionForecast:
        dist = self.forecaster.predict_dist(X, **kwargs)
        if getattr(self.calibrator, "fitted_", True):
            return self.calibrator.transform(dist)
        return dist


# ---------------------------------------------------------------------------
# Metric dispatch registry. Used by PipelineResult.score (and downstream
# leaderboard helpers) to turn a (metric_name, distribution) pair into a
# scalar mean. The registry is keyed by metric; per-backing dispatch lives
# inside each entry so adding a new backing means touching one table, not
# four if/elif blocks (audit item §3.S3).
# ---------------------------------------------------------------------------


def _compute_metric(
    metric: str, dist, y, *, ladder, scoremod,
) -> dict[str, float]:
    """Dispatch one (metric, distribution) pair to a scalar value, or to
    a small dict (PIT contributes both mean and std).
    """
    if metric == "crps":
        return {"crps": float(dist.crps(y).mean())}
    if metric == "log_score":
        return {"log_score": float(dist.log_score(y).mean())}
    if metric in ("pit", "pit_mean", "pit_std"):
        pits = dist.pit(y)
        return {"pit_mean": float(pits.mean()), "pit_std": float(pits.std())}
    if metric == "log_loss_bracket":
        contracts = ladder.price(dist)
        return {"log_loss_bracket": scoremod.log_loss_bracket(
            contracts, ladder.edges, y,
        )}
    if metric == "brier_bracket":
        contracts = ladder.price(dist)
        return {"brier_bracket": scoremod.brier_bracket(
            contracts, ladder.edges, y,
        )}
    raise ValueError(f"unknown metric: {metric!r}")


@dataclass
class _Stage:
    name: str
    forecaster: Any
    depends_on: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# PipelineResult — owns OOF coverage alignment + scoring.
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Holds the OOF DistributionForecast per stage plus the row mapping
    back into the original data (so scoring can align y itself).

    Indexing:
        result["ridge"]  # → DistributionForecast for stage 'ridge'
        result.stages    # → list[str] of stage names

    Scoring (the user never touches dist.ids):
        result.score(y, metrics=["crps", "log_score"])
        result.score(y, metrics=["log_loss_bracket", "brier"], ladder=ladder)
    """

    forecasts: dict[str, DistributionForecast]

    @property
    def stages(self) -> list[str]:
        return list(self.forecasts.keys())

    def __getitem__(self, name: str) -> DistributionForecast:
        return self.forecasts[name]

    def __iter__(self):
        return iter(self.forecasts)

    def items(self):
        return self.forecasts.items()

    def score(
        self,
        y: np.ndarray,
        *,
        metrics: Sequence[str] = ("crps", "log_score", "pit"),
        ladder: Any = None,
    ) -> dict[str, dict[str, float]]:
        """Return {stage_name: {metric_name: value}}.

        Available metrics:
          - "crps"             — mean CRPS for Gaussian backing
          - "log_score"        — mean predictive negative log-likelihood
          - "pit_mean"         — mean PIT (≈ 0.5 if calibrated)
          - "pit_std"          — std of PIT
          - "log_loss_bracket" — requires ladder
          - "brier_bracket"    — requires ladder

        y is the full original target vector; PipelineResult slices it to
        match each stage's OOF coverage via dist.ids.
        """
        from bracketlearn import score as scoremod

        y = np.asarray(y, dtype=float)
        out: dict[str, dict[str, float]] = {}

        needs_ladder = {"log_loss_bracket", "brier_bracket"}
        if needs_ladder & set(metrics) and ladder is None:
            raise ValueError(
                f"metrics {needs_ladder & set(metrics)} require ladder=..."
            )

        for name, dist in self.forecasts.items():
            y_oof = y[dist.ids.astype(int)]
            row: dict[str, float] = {"n_oof": int(dist.ids.shape[0])}
            for m in metrics:
                row.update(_compute_metric(m, dist, y_oof, ladder=ladder, scoremod=scoremod))
            out[name] = row
        return out

    def to_table(
        self,
        y: np.ndarray,
        *,
        metrics: Sequence[str] = ("crps", "log_score", "pit"),
        ladder: Any = None,
    ) -> str:
        """Render score() output as an aligned text table."""
        scores = self.score(y, metrics=metrics, ladder=ladder)
        # Collect columns by union across stages, preserving insertion order.
        cols: list[str] = []
        for row in scores.values():
            for k in row:
                if k not in cols:
                    cols.append(k)
        header = f"{'stage':<10}" + "".join(f"{c:>14}" for c in cols)
        lines = [header, "-" * len(header)]
        for name, row in scores.items():
            line = f"{name:<10}"
            for c in cols:
                v = row.get(c)
                if v is None:
                    line += f"{'-':>14}"
                elif isinstance(v, int):
                    line += f"{v:>14d}"
                else:
                    line += f"{v:>14.4f}"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ForecastPipeline — sklearn-style construction.
# ---------------------------------------------------------------------------


class ForecastPipeline:
    """sklearn-style pipeline for forecasters.

    Construct with a flat list of `(name, forecaster)` tuples — wrap
    PointForecasters in `LiftedForecaster(base, lifter)` and add calibration
    via `CalibratedForecaster(dist_forecaster, calibrator)` at the call site.

    Example::

        from bracketlearn.lift import GlobalResidual, Isotonic
        from bracketlearn.trainers import SklearnPoint, EMOS, StackedParametric
        from sklearn.linear_model import Ridge

        p = ForecastPipeline(
            steps=[
                ("ridge", LiftedForecaster(SklearnPoint(Ridge()), GlobalResidual())),
                ("emos",  CalibratedForecaster(EMOS(), Isotonic())),
                ("stack", StackedParametric(deps=("ridge", "emos"))),
            ],
            n_folds=5,
        )
        result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    """

    def __init__(
        self,
        steps: Sequence[tuple[str, Any]] | None = None,
        *,
        cv: str = "expanding-window",
        n_folds: int = 5,
        embargo: int = 0,
        calibration_fraction: float = 0.2,
        refit_on_full: bool = True,
        shuffle: bool = False,
        random_state: int | None = None,
        rolling_window: int | None = None,
    ):
        _VALID_CV = ("expanding-window", "rolling-window", "kfold")
        if cv not in _VALID_CV:
            raise ValueError(f"cv={cv!r} not in {_VALID_CV}")
        if cv == "rolling-window" and rolling_window is None:
            raise ValueError(
                "cv='rolling-window' requires rolling_window=<int> (train chunk size)"
            )
        if cv == "expanding-window" and shuffle:
            raise ValueError("shuffle=True is incompatible with time-series CV")
        self.cv = cv
        self.n_folds = n_folds
        self.embargo = embargo
        self.calibration_fraction = calibration_fraction
        self.refit_on_full = refit_on_full
        self.shuffle = shuffle
        self.random_state = random_state
        self.rolling_window = rolling_window
        self._stages: list[_Stage] = []
        self._names: set[str] = set()
        # Set by fit_predict when refit_on_full=True; consumed by .predict().
        self._fitted_stages: dict[str, Any] = {}
        self._fitted_calibrators: dict[str, Any] = {}
        for name, forecaster in (steps or []):
            self._register(name, forecaster)

    def _register(self, name: str, forecaster: Any) -> None:
        if name in self._names:
            raise ValueError(f"stage {name!r} already registered")
        stage_deps = tuple(getattr(forecaster, "depends_on", ()))
        for d in stage_deps:
            if d not in self._names:
                raise ValueError(
                    f"stage {name!r} depends on {d!r} but {d!r} not yet registered "
                    "(register dependencies first)"
                )
        self._stages.append(_Stage(name=name, forecaster=forecaster, depends_on=stage_deps))
        self._names.add(name)

    # ------------------------------------------------------------------ run

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> PipelineResult:
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError(
                    f"sample_weight length {sample_weight.shape[0]} != y length {y.shape[0]}"
                )
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape[0] != y.shape[0]:
                raise ValueError(
                    f"groups length {groups.shape[0]} != y length {y.shape[0]}"
                )
        N = y.shape[0]

        feature_hash = _hash_array(X)
        fit_window = (_to_dt(timestamps.min()), _to_dt(timestamps.max()))
        code_sha = "dev"

        # Time-series CV sorts by timestamp; k-fold does not (rows are i.i.d.).
        if self.cv == "kfold":
            order = np.arange(N)
        else:
            order = np.argsort(timestamps, kind="stable")
        Xo = X[order]
        yo = y[order]
        ids_o = ids[order]
        ts_o = timestamps[order]
        sw_o = sample_weight[order] if sample_weight is not None else None
        g_o = groups[order] if groups is not None else None

        folds = self._make_folds(N)

        # Per stage, accumulate (orig_row_index, fold_dist) pairs across folds.
        per_stage_folds: dict[str, list[tuple[np.ndarray, DistributionForecast]]] = {
            s.name: [] for s in self._stages
        }

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            fold_train_dist: dict[str, DistributionForecast] = {}
            fold_test_dist: dict[str, DistributionForecast] = {}

            for stage in self._stages:
                deps_for_fit = {d: fold_train_dist[d] for d in stage.depends_on}
                deps_for_pred = {d: fold_test_dist[d] for d in stage.depends_on}
                prov = ProvenanceMeta(
                    forecaster_name=stage.name,
                    forecaster_version="0.1",
                    fit_window=fit_window,
                    fold_idx=fold_idx,
                    calibration_set_hash=None,
                    random_seed=None,
                    code_sha=code_sha,
                    feature_matrix_hash=feature_hash,
                    created_at=datetime.now(),
                )
                # Clone-per-fold: never mutate the user's instance, and
                # guarantee no fitted-state bleed between folds.
                fold_forecaster = clone(stage.forecaster)
                dist_train, dist_test = self._fit_stage_on_fold(
                    fold_forecaster, Xo, yo, ids_o, ts_o, train_idx, test_idx,
                    deps_for_fit=deps_for_fit, deps_for_pred=deps_for_pred,
                    prov=prov, sample_weight=sw_o, groups=g_o,
                )
                fold_train_dist[stage.name] = dist_train
                fold_test_dist[stage.name] = dist_test
                # ids_o is sorted; map back to original row index via `order`.
                orig_rows = order[test_idx]
                per_stage_folds[stage.name].append((orig_rows, dist_test))

        out: dict[str, DistributionForecast] = {}
        for stage in self._stages:
            prov = ProvenanceMeta(
                forecaster_name=stage.name,
                forecaster_version="0.1",
                fit_window=fit_window,
                fold_idx=None,
                calibration_set_hash=None,
                random_seed=None,
                code_sha=code_sha,
                feature_matrix_hash=feature_hash,
                created_at=datetime.now(),
                sigma_source="native",
            )
            out[stage.name] = _stitch_folds(
                per_stage_folds[stage.name],
                timestamps=timestamps,
                provenance=prov,
            )

        # Refit each stage on the full training data → canonical models for
        # .predict() on truly unseen rows. Stored on self for later use.
        # This is sklearn's standard pattern: CV produces OOF metrics, the
        # final-model fit produces the artefact that scores new data.
        if self.refit_on_full:
            self._fit_canonical_models(Xo, yo, ids_o, ts_o, sw_o, groups=g_o)

        return PipelineResult(forecasts=out)

    def _fit_canonical_models(
        self,
        X: np.ndarray, y: np.ndarray, ids: np.ndarray, ts: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        groups: np.ndarray | None = None,
    ) -> None:
        """Fit each stage on the *full* training data, storing the result on
        ``self._fitted_stages`` for later use by ``predict()``.

        Calibrators are fit on a held-out tail of the full training data,
        matching the per-fold calibration logic. Downstream stages with
        ``depends_on`` receive the upstream stage's in-sample dist on the
        full training data.
        """

        self._fitted_stages = {}
        self._fitted_calibrators = {}
        canonical_dists: dict[str, DistributionForecast] = {}
        N = X.shape[0]
        train_idx = np.arange(N)

        for stage in self._stages:
            deps = {d: canonical_dists[d] for d in stage.depends_on}
            f = clone(stage.forecaster)
            calibrator: Calibrator | None = None
            inner = f
            if isinstance(f, CalibratedForecaster):
                calibrator = f.calibrator
                inner = f.forecaster

            # Fit calibrator on a tail of full train (mirrors fold logic).
            if calibrator is not None:
                calib_n = max(2, int(N * self.calibration_fraction))
                tr_minus = train_idx[:-calib_n]
                calib_idx = train_idx[-calib_n:]
                if len(tr_minus) >= 2:
                    _, cal_dist = self._fit_inner(
                        inner, X, y, ids, ts, tr_minus, calib_idx, deps, deps,
                        sample_weight=sample_weight, groups=groups,
                    )
                    calibrator.fit(cal_dist, y[calib_idx])
                else:
                    calibrator = None

            # Refit inner on the full train and record its in-sample dist for
            # downstream deps. Self-predict on the same N rows so the deps
            # row-alignment invariant (deps_oof[name].params['mu'].shape[0] == N)
            # holds for whatever downstream StackedParametric expects.
            dist_train_full = self._refit_and_predict_full(
                inner, X, y, ids, ts, deps, sample_weight=sample_weight,
                groups=groups,
            )
            if calibrator is not None:
                dist_train_full = calibrator.transform(dist_train_full)
                self._fitted_calibrators[stage.name] = calibrator
            self._fitted_stages[stage.name] = inner
            canonical_dists[stage.name] = dist_train_full

    def _refit_and_predict_full(
        self,
        f: Any,
        X: np.ndarray, y: np.ndarray, ids: np.ndarray, ts: np.ndarray,
        deps: dict[str, DistributionForecast],
        sample_weight: np.ndarray | None = None,
        *,
        groups: np.ndarray | None = None,
    ) -> DistributionForecast:
        """Refit ``f`` on full (X, y); return its in-sample predict_dist."""

        if isinstance(f, LiftedForecaster):
            half = X.shape[0] // 2
            sw_half = sample_weight[:half] if sample_weight is not None else None
            sw_full = sample_weight
            _fit_with_optional_weight(f.base, X[:half], y[:half], sw_half)
            base_pt_oof = f.base.predict(X[half:], ids=ids[half:], timestamps=ts[half:])
            _fit_with_optional_weight(f.base, X, y, sw_full)
            if f.lifter.requires_X:
                f.lifter.fit(base_pt_oof, y[half:], X=X[half:])
            else:
                f.lifter.fit(base_pt_oof, y[half:])
            return f.predict_dist(X, ids=ids, timestamps=ts)
        extras: dict[str, Any] = {"ids": ids, "timestamps": ts}
        if deps:
            extras["deps_oof"] = deps
        if groups is not None:
            extras["groups"] = groups
        _fit_with_optional_weight(f, X, y, sample_weight, **extras)
        return _predict_with_extras(
            f, X, ids, ts, deps_oof=deps if deps else None, groups=groups,
        )

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> dict[str, DistributionForecast]:
        """Predict on unseen X using the canonical (full-train) models.

        Returns ``{stage_name: DistributionForecast}``. Requires
        ``refit_on_full=True`` at construction (the default) and a prior
        ``fit_predict()`` call. ``groups`` is forwarded to any stage whose
        ``predict_dist`` declares it (e.g. ``HierarchicalNormal``).
        """

        if not self._fitted_stages:
            raise RuntimeError(
                "predict() requires a prior fit_predict() with refit_on_full=True"
            )
        X = np.asarray(X)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        if groups is not None:
            groups = np.asarray(groups)
        out: dict[str, DistributionForecast] = {}
        for stage in self._stages:
            f = self._fitted_stages[stage.name]
            deps = {d: out[d] for d in stage.depends_on}
            if isinstance(f, LiftedForecaster):
                dist = f.predict_dist(X, ids=ids, timestamps=timestamps)
            else:
                dist = _predict_with_extras(
                    f, X, ids, timestamps,
                    deps_oof=deps if deps else None,
                    groups=groups,
                )
            cal = self._fitted_calibrators.get(stage.name)
            if cal is not None:
                dist = cal.transform(dist)
            out[stage.name] = dist
        return out

    # ---------------------------------------------------------- fold helpers

    def _make_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.cv == "expanding-window":
            return self._expanding_folds(N)
        if self.cv == "rolling-window":
            return self._rolling_folds(N)
        if self.cv == "kfold":
            return self._kfold_folds(N)
        raise ValueError(f"unknown cv={self.cv!r}")

    def _expanding_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
        chunk_size = N // (self.n_folds + 1)
        if chunk_size < 2:
            raise ValueError(f"N={N} too small for n_folds={self.n_folds}")
        folds = []
        for k in range(self.n_folds):
            train_end = (k + 1) * chunk_size
            test_start = train_end + self.embargo
            # Final fold absorbs N % (n_folds + 1) trailing rows so the
            # OOF prediction set covers every row of the input. Dropping
            # the tail biases summary metrics toward whichever regime
            # the early chunks happen to land in.
            is_last = (k == self.n_folds - 1)
            test_end = N if is_last else min(N, test_start + chunk_size)
            if test_start >= N:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            folds.append((train_idx, test_idx))
        return folds

    def _rolling_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Rolling-window CV: train slice has fixed width ``rolling_window``,
        slides forward by chunk_size each fold. Older rows roll out.
        """
        w = int(self.rolling_window)
        chunk_size = max(2, (N - w) // self.n_folds) if w < N else 0
        if chunk_size < 2:
            raise ValueError(
                f"N={N} too small for rolling_window={w} + n_folds={self.n_folds}"
            )
        folds = []
        for k in range(self.n_folds):
            train_start = k * chunk_size
            train_end = train_start + w
            test_start = train_end + self.embargo
            test_end = min(N, test_start + chunk_size)
            if test_start >= N or train_end > N:
                break
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            folds.append((train_idx, test_idx))
        if not folds:
            raise ValueError(
                f"rolling-window CV produced 0 folds (N={N}, w={w}, n_folds={self.n_folds})"
            )
        return folds

    def _kfold_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Plain k-fold CV: rows split into n_folds disjoint test sets, each
        trained on the complement. Use only when rows are exchangeable —
        not for time-series data (use 'expanding-window' or 'rolling-window').
        """
        if self.n_folds < 2:
            raise ValueError(f"kfold needs n_folds >= 2; got {self.n_folds}")
        if self.n_folds > N:
            raise ValueError(f"N={N} < n_folds={self.n_folds}")
        idx = np.arange(N)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(idx)
        chunks = np.array_split(idx, self.n_folds)
        folds = []
        for k in range(self.n_folds):
            test_idx = np.sort(chunks[k])
            train_idx = np.sort(np.concatenate([chunks[j] for j in range(self.n_folds) if j != k]))
            folds.append((train_idx, test_idx))
        return folds

    # ---------------------------------------------------------- stage fit

    def _fit_stage_on_fold(
        self,
        f: Any,
        X: np.ndarray, y: np.ndarray, ids: np.ndarray, ts: np.ndarray,
        train_idx: np.ndarray, test_idx: np.ndarray,
        *,
        deps_for_fit: dict[str, DistributionForecast],
        deps_for_pred: dict[str, DistributionForecast],
        prov: ProvenanceMeta,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> tuple[DistributionForecast, DistributionForecast]:
        """Fit ``f`` on train_idx; return (dist_on_train, dist_on_test).

        ``f`` should already be a clone — the pipeline never mutates the
        user-supplied forecaster instance.
        """

        # Unwrap CalibratedForecaster: fit calibrator on a tail of train_idx.
        calibrator: Calibrator | None = None
        inner = f
        if isinstance(f, CalibratedForecaster):
            calibrator = f.calibrator
            inner = f.forecaster

        # Fit + predict the inner forecaster, with the train-tail calibration
        # split if needed.
        if calibrator is not None:
            calib_n = max(2, int(len(train_idx) * self.calibration_fraction))
            tr_minus = train_idx[:-calib_n]
            calib_idx = train_idx[-calib_n:]
            if len(tr_minus) < 2:
                # Too few rows to split — skip calibration this fold.
                calibrator = None

        if calibrator is None:
            dist_train, dist_test = self._fit_inner(
                inner, X, y, ids, ts, train_idx, test_idx,
                deps_for_fit, deps_for_pred,
                sample_weight=sample_weight, groups=groups,
            )
        else:
            # 1) fit on train minus calibration tail
            cal_train_dist, cal_dist = self._fit_inner(
                inner, X, y, ids, ts, tr_minus, calib_idx,
                deps_for_fit, deps_for_pred,    # deps approximation OK in v0.1
                sample_weight=sample_weight, groups=groups,
            )
            calibrator.fit(cal_dist, y[calib_idx])
            # 2) refit on full train for canonical predictions
            dist_train, dist_test = self._fit_inner(
                inner, X, y, ids, ts, train_idx, test_idx,
                deps_for_fit, deps_for_pred,
                sample_weight=sample_weight, groups=groups,
            )
            dist_train = calibrator.transform(dist_train)
            dist_test = calibrator.transform(dist_test)

        dist_train = _set_provenance(dist_train, prov)
        dist_test = _set_provenance(dist_test, prov)
        return dist_train, dist_test

    def _fit_inner(
        self,
        f: Any,
        X: np.ndarray, y: np.ndarray, ids: np.ndarray, ts: np.ndarray,
        train_idx: np.ndarray, test_idx: np.ndarray,
        deps_for_fit: dict[str, DistributionForecast],
        deps_for_pred: dict[str, DistributionForecast],
        sample_weight: np.ndarray | None = None,
        *,
        groups: np.ndarray | None = None,
    ) -> tuple[DistributionForecast, DistributionForecast]:

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        ids_tr, ts_tr = ids[train_idx], ts[train_idx]
        ids_te, ts_te = ids[test_idx], ts[test_idx]
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
        g_tr = groups[train_idx] if groups is not None else None
        g_te = groups[test_idx] if groups is not None else None

        if isinstance(f, LiftedForecaster):
            half = len(train_idx) // 2
            base_fit_idx = train_idx[:half]
            base_oof_idx = train_idx[half:]
            sw_half = sample_weight[base_fit_idx] if sample_weight is not None else None
            _fit_with_optional_weight(f.base, X[base_fit_idx], y[base_fit_idx], sw_half)
            base_pt_oof = f.base.predict(
                X[base_oof_idx], ids=ids[base_oof_idx], timestamps=ts[base_oof_idx],
            )
            _fit_with_optional_weight(f.base, X_tr, y_tr, sw_tr)
            if f.lifter.requires_X:
                f.lifter.fit(base_pt_oof, y[base_oof_idx], X=X[base_oof_idx])
            else:
                f.lifter.fit(base_pt_oof, y[base_oof_idx])
            dist_train = f.predict_dist(X_tr, ids=ids_tr, timestamps=ts_tr)
            dist_test = f.predict_dist(X_te, ids=ids_te, timestamps=ts_te)
            return dist_train, dist_test

        # Pass ids/timestamps/deps/groups explicitly. Signature-based
        # routing in _fit_with_optional_weight + _predict_with_extras
        # drops anything a particular trainer doesn't declare.
        fit_extras: dict[str, Any] = {"ids": ids_tr, "timestamps": ts_tr}
        if deps_for_fit:
            fit_extras["deps_oof"] = deps_for_fit
        if g_tr is not None:
            fit_extras["groups"] = g_tr
        _fit_with_optional_weight(f, X_tr, y_tr, sw_tr, **fit_extras)
        dist_train = _predict_with_extras(
            f, X_tr, ids_tr, ts_tr,
            deps_oof=deps_for_fit if deps_for_fit else None,
            groups=g_tr,
        )
        dist_test = _predict_with_extras(
            f, X_te, ids_te, ts_te,
            deps_oof=deps_for_pred if deps_for_pred else None,
            groups=g_te,
        )
        return dist_train, dist_test


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()[:16]


def _to_dt(x: Any) -> datetime:
    if isinstance(x, datetime):
        return x
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[s]").astype(datetime)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return datetime.fromtimestamp(float(x)) if float(x) > 1e6 else datetime(2024, 1, 1)
    return datetime(2024, 1, 1)


def _fit_with_optional_weight(
    forecaster: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    **extra: Any,
) -> None:
    """Call ``forecaster.fit`` with the kwargs the signature actually accepts.

    Drops ``sample_weight`` if not supported (online-learning trainers like
    ``OnlineAggregator`` and pure-sequence trainers like ``RNNHourly``).
    Also drops other extras (``ids``, ``timestamps``, ``deps_oof``) that
    a particular trainer doesn't declare — keeps callers free to pass the
    full row-alignment context without worrying about each trainer's API.

    Detection is signature-based, not TypeError-based, so a missing kwarg
    doesn't mask an unrelated bug.
    """
    import inspect

    try:
        sig = inspect.signature(forecaster.fit)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    def _accepts(name: str) -> bool:
        return accepts_var_kw or name in params

    call_kwargs: dict[str, Any] = {}
    if sample_weight is not None and _accepts("sample_weight"):
        call_kwargs["sample_weight"] = sample_weight
    for k, v in extra.items():
        if _accepts(k):
            call_kwargs[k] = v
    forecaster.fit(X, y, **call_kwargs)


def _predict_with_extras(
    forecaster: Any,
    X: np.ndarray,
    ids: np.ndarray,
    ts: np.ndarray,
    **extra: Any,
) -> DistributionForecast:
    """Call predict_dist threading any extras (``deps_oof``, ``groups``, …)
    that the forecaster's signature declares.

    Signature-based introspection — never a bare ``except TypeError``,
    which would swallow real bugs raised inside predict_dist.
    """
    import inspect

    try:
        sig = inspect.signature(forecaster.predict_dist)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    def _accepts(name: str) -> bool:
        return accepts_var_kw or name in params

    call_kwargs: dict[str, Any] = {"ids": ids, "timestamps": ts}
    for k, v in extra.items():
        if v is None:
            continue
        if _accepts(k):
            call_kwargs[k] = v
    return forecaster.predict_dist(X, **call_kwargs)


def _predict_with_deps(
    forecaster: Any,
    X: np.ndarray,
    ids: np.ndarray,
    ts: np.ndarray,
    deps_oof: dict[str, DistributionForecast],
) -> DistributionForecast:
    """Back-compat wrapper for the deps-only call path."""
    return _predict_with_extras(forecaster, X, ids, ts, deps_oof=deps_oof)


def _stitch_folds(
    folds: list[tuple[np.ndarray, DistributionForecast]],
    *,
    timestamps: np.ndarray,
    provenance: ProvenanceMeta,
) -> DistributionForecast:
    """Concatenate per-fold OOF dists into one whole-data OOF dist.

    All folds must be the same DistributionForecast subclass. Per-subclass
    concat logic lives in ``cls.stitch``. Output ids are the original row
    indices so ``y[ids]`` recovers the realized targets for OOF scoring.
    """
    if not folds:
        raise RuntimeError("no folds to stitch — pipeline emitted nothing")
    types = {type(d) for _, d in folds}
    if len(types) > 1:
        raise ValueError(
            f"mixed dist subclasses across folds: {types}. Pipeline folds "
            f"must share one subclass — a single forecaster cannot emit "
            f"different distribution types on different folds."
        )
    cls = next(iter(types))
    return cls.stitch(folds, timestamps=timestamps, provenance=provenance)


def _set_provenance(dist: DistributionForecast, prov: ProvenanceMeta) -> DistributionForecast:
    """Return a copy of ``dist`` with provenance replaced.

    Subclass-agnostic: ``dataclasses.replace`` copies all dataclass fields
    (which differ per concrete subclass) and overrides only ``provenance``.
    """
    import dataclasses
    return dataclasses.replace(dist, provenance=prov)


# ---------------------------------------------------------------------------
# Pipeline — flat, sequential chain of stages (= sklearn `Pipeline`).
#
# A *stage* is one of: Transformer, PointForecaster, Lifter, Calibrator,
# DistForecaster. The chain is wired left→right by stage kind into a single
# DistForecaster; a leading Transformer standardizes X (+ target at fit) and
# its `inverse_dist` maps the forecaster's distribution back to the original
# scale at the tail — so downstream bracket integration is unchanged.
#
# Track 1 supports the shape the weather fleet needs: [Transformer*,
# DistForecaster]. Point→Lifter and Calibrator stages need out-of-fold
# predictions threaded by the `WalkForward` driver and are deferred to
# Track 2 (raised loud here, not silently ignored).
# ---------------------------------------------------------------------------


def _stage_kind(stage) -> str:
    # Duck-typed (robust to data-attribute protocols): a Transformer carries
    # transform_target + inverse_dist; a DistForecaster carries predict_dist;
    # a PointForecaster carries predict (and no predict_dist); a Lifter lift;
    # a Calibrator transforms a dist (transform, no predict[_dist]).
    if hasattr(stage, "transform_target") and hasattr(stage, "inverse_dist"):
        return "transformer"
    if hasattr(stage, "predict_dist"):
        return "dist"
    if hasattr(stage, "lift"):
        return "lifter"
    if hasattr(stage, "predict"):
        return "point"
    if hasattr(stage, "transform"):
        return "calibrator"
    raise TypeError(
        f"Pipeline: stage {type(stage).__name__!r} matches no known stage "
        f"protocol (Transformer / PointForecaster / Lifter / Calibrator / "
        f"DistForecaster)"
    )


class Pipeline:
    """Sequential chain of stages, exposed as a `DistForecaster`.

    A *stage* is one of: `Transformer`, `PointForecaster`, `Lifter`,
    `Calibrator`, `DistForecaster`. The chain is wired left→right into a single
    distribution forecaster; the valid shapes are::

        [Transformer*, DistForecaster, Calibrator?]
        [Transformer*, PointForecaster, Lifter, Calibrator?]

    Examples::

        Pipeline([GroupByZScore(...), EMOS()])                 # normalize → dist
        Pipeline([EMOS()])                                     # ≡ bare EMOS
        Pipeline([SklearnPoint(Ridge()), GlobalResidual()])    # point → lift → dist
        Pipeline([EMOS(), Isotonic()])                         # dist → calibrate

    This **absorbs** the old `LiftedForecaster` / `CalibratedForecaster`
    wrapper classes: a Point→Lifter pair is fit with an internal out-of-fold
    half-split (the point fits on the first part, predicts the rest, the lifter
    fits on those OOF predictions, the point refits on full); a trailing
    Calibrator fits on a held-out tail of the (transformed) training data. The
    chain is self-contained — given ``(X, y, ids, timestamps)`` it fits itself,
    including the inner splits its stages need, so `WalkForward` only owns the
    *outer* CV.

    ``name`` is an optional leaderboard label (auto-derived otherwise).
    """

    def __init__(self, stages, *, name=None,
                 calibration_fraction=0.2, lifter_oof_fraction=0.5):
        stages = list(stages)
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        self.stages = stages
        self.calibration_fraction = calibration_fraction
        self.lifter_oof_fraction = lifter_oof_fraction
        self._transformers: list = []
        self._point = None        # PointForecaster
        self._lifter = None       # Lifter (requires a preceding point)
        self._model = None        # DistForecaster
        self._calibrator = None   # Calibrator (requires a preceding core)
        seen_core = False
        for st in stages:
            kind = _stage_kind(st)
            if kind == "transformer":
                if seen_core:
                    raise ValueError(
                        "Pipeline: Transformer stages must precede the forecaster"
                    )
                self._transformers.append(st)
            elif kind == "point":
                if seen_core:
                    raise ValueError(
                        "Pipeline: only one core forecaster; got a PointForecaster "
                        "after the core"
                    )
                self._point = st
                seen_core = True
            elif kind == "lifter":
                if self._point is None:
                    raise ValueError("Pipeline: a Lifter must follow a PointForecaster")
                if self._lifter is not None:
                    raise ValueError("Pipeline: at most one Lifter")
                self._lifter = st
            elif kind == "dist":
                if seen_core:
                    raise ValueError("Pipeline: only one core forecaster stage")
                self._model = st
                seen_core = True
            elif kind == "calibrator":
                if not seen_core:
                    raise ValueError("Pipeline: a Calibrator must follow the forecaster")
                if self._calibrator is not None:
                    raise ValueError("Pipeline: at most one Calibrator")
                self._calibrator = st
            else:  # pragma: no cover — _stage_kind already raised
                raise TypeError(f"Pipeline: unsupported stage kind {kind!r}")
        if self._point is not None and self._lifter is None:
            raise ValueError(
                "Pipeline: a PointForecaster needs a following Lifter to become "
                "a distribution"
            )
        if self._model is None and self._point is None:
            raise ValueError("Pipeline needs a forecaster stage")
        self.name = name or "->".join(
            getattr(s, "name", type(s).__name__) for s in stages
        )
        core = self._model if self._model is not None else self._point
        self.depends_on = tuple(getattr(core, "depends_on", ()))

    # ---- fit ----

    def fit(self, X, y, *, ids, timestamps=None, center=None,
            sample_weight=None, deps_oof=None, upstream=None, **kwargs):
        Xz = np.asarray(X, dtype=float)
        yz = np.asarray(y, dtype=float)
        n = yz.shape[0]
        ids_arr = np.asarray(ids)
        ts = np.zeros(n) if timestamps is None else np.asarray(timestamps)
        for t in self._transformers:
            t.fit(Xz, yz, ids=ids_arr, center=center)
            Xz = t.transform(Xz, ids=ids_arr, center=center)
            yz = t.transform_target(yz)

        if self._point is not None:
            # Point→Lifter with an internal OOF half-split (was LiftedForecaster).
            half = max(1, int(n * self.lifter_oof_fraction))
            if half >= n:
                half = max(1, n - 1)
            sw_first = sample_weight[:half] if sample_weight is not None else None
            _fit_with_optional_weight(self._point, Xz[:half], yz[:half], sw_first)
            base_oof = self._point.predict(
                Xz[half:], ids=ids_arr[half:], timestamps=ts[half:],
            )
            if self._lifter.requires_X:
                self._lifter.fit(base_oof, yz[half:], X=Xz[half:])
            else:
                self._lifter.fit(base_oof, yz[half:])
            _fit_with_optional_weight(self._point, Xz, yz, sample_weight)
        else:
            extras: dict[str, Any] = {"ids": ids_arr, "timestamps": ts}
            if deps_oof is not None:
                extras["deps_oof"] = deps_oof
            if upstream is not None:
                extras["upstream"] = upstream
            extras.update(kwargs)
            _fit_with_optional_weight(self._model, Xz, yz, sample_weight, **extras)

        if self._calibrator is not None:
            if deps_oof is not None or upstream is not None:
                raise NotImplementedError(
                    "Pipeline: a Calibrator combined with a deps-consuming "
                    "forecaster is not supported (would require slicing upstream "
                    "OOF onto the calibration tail)"
                )
            c = max(2, int(n * self.calibration_fraction))
            if n - c >= 2:
                cal_dist = self._core_predict_dist(
                    Xz[-c:], ids_arr[-c:], ts[-c:],
                )
                self._calibrator.fit(cal_dist, yz[-c:])
            else:
                self._calibrator = None   # too few rows to calibrate
        return self

    # ---- predict ----

    def _core_predict_dist(self, Xz, ids, ts, deps_oof=None, upstream=None):
        """The core forecaster's dist in the model's working (z) space —
        before calibration and before the transformers' inverse."""
        if self._point is not None:
            pt = self._point.predict(Xz, ids=ids, timestamps=ts)
            return self._lifter.lift(pt)
        extras: dict[str, Any] = {}
        if deps_oof is not None:
            extras["deps_oof"] = deps_oof
        if upstream is not None:
            extras["upstream"] = upstream
        return _predict_with_extras(self._model, Xz, ids, ts, **extras)

    def predict_dist(self, X, *, ids, timestamps, center=None,
                     deps_oof=None, upstream=None):
        Xz = np.asarray(X, dtype=float)
        ids_arr = np.asarray(ids)
        ts = np.asarray(timestamps)
        for t in self._transformers:
            Xz = t.transform(Xz, ids=ids_arr, center=center)   # stamps test (c, s)
        dist = self._core_predict_dist(Xz, ids_arr, ts, deps_oof, upstream)
        if self._calibrator is not None and getattr(self._calibrator, "fitted_", True):
            dist = self._calibrator.transform(dist)
        for t in reversed(self._transformers):
            dist = t.inverse_dist(dist)                     # z-space → original
        return dist
