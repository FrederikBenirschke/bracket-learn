"""ForecastPipeline — orchestration (CV + OOF stitching + DAG injection).

sklearn-style API::

    pipeline = ForecastPipeline(
        steps=[
            ("ridge", LiftedForecaster(SklearnPoint(Ridge()), GlobalResidual())),
            ("emos",  CalibratedForecaster(EMOS(), Isotonic())),
            ("stack", Stacking(deps=("ridge", "emos"))),
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

from bracketlearn.base import clone
from bracketlearn.forecast import (
    DistributionForecast,
    ProvenanceMeta,
)
from bracketlearn.protocols import (
    Calibrator,
)


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

        from bracketlearn.forecast import Backing, ParametricFamily

        for name, dist in self.forecasts.items():
            y_oof = y[dist.ids.astype(int)]
            row: dict[str, float] = {"n_oof": int(dist.ids.shape[0])}
            for m in metrics:
                if m == "crps":
                    if dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.NORMAL:
                        row["crps"] = float(scoremod.crps_gaussian(dist, y_oof).mean())
                    elif dist.backing == Backing.BRACKET:
                        row["crps"] = float(scoremod.crps_bracket(dist, y_oof).mean())
                    elif dist.backing == Backing.QUANTILE:
                        row["crps"] = float(scoremod.crps_quantile(dist, y_oof).mean())
                    else:
                        # Mixture: no closed-form CRPS; skip.
                        row["crps"] = float("nan")
                elif m == "log_score":
                    if dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.NORMAL:
                        row["log_score"] = float(scoremod.log_score_gaussian(dist, y_oof).mean())
                    elif dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.MIXTURE_NORMAL:
                        row["log_score"] = float(scoremod.log_score_mixture_normal(dist, y_oof).mean())
                    elif dist.backing == Backing.BRACKET:
                        row["log_score"] = float(scoremod.log_score_bracket(dist, y_oof).mean())
                    else:
                        row["log_score"] = float("nan")
                elif m in ("pit", "pit_mean"):
                    pits = scoremod.pit(dist, y_oof)
                    row["pit_mean"] = float(pits.mean())
                    row["pit_std"] = float(pits.std())
                elif m == "pit_std":
                    pits = scoremod.pit(dist, y_oof)
                    row["pit_std"] = float(pits.std())
                elif m == "log_loss_bracket":
                    contracts = ladder.price(dist)
                    row["log_loss_bracket"] = scoremod.log_loss_bracket(
                        contracts, ladder.edges, y_oof,
                    )
                elif m == "brier_bracket":
                    contracts = ladder.price(dist)
                    row["brier_bracket"] = scoremod.brier_bracket(
                        contracts, ladder.edges, y_oof,
                    )
                else:
                    raise ValueError(f"unknown metric: {m!r}")
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

        from bracketlearn.composite import LiftedForecaster, CalibratedForecaster
        from bracketlearn.lift import GlobalResidual, Isotonic
        from bracketlearn.trainers import SklearnPoint, EMOS, Stacking
        from sklearn.linear_model import Ridge

        p = ForecastPipeline(
            steps=[
                ("ridge", LiftedForecaster(SklearnPoint(Ridge()), GlobalResidual())),
                ("emos",  CalibratedForecaster(EMOS(), Isotonic())),
                ("stack", Stacking(deps=("ridge", "emos"))),
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
                    prov=prov, sample_weight=sw_o,
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
            self._fit_canonical_models(Xo, yo, ids_o, ts_o, sw_o)

        return PipelineResult(forecasts=out)

    def _fit_canonical_models(
        self,
        X: np.ndarray, y: np.ndarray, ids: np.ndarray, ts: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit each stage on the *full* training data, storing the result on
        ``self._fitted_stages`` for later use by ``predict()``.

        Calibrators are fit on a held-out tail of the full training data,
        matching the per-fold calibration logic. Downstream stages with
        ``depends_on`` receive the upstream stage's in-sample dist on the
        full training data.
        """
        from bracketlearn.composite import CalibratedForecaster

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
                        sample_weight=sample_weight,
                    )
                    calibrator.fit(cal_dist, y[calib_idx])
                else:
                    calibrator = None

            # Refit inner on the full train and record its in-sample dist for
            # downstream deps. Self-predict on the same N rows so the deps
            # row-alignment invariant (deps_oof[name].params['mu'].shape[0] == N)
            # holds for whatever downstream Stacking expects.
            dist_train_full = self._refit_and_predict_full(
                inner, X, y, ids, ts, deps, sample_weight=sample_weight,
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
    ) -> DistributionForecast:
        """Refit ``f`` on full (X, y); return its in-sample predict_dist."""
        from bracketlearn.composite import LiftedForecaster

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
        if deps:
            _fit_with_optional_weight(f, X, y, sample_weight, deps_oof=deps)
            return _predict_with_deps(f, X, ids, ts, deps)
        _fit_with_optional_weight(f, X, y, sample_weight)
        return f.predict_dist(X, ids=ids, timestamps=ts)

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, DistributionForecast]:
        """Predict on unseen X using the canonical (full-train) models.

        Returns ``{stage_name: DistributionForecast}``. Requires
        ``refit_on_full=True`` at construction (the default) and a prior
        ``fit_predict()`` call.
        """
        from bracketlearn.composite import LiftedForecaster

        if not self._fitted_stages:
            raise RuntimeError(
                "predict() requires a prior fit_predict() with refit_on_full=True"
            )
        X = np.asarray(X)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        out: dict[str, DistributionForecast] = {}
        for stage in self._stages:
            f = self._fitted_stages[stage.name]
            deps = {d: out[d] for d in stage.depends_on}
            if isinstance(f, LiftedForecaster):
                dist = f.predict_dist(X, ids=ids, timestamps=timestamps)
            elif deps:
                dist = _predict_with_deps(f, X, ids, timestamps, deps)
            else:
                dist = f.predict_dist(X, ids=ids, timestamps=timestamps)
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
            test_end = min(N, test_start + chunk_size)
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
    ) -> tuple[DistributionForecast, DistributionForecast]:
        """Fit ``f`` on train_idx; return (dist_on_train, dist_on_test).

        ``f`` should already be a clone — the pipeline never mutates the
        user-supplied forecaster instance.
        """
        from bracketlearn.composite import CalibratedForecaster

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
                sample_weight=sample_weight,
            )
        else:
            # 1) fit on train minus calibration tail
            cal_train_dist, cal_dist = self._fit_inner(
                inner, X, y, ids, ts, tr_minus, calib_idx,
                deps_for_fit, deps_for_pred,    # deps approximation OK in v0.1
                sample_weight=sample_weight,
            )
            calibrator.fit(cal_dist, y[calib_idx])
            # 2) refit on full train for canonical predictions
            dist_train, dist_test = self._fit_inner(
                inner, X, y, ids, ts, train_idx, test_idx,
                deps_for_fit, deps_for_pred,
                sample_weight=sample_weight,
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
    ) -> tuple[DistributionForecast, DistributionForecast]:
        from bracketlearn.composite import LiftedForecaster

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        ids_tr, ts_tr = ids[train_idx], ts[train_idx]
        ids_te, ts_te = ids[test_idx], ts[test_idx]
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None

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

        if deps_for_fit:
            _fit_with_optional_weight(f, X_tr, y_tr, sw_tr, deps_oof=deps_for_fit)
            dist_train = _predict_with_deps(f, X_tr, ids_tr, ts_tr, deps_for_fit)
            dist_test = _predict_with_deps(f, X_te, ids_te, ts_te, deps_for_pred)
        else:
            _fit_with_optional_weight(f, X_tr, y_tr, sw_tr)
            dist_train = f.predict_dist(X_tr, ids=ids_tr, timestamps=ts_tr)
            dist_test = f.predict_dist(X_te, ids=ids_te, timestamps=ts_te)
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
    """Call ``forecaster.fit`` with ``sample_weight`` if the signature accepts
    it; otherwise drop it silently. Extra kwargs (e.g. ``deps_oof``) pass
    through verbatim.

    Why: trainers that genuinely support weights take ``sample_weight=`` as
    a keyword; those that don't (online-learning trainers like
    OnlineAggregator, or pure-sequence trainers like RNNHourly) shouldn't
    crash a pipeline that happens to thread weights through. The detection
    is signature-based, not TypeError-based, so a missing kwarg doesn't
    mask an unrelated bug in the trainer.
    """
    import inspect

    if sample_weight is None:
        forecaster.fit(X, y, **extra)
        return
    try:
        sig = inspect.signature(forecaster.fit)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}
    accepts_sw = (
        "sample_weight" in params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    )
    if accepts_sw:
        forecaster.fit(X, y, sample_weight=sample_weight, **extra)
    else:
        forecaster.fit(X, y, **extra)


def _predict_with_deps(
    forecaster: Any,
    X: np.ndarray,
    ids: np.ndarray,
    ts: np.ndarray,
    deps_oof: dict[str, DistributionForecast],
) -> DistributionForecast:
    try:
        return forecaster.predict_dist(X, ids=ids, timestamps=ts, deps_oof=deps_oof)
    except TypeError:
        return forecaster.predict_dist(X, ids=ids, timestamps=ts)


def _stitch_folds(
    folds: list[tuple[np.ndarray, DistributionForecast]],
    *,
    timestamps: np.ndarray,
    provenance: ProvenanceMeta,
) -> DistributionForecast:
    """Concatenate per-fold OOF dists into one whole-data OOF dist.

    All folds must share backing/family (and edges for bracket, K for mixture).
    Output ids are the original row indices so y[ids] recovers the realized
    targets for OOF scoring.
    """
    from bracketlearn.forecast import Backing, ParametricFamily

    if not folds:
        raise RuntimeError("no folds to stitch — pipeline emitted nothing")
    backings = {d.backing for _, d in folds}
    if len(backings) > 1:
        raise NotImplementedError(f"mixed backings across folds: {backings}")
    backing = next(iter(backings))

    all_rows = np.concatenate([rows for rows, _ in folds])
    all_ts = timestamps[all_rows]
    # Sort by original row index so downstream y[ids] aligns trivially.
    order = np.argsort(all_rows, kind="stable")
    ids_sorted = all_rows[order]
    ts_sorted = all_ts[order]

    if backing == Backing.PARAMETRIC:
        families = {d.family for _, d in folds}
        if len(families) > 1:
            raise NotImplementedError(f"mixed parametric families: {families}")
        family = next(iter(families))
        if family == ParametricFamily.NORMAL:
            mu = np.concatenate([d.params["mu"] for _, d in folds])[order]
            sigma = np.concatenate([d.params["sigma"] for _, d in folds])[order]
            return DistributionForecast.from_normal(
                mu, sigma, ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
            )
        if family == ParametricFamily.MIXTURE_NORMAL:
            weights = np.concatenate([d.params["weights"] for _, d in folds], axis=0)[order]
            mus = np.concatenate([d.params["mus"] for _, d in folds], axis=0)[order]
            sigmas = np.concatenate([d.params["sigmas"] for _, d in folds], axis=0)[order]
            return DistributionForecast.from_mixture_normal(
                weights=weights, mus=mus, sigmas=sigmas,
                ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
            )
        raise NotImplementedError(f"stitching not implemented for parametric family {family}")

    if backing == Backing.BRACKET:
        edges_set = {tuple(d.edges.tolist()) for _, d in folds}
        if len(edges_set) > 1:
            raise NotImplementedError("bracket folds with different edges cannot be stitched")
        edges = folds[0][1].edges
        probs = np.concatenate([d.probs for _, d in folds], axis=0)[order]
        return DistributionForecast.from_brackets(
            edges=edges, probs=probs,
            ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
        )

    if backing == Backing.QUANTILE:
        taus_set = {tuple(d.taus.tolist()) for _, d in folds}
        if len(taus_set) > 1:
            raise NotImplementedError("quantile folds with different taus cannot be stitched")
        taus = folds[0][1].taus
        qvals = np.concatenate([d.qvals for _, d in folds], axis=0)[order]
        # Tail policy must agree across folds (all folds in a pipeline come
        # from the same trainer); pick from first fold.
        tail_policy = folds[0][1].tail_policy
        return DistributionForecast.from_quantiles(
            taus=taus, qvals=qvals, tail_policy=tail_policy,
            ids=ids_sorted, timestamps=ts_sorted, provenance=provenance,
        )

    raise NotImplementedError(f"stitching not implemented for backing {backing}")


def _set_provenance(dist: DistributionForecast, prov: ProvenanceMeta) -> DistributionForecast:
    return DistributionForecast(
        backing=dist.backing,
        family=dist.family,
        params=dist.params,
        taus=dist.taus,
        qvals=dist.qvals,
        members=dist.members,
        edges=dist.edges,
        probs=dist.probs,
        ids=dist.ids,
        timestamps=dist.timestamps,
        provenance=prov,
        tail_policy=dist.tail_policy,
        tail_support=dist.tail_support,
    )
