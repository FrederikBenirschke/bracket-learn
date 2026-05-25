"""ForecastPipeline — orchestration (CV + OOF stitching + DAG injection).

sklearn-style API:

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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from bracketlearn.forecast import (
    DistributionForecast,
    PointForecast,
    ProvenanceMeta,
)
from bracketlearn.protocols import (
    Calibrator,
    DistForecaster,
    Lifter,
    PointForecaster,
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

        for name, dist in self.forecasts.items():
            y_oof = y[dist.ids.astype(int)]
            row: dict[str, float] = {"n_oof": int(dist.ids.shape[0])}
            for m in metrics:
                if m == "crps":
                    row["crps"] = float(scoremod.crps_gaussian(dist, y_oof).mean())
                elif m == "log_score":
                    row["log_score"] = float(scoremod.log_score_gaussian(dist, y_oof).mean())
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

    Example:
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
    ):
        if cv != "expanding-window":
            raise NotImplementedError("v0.1: only 'expanding-window'")
        self.cv = cv
        self.n_folds = n_folds
        self.embargo = embargo
        self.calibration_fraction = calibration_fraction
        self._stages: list[_Stage] = []
        self._names: set[str] = set()
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
    ) -> PipelineResult:
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        N = y.shape[0]

        feature_hash = _hash_array(X)
        fit_window = (_to_dt(timestamps.min()), _to_dt(timestamps.max()))
        code_sha = "dev"

        order = np.argsort(timestamps, kind="stable")
        Xo = X[order]
        yo = y[order]
        ids_o = ids[order]
        ts_o = timestamps[order]

        folds = self._expanding_folds(N)

        oof_mu = {s.name: np.full(N, np.nan) for s in self._stages}
        oof_sigma = {s.name: np.full(N, np.nan) for s in self._stages}

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
                dist_train, dist_test = self._fit_stage_on_fold(
                    stage, Xo, yo, ids_o, ts_o, train_idx, test_idx,
                    deps_for_fit=deps_for_fit, deps_for_pred=deps_for_pred,
                    prov=prov,
                )
                fold_train_dist[stage.name] = dist_train
                fold_test_dist[stage.name] = dist_test
                if dist_test.backing.value != "parametric":
                    raise NotImplementedError(
                        f"v0.1 pipeline supports only parametric-normal OOF; got {dist_test.backing}"
                    )
                # Note: ids_o[test_idx] are the *original* row indices because
                # ids was constructed as np.arange in the demo. For real data,
                # the user's ids may not equal row indices — we record the
                # *sorted-row position* (test_idx itself) for OOF scoring,
                # by mapping back to original via the `order` permutation.
                orig_rows = order[test_idx]
                oof_mu[stage.name][orig_rows] = dist_test.params["mu"]
                oof_sigma[stage.name][orig_rows] = dist_test.params["sigma"]

        out: dict[str, DistributionForecast] = {}
        for stage in self._stages:
            mu = oof_mu[stage.name]
            sigma = oof_sigma[stage.name]
            valid = ~np.isnan(mu)
            if not valid.all():
                mu_v = mu[valid]
                sigma_v = sigma[valid]
                # ids of valid rows = the original row index, so y[ids] aligns.
                ids_out = np.where(valid)[0]
                ts_out = timestamps[valid]
            else:
                mu_v = mu
                sigma_v = sigma
                ids_out = np.arange(N)
                ts_out = timestamps
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
            out[stage.name] = DistributionForecast.from_normal(
                mu_v, sigma_v, ids=ids_out, timestamps=ts_out, provenance=prov,
            )
        return PipelineResult(forecasts=out)

    # ---------------------------------------------------------- fold helpers

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

    # ---------------------------------------------------------- stage fit

    def _fit_stage_on_fold(
        self,
        stage: _Stage,
        X: np.ndarray, y: np.ndarray, ids: np.ndarray, ts: np.ndarray,
        train_idx: np.ndarray, test_idx: np.ndarray,
        *,
        deps_for_fit: dict[str, DistributionForecast],
        deps_for_pred: dict[str, DistributionForecast],
        prov: ProvenanceMeta,
    ) -> tuple[DistributionForecast, DistributionForecast]:
        """Fit stage on train_idx; return (dist_on_train, dist_on_test)."""
        from bracketlearn.composite import CalibratedForecaster, LiftedForecaster

        f = stage.forecaster
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        ids_tr, ts_tr = ids[train_idx], ts[train_idx]
        ids_te, ts_te = ids[test_idx], ts[test_idx]

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
            )
        else:
            # 1) fit on train minus calibration tail
            cal_train_dist, cal_dist = self._fit_inner(
                inner, X, y, ids, ts, tr_minus, calib_idx,
                deps_for_fit, deps_for_pred,    # deps approximation OK in v0.1
            )
            calibrator.fit(cal_dist, y[calib_idx])
            # 2) refit on full train for canonical predictions
            dist_train, dist_test = self._fit_inner(
                inner, X, y, ids, ts, train_idx, test_idx,
                deps_for_fit, deps_for_pred,
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
    ) -> tuple[DistributionForecast, DistributionForecast]:
        from bracketlearn.composite import LiftedForecaster

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        ids_tr, ts_tr = ids[train_idx], ts[train_idx]
        ids_te, ts_te = ids[test_idx], ts[test_idx]

        if isinstance(f, LiftedForecaster):
            half = len(train_idx) // 2
            base_fit_idx = train_idx[:half]
            base_oof_idx = train_idx[half:]
            f.base.fit(X[base_fit_idx], y[base_fit_idx])
            base_pt_oof = f.base.predict(
                X[base_oof_idx], ids=ids[base_oof_idx], timestamps=ts[base_oof_idx],
            )
            f.base.fit(X_tr, y_tr)
            if f.lifter.requires_X:
                f.lifter.fit(base_pt_oof, y[base_oof_idx], X=X[base_oof_idx])
            else:
                f.lifter.fit(base_pt_oof, y[base_oof_idx])
            dist_train = f.predict_dist(X_tr, ids=ids_tr, timestamps=ts_tr)
            dist_test = f.predict_dist(X_te, ids=ids_te, timestamps=ts_te)
            return dist_train, dist_test

        if deps_for_fit:
            f.fit(X_tr, y_tr, deps_oof=deps_for_fit)
            dist_train = _predict_with_deps(f, X_tr, ids_tr, ts_tr, deps_for_fit)
            dist_test = _predict_with_deps(f, X_te, ids_te, ts_te, deps_for_pred)
        else:
            f.fit(X_tr, y_tr)
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
