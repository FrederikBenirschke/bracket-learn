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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bracketlearn.base import BaseEstimator
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
        """DEPRECATED surface — delegates to ``WalkForward`` over the object
        graph built from ``steps``. Kept so existing callers stay green; prefer
        ``WalkForward(...).fit_predict(Pipeline/Stacker, ...)`` directly."""
        from bracketlearn.compose import WalkForward

        roots = self._build_graph()
        self._wf = WalkForward(
            cv=self.cv, n_folds=self.n_folds, embargo=self.embargo,
            refit_on_full=self.refit_on_full, shuffle=self.shuffle,
            random_state=self.random_state, rolling_window=self.rolling_window,
        )
        return self._wf.fit_predict(
            roots, X, y, ids=ids, timestamps=timestamps,
            sample_weight=sample_weight, groups=groups,
        )

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> dict[str, DistributionForecast]:
        """Predict on unseen rows via the canonical (full-train) models.
        Requires ``refit_on_full=True`` (default) and a prior ``fit_predict``."""
        if getattr(self, "_wf", None) is None:
            raise RuntimeError(
                "predict() requires a prior fit_predict() with refit_on_full=True"
            )
        return self._wf.predict(X, ids=ids, timestamps=timestamps, groups=groups)

    # ---- object-graph translation: named steps -> Pipeline / Stacker ----

    def _build_graph(self) -> list:
        """Translate ``self._stages`` (named, depends_on strings) into the
        object graph WalkForward runs. Object identity is preserved so a stage
        referenced as a dep is the SAME object as its standalone output (and is
        therefore computed once)."""
        from bracketlearn.compose import Stacker

        node_by_name: dict[str, Any] = {}
        roots: list = []
        for stage in self._stages:
            if stage.depends_on:
                ups = [node_by_name[d] for d in stage.depends_on]
                node = Stacker(ups, stage.forecaster, name=stage.name)
            else:
                node = self._wrap_stage(stage.name, stage.forecaster)
            node_by_name[stage.name] = node
            roots.append(node)
        return roots

    def _wrap_stage(self, name: str, f: Any):
        """A non-meta stage -> a Pipeline. Unwrap the legacy LiftedForecaster /
        CalibratedForecaster wrappers into plain Pipeline stages."""
        calibrator = None
        inner = f
        if isinstance(f, CalibratedForecaster):
            calibrator = f.calibrator
            inner = f.forecaster
        if isinstance(inner, LiftedForecaster):
            stages = [inner.base, inner.lifter]
        else:
            stages = [inner]
        if calibrator is not None:
            stages = stages + [calibrator]
        return Pipeline(
            stages, name=name, calibration_fraction=self.calibration_fraction,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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

    def _core_predict_dist(self, Xz, ids, ts, deps_oof=None, upstream=None,
                           groups=None):
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
        if groups is not None:
            extras["groups"] = groups
        return _predict_with_extras(self._model, Xz, ids, ts, **extras)

    def predict_dist(self, X, *, ids, timestamps, center=None,
                     deps_oof=None, upstream=None, groups=None):
        Xz = np.asarray(X, dtype=float)
        ids_arr = np.asarray(ids)
        ts = np.asarray(timestamps)
        for t in self._transformers:
            Xz = t.transform(Xz, ids=ids_arr, center=center)   # stamps test (c, s)
        dist = self._core_predict_dist(Xz, ids_arr, ts, deps_oof, upstream, groups)
        if self._calibrator is not None and getattr(self._calibrator, "fitted_", True):
            dist = self._calibrator.transform(dist)
        for t in reversed(self._transformers):
            dist = t.inverse_dist(dist)                     # z-space → original
        return dist
