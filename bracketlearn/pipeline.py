"""Pipeline, the sequential chain forecaster, plus PipelineResult + the
shared fold helpers used by the WalkForward CV driver.

`Pipeline([...stages...])` wires a left→right chain of stages (Transformer*,
then one core PointForecaster+Lifter or DistForecaster, then an optional
Calibrator) into a single `DistForecaster`. It absorbs the lifter half-split
and calibrator tail-fit internally, so the `WalkForward` driver
(``bracketlearn.compose``) owns only the outer CV. Compose a parallel ensemble
with `Stacker`; run either under `WalkForward(...).fit_predict(model, ...)`.

`PipelineResult` (returned by `WalkForward`) maps each node name → its stitched
out-of-fold `DistributionForecast` and owns OOF-aligned scoring, so the caller
never touches ``dist.ids``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bracketlearn.forecast import (
    DistributionForecast,
    ProvenanceMeta,
)

# ---------------------------------------------------------------------------
# Metric dispatch registry. Used by PipelineResult.score (and downstream
# leaderboard helpers) to turn a (metric_name, distribution) pair into a
# scalar mean. The registry is keyed by metric; per-backing dispatch lives
# inside each entry so adding a new backing means touching one table, not
# four if/elif blocks (audit item §3.S3).
# ---------------------------------------------------------------------------


def _slice_edges(edges: Any, idx: np.ndarray) -> Any:
    """Take the rows ``idx`` out of a ladder, whatever shape it is.

    A shared 1-D ``(B+1,)`` vector describes every row, so it passes through;
    a dense ``(N, B+1)`` grid or a ragged per-row sequence is indexed.
    """
    if edges is None:
        return None
    if isinstance(edges, np.ndarray) and edges.ndim == 1:
        return edges
    if isinstance(edges, np.ndarray) and edges.ndim == 2:
        return edges[idx]
    seq = list(edges)
    if seq and np.ndim(seq[0]) == 0:      # a 1-D ladder handed in as a list
        return np.asarray(seq, dtype=float)
    return [seq[i] for i in idx]


def _compute_metric(
    metric: str, dist, y, *, edges, scoremod,
) -> dict[str, float]:
    """Dispatch one (metric, distribution) pair to a scalar value, or to
    a small dict (PIT contributes both mean and std).

    ``edges`` is the bracket ladder for the bracket metrics (None otherwise),
    in any of the three shapes ``integrate`` accepts: a shared 1-D ``(B+1,)``
    vector, a dense ``(N, B+1)`` grid, or a ragged per-row sequence. A shared
    vector is broadcast to every row; per-row grids are passed through, which
    is what a rotating venue ladder (Kalshi relists daily) needs.
    """
    if metric == "crps":
        return {"crps": float(dist.crps(y).mean())}
    if metric == "log_score":
        return {"log_score": float(dist.log_score(y).mean())}
    if metric in ("pit", "pit_mean", "pit_std"):
        pits = dist.pit(y)
        return {"pit_mean": float(pits.mean()), "pit_std": float(pits.std())}
    if metric in ("log_loss_bracket", "brier_bracket"):
        from bracketlearn.adapters import BracketLadder

        n_rows = dist.ids.shape[0]
        if isinstance(edges, np.ndarray) and edges.ndim == 1:
            per_row = [edges.astype(float)] * n_rows
        elif isinstance(edges, np.ndarray) and edges.ndim == 2:
            per_row = [np.asarray(e, float) for e in edges]
        else:
            seq = list(edges)
            per_row = (
                [np.asarray(seq, float)] * n_rows
                if seq and np.ndim(seq[0]) == 0
                else [np.asarray(e, float) for e in seq]
            )
        if len(per_row) != n_rows:
            raise ValueError(
                f"edges describe {len(per_row)} rows; the forecast has "
                f"{n_rows}")
        ladder = BracketLadder(edges_per_row=per_row)
        contracts = ladder.price(dist)
        fn = getattr(scoremod, metric)
        return {metric: fn(contracts, per_row, y)}
    raise ValueError(f"unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# PipelineResult, owns OOF coverage alignment + scoring.
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
        result.score(y, metrics=["log_loss_bracket", "brier_bracket"], edges=edges)
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
        edges: Any | None = None,
    ) -> dict[str, dict[str, float]]:
        """Return {stage_name: {metric_name: value}}.

        Available metrics:
          - "crps"             - mean CRPS for Gaussian backing
          - "log_score"        - mean predictive negative log-likelihood
          - "pit_mean"         - mean PIT (≈ 0.5 if calibrated)
          - "pit_std"          - std of PIT
          - "log_loss_bracket", requires ``edges`` (any ladder shape)
          - "brier_bracket", requires ``edges`` (any ladder shape)

        ``edges`` takes any of the three shapes the scorers accept: a shared
        1-D ``(B+1,)`` vector, a dense ``(N, B+1)`` grid, or a ragged per-row
        sequence. A rotating ladder needs one of the latter two, and passing
        a single vector for one is refused rather than scored against row 0's
        grid. It is sliced alongside ``y`` for stages whose out-of-fold
        coverage is a subset of the rows.

        y is the full original target vector; PipelineResult slices it to
        match each stage's OOF coverage via dist.ids.
        """
        from bracketlearn import score as scoremod

        y = np.asarray(y, dtype=float)
        out: dict[str, dict[str, float]] = {}

        needs_edges = {"log_loss_bracket", "brier_bracket"}
        if needs_edges & set(metrics) and edges is None:
            raise ValueError(
                f"metrics {needs_edges & set(metrics)} require edges=... "
                f"(a shared (B+1,) vector, a dense (N, B+1) grid, or a ragged "
                f"per-row sequence)"
            )

        for name, dist in self.forecasts.items():
            idx = dist.ids.astype(int)
            y_oof = y[idx]
            # `edges` must be sliced alongside `y`: a stage's OOF coverage can
            # be a subset of the rows, and a per-row ladder is indexed by row.
            # Passing the full-length ladder against a sliced y raised "edges
            # describe N rows; the forecast has M", which made the per-row
            # shape unusable through exactly this API.
            edges_oof = _slice_edges(edges, idx)
            row: dict[str, float] = {"n_oof": int(dist.ids.shape[0])}
            for m in metrics:
                row.update(
                    _compute_metric(m, dist, y_oof, edges=edges_oof,
                                    scoremod=scoremod))
            out[name] = row
        return out

    def to_table(
        self,
        y: np.ndarray,
        *,
        metrics: Sequence[str] = ("crps", "log_score", "pit"),
        edges: Any | None = None,
    ) -> str:
        """Render score() output as an aligned text table."""
        scores = self.score(y, metrics=metrics, edges=edges)
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
    Also drops other extras (``ids``, ``timestamps``, ``groups``) that
    a particular trainer doesn't declare, keeps callers free to pass the
    full row-alignment context without worrying about each trainer's API.

    Detection is signature-based, not TypeError-based, so a missing kwarg
    doesn't mask an unrelated bug.
    """
    import inspect

    params: dict[str, inspect.Parameter] = {}
    try:
        params = dict(inspect.signature(forecaster.fit).parameters)
    except (TypeError, ValueError):
        pass
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
    """Call predict_dist threading any extras (``groups``, …)
    that the forecaster's signature declares.

    Signature-based introspection, never a bare ``except TypeError``,
    which would swallow real bugs raised inside predict_dist.
    """
    import inspect

    params: dict[str, inspect.Parameter] = {}
    try:
        params = dict(inspect.signature(forecaster.predict_dist).parameters)
    except (TypeError, ValueError):
        pass
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
        raise RuntimeError("no folds to stitch, pipeline emitted nothing")
    types = {type(d) for _, d in folds}
    if len(types) > 1:
        raise ValueError(
            f"mixed dist subclasses across folds: {types}. Pipeline folds "
            f"must share one subclass, a single forecaster cannot emit "
            f"different distribution types on different folds."
        )
    cls = next(iter(types))
    return cls.stitch(folds, timestamps=timestamps, provenance=provenance)


# ---------------------------------------------------------------------------
# Pipeline, flat, sequential chain of stages (= sklearn `Pipeline`).
#
# A *stage* is one of: Transformer, PointForecaster, Lifter, Calibrator,
# DistForecaster. The chain is wired left→right by stage kind into a single
# DistForecaster; a leading Transformer standardizes X (+ target at fit) and
# its `inverse_dist` maps the forecaster's distribution back to the original
# scale at the tail, so downstream bracket integration is unchanged.
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

    A Point→Lifter pair is fit with an internal out-of-fold
    half-split (the point fits on the first part, predicts the rest, the lifter
    fits on those OOF predictions, the point refits on full); a trailing
    Calibrator fits on a held-out tail of the (transformed) training data. The
    chain is self-contained, given ``(X, y, ids, timestamps)`` it fits itself,
    including the inner splits its stages need, so `WalkForward` only owns the
    *outer* CV.

    ``name`` is an optional leaderboard label (auto-derived otherwise).
    """

    def __init__(self, stages: Sequence[Any], *, name: str | None = None,
                 calibration_fraction: float = 0.2,
                 lifter_oof_fraction: float = 0.5) -> None:
        stages = list(stages)
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        self.stages = stages
        self.calibration_fraction = calibration_fraction
        self.lifter_oof_fraction = lifter_oof_fraction
        self._transformers: list[Any] = []
        self._point: Any | None = None       # PointForecaster
        self._lifter: Any | None = None      # Lifter (requires a preceding point)
        self._model: Any | None = None       # DistForecaster
        self._calibrator: Any | None = None  # Calibrator (requires a preceding core)
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
            else:  # pragma: no cover, _stage_kind already raised
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

    # ---- fit ----

    def fit(self, X: Any, y: Any, *, ids: Any, timestamps: Any = None,
            center: Any = None, sample_weight: Any = None,
            upstream: Any = None, **kwargs: Any) -> Pipeline:
        Xz = np.asarray(X, dtype=float)
        yz = np.asarray(y, dtype=float)
        n = yz.shape[0]
        ids_arr = np.asarray(ids)
        ts = np.zeros(n) if timestamps is None else np.asarray(timestamps)

        # How many trailing rows the calibrator will be fit on, decided BEFORE
        # anything is fit so both the transformers and the core can be held out
        # of them. `c == 0` means no calibrator, or too few rows to hold any
        # out.
        c = 0
        if self._calibrator is not None:
            if upstream is not None:
                raise NotImplementedError(
                    "Pipeline: a Calibrator combined with an upstream-consuming "
                    "forecaster is not supported (would require slicing upstream "
                    "OOF onto the calibration tail)"
                )
            c = max(2, int(n * self.calibration_fraction))
            if n - c < 2:
                self._calibrator = None   # too few rows to calibrate
                c = 0

        # Transformers are fit on the same head the core is, for the same
        # reason: GroupByZScore learns its scale from std(y - center), and
        # including the calibration tail lets the tail set the scale that the
        # calibrator's own inputs are then divided by. Measured on a synthetic
        # tail with a wider spread: scale 25.30 fit on all rows vs 0.96 fit on
        # the head. Smaller in effect than the core leak (one scalar per
        # group), but the same leak, in the same function.
        #
        # The rows the CORE may see while the calibrator's tail is being
        # produced. Fitting the core on everything and then calibrating on a
        # slice of that same everything is the leak this split exists to stop:
        # the calibrator would be learning a correction to in-sample
        # predictions, which are systematically better than the out-of-sample
        # ones it will actually be applied to, so it under-corrects in
        # production. The Point→Lifter branch below already had this shape;
        # the calibrator did not.
        head = slice(0, n - c) if c else slice(0, n)

        def _fit_transformers(rows: slice) -> tuple[np.ndarray, np.ndarray]:
            """Fit every transformer on ``rows``, then apply to ALL n rows.

            Fitting must not see the tail; transforming it is required, since the
            calibrator needs those rows in the model's working space.
            """
            Xw, yw = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
            for t in self._transformers:
                t.fit(Xw[rows], yw[rows], ids=ids_arr[rows], center=center)
                Xw = t.transform(Xw, ids=ids_arr, center=center)
                yw = t.transform_target(yw)
            return Xw, yw

        def _fit_core(rows: slice) -> None:
            """Fit the core forecaster on ``rows`` only."""
            if self._point is not None:
                # Point→Lifter with an internal OOF half-split, nested inside
                # whatever slice it is handed. __init__ refuses a
                # PointForecaster without a following Lifter, so _lifter is
                # bound whenever _point is.
                assert self._lifter is not None

                sub_n = rows.stop - rows.start
                half = max(1, int(sub_n * self.lifter_oof_fraction))
                if half >= sub_n:
                    half = max(1, sub_n - 1)
                a, b = rows.start, rows.start + half
                sw_first = sample_weight[a:b] if sample_weight is not None else None
                _fit_with_optional_weight(self._point, Xz[a:b], yz[a:b], sw_first)
                base_oof = self._point.predict(
                    Xz[b:rows.stop], ids=ids_arr[b:rows.stop],
                    timestamps=ts[b:rows.stop],
                )
                if self._lifter.requires_X:
                    self._lifter.fit(base_oof, yz[b:rows.stop], X=Xz[b:rows.stop])
                else:
                    self._lifter.fit(base_oof, yz[b:rows.stop])
                sw = sample_weight[rows] if sample_weight is not None else None
                _fit_with_optional_weight(self._point, Xz[rows], yz[rows], sw)
            else:
                extras: dict[str, Any] = {
                    "ids": ids_arr[rows], "timestamps": ts[rows],
                }
                if upstream is not None:
                    extras["upstream"] = upstream
                extras.update(kwargs)
                sw = sample_weight[rows] if sample_weight is not None else None
                _fit_with_optional_weight(self._model, Xz[rows], yz[rows], sw,
                                          **extras)

        if c:
            # 1. transformers + core on the head only, so the tail is unseen
            Xz, yz = _fit_transformers(head)
            _fit_core(head)
            # 2. calibrator on the core's OUT-OF-SAMPLE tail predictions
            cal_dist = self._core_predict_dist(
                Xz[-c:], ids_arr[-c:], ts[-c:], **kwargs,
            )
            assert self._calibrator is not None   # c > 0 implies one is set
            self._calibrator.fit(cal_dist, yz[-c:])
            # 3. refit the CORE on everything, for prediction. The
            # transformers are deliberately NOT refit: they define the z
            # space the calibrator just learned its correction in, and
            # Isotonic maps absolute z values, so it is not scale
            # invariant. Refitting them here moved the space underneath a
            # calibrator that is never refit, and the correction was then
            # applied at the wrong scale (measured: fit at std(mu)=8.90,
            # applied at std(mu)=0.99). Keeping the head-fit transformers
            # costs nothing, since _fit_transformers already transforms
            # all n rows and only its FIT is restricted to the head, which
            # is also what keeps the tail out of them.
            _fit_core(slice(0, n))
        else:
            Xz, yz = _fit_transformers(slice(0, n))
            _fit_core(slice(0, n))
        return self

    # ---- predict ----

    def _core_predict_dist(self, Xz: Any, ids: Any, ts: Any,
                           upstream: Any = None, groups: Any = None,
                           **kwargs: Any) -> Any:
        """The core forecaster's dist in the model's working (z) space,
        before calibration and before the transformers' inverse. ``kwargs`` are
        id-keyed side inputs (e.g. ``cutpoints_by_id`` / ``brackets_by_id``)
        forwarded verbatim; signature-filtered for the core forecaster."""
        if self._point is not None:
            assert self._lifter is not None   # paired at construction
            pt = self._point.predict(Xz, ids=ids, timestamps=ts)
            return self._lifter.lift(pt)
        extras: dict[str, Any] = {**kwargs}
        if upstream is not None:
            extras["upstream"] = upstream
        if groups is not None:
            extras["groups"] = groups
        return _predict_with_extras(self._model, Xz, ids, ts, **extras)

    def predict_dist(self, X: Any, *, ids: Any, timestamps: Any,
                     center: Any = None, upstream: Any = None,
                     groups: Any = None, **kwargs: Any) -> Any:
        Xz = np.asarray(X, dtype=float)
        ids_arr = np.asarray(ids)
        ts = np.asarray(timestamps)
        for t in self._transformers:
            Xz = t.transform(Xz, ids=ids_arr, center=center)   # stamps test (c, s)
        dist = self._core_predict_dist(Xz, ids_arr, ts, upstream, groups, **kwargs)
        if self._calibrator is not None and getattr(self._calibrator, "fitted_", True):
            dist = self._calibrator.transform(dist)
        for t in reversed(self._transformers):
            dist = t.inverse_dist(dist)                     # z-space → original
        return dist
