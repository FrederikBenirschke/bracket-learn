"""The clean composition surface: ``Stacker`` (parallel combiner) +
``WalkForward`` (CV/OOF driver).

Three orthogonal concepts, object-nested, names only label the leaderboard::

    ridge = Pipeline([SklearnPoint(Ridge()), GlobalResidual()], name="ridge")
    emos  = Pipeline([EMOS()], name="emos")
    model = Stacker([ridge, emos], StackedParametric())
    result = WalkForward(n_folds=5).fit_predict(model, X, y, ids=ids, timestamps=ts)
    result["ridge"]            # upstream leaderboard rows, addressable

- ``Pipeline`` is the sequential chain (a self-contained `DistForecaster`).
- ``Stacker`` holds upstream *objects* and a meta-combiner; the dependency IS
  the nesting (no name-string ``deps``). Shared upstreams (same object) are
  computed once per fold; nested stackers recurse.
- ``WalkForward`` owns ONLY the outer expanding/rolling/kfold CV. Each node is
  cloned per fold, fit on the fold's train slice, and predicted on train+test;
  a meta receives its upstreams' fold dists **positionally** via ``upstream=``.

This is the homogeneous replacement for ``ForecastPipeline`` +
``LiftedForecaster`` + ``CalibratedForecaster`` + name-keyed ``deps_oof``. Those
remain (for now) as the legacy surface; this module is the new core.

Per Rule #0.5: a meta whose upstream is missing, or a fold that emits nothing,
raises loud rather than returning a partial result.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from bracketlearn.forecast import DistributionForecast  # noqa: F401  (type clarity)
from bracketlearn.forecast._meta import ProvenanceMeta
from bracketlearn.pipeline import (
    PipelineResult,
    _fit_with_optional_weight,
    _predict_with_extras,
    _stitch_folds,
)


# ---------------------------------------------------------------------------
# Stacker — parallel combiner over upstream model objects.
# ---------------------------------------------------------------------------


class Stacker:
    """Combine upstream models with a meta-combiner.

    ``Stacker([p1, p2], meta, name=...)`` — the upstreams are `Pipeline` /
    forecaster *objects* (not name strings); ``meta`` is a meta-combiner
    (`StackedParametric`, `BMAStacking`, `BracketStacking`, `DistAsFeatures`)
    that receives the upstreams' out-of-fold distributions positionally, in
    declared order, via ``upstream=[...]`` when run under `WalkForward`.

    ``Stacker`` is pure structure — it holds no fitted state and is not run
    directly; pass it to `WalkForward.fit_predict`.
    """

    def __init__(self, upstreams, meta, *, name=None):
        ups = list(upstreams)
        if not ups:
            raise ValueError("Stacker needs at least one upstream model")
        if meta is None:
            raise ValueError("Stacker needs a meta-combiner")
        self.upstreams = ups
        self.meta = meta
        self.name = name or getattr(meta, "name", type(meta).__name__)
        # A Stacker has no external feature deps; its dependency is its nesting.
        self.depends_on = ()


def _flatten(model) -> list[dict]:
    """Topo-flatten an object graph into ordered nodes (deps before dependents).

    ``model`` is a single `Pipeline`/`Stacker`/forecaster or a list of them
    (multiple independent leaderboard outputs). Each node:
    ``{obj, name, deps: list[int], is_meta: bool}``. Shared objects (same
    identity) collapse to one node, so a reused upstream is computed once.
    """
    nodes: list[dict] = []
    idx_by_id: dict[int, int] = {}

    def visit(node) -> int:
        if id(node) in idx_by_id:
            return idx_by_id[id(node)]
        if isinstance(node, Stacker):
            dep_idxs = [visit(u) for u in node.upstreams]
            i = len(nodes)
            nodes.append(
                {"obj": node.meta, "name": node.name, "deps": dep_idxs, "is_meta": True}
            )
        else:
            i = len(nodes)
            nodes.append({
                "obj": node,
                "name": getattr(node, "name", type(node).__name__),
                "deps": [],
                "is_meta": False,
            })
        idx_by_id[id(node)] = i
        return i

    roots = list(model) if isinstance(model, (list, tuple)) else [model]
    for r in roots:
        visit(r)
    seen: set[str] = set()
    for n in nodes:
        if n["name"] in seen:
            raise ValueError(
                f"WalkForward: duplicate node name {n['name']!r} — names label "
                f"leaderboard rows and must be unique across the graph"
            )
        seen.add(n["name"])
    return nodes


# ---------------------------------------------------------------------------
# WalkForward — the CV / OOF driver.
# ---------------------------------------------------------------------------


_VALID_CV = ("expanding-window", "rolling-window", "kfold")


class WalkForward:
    """Cross-validation driver: produces out-of-fold distributions for every
    node in a `Pipeline` / `Stacker` graph (and bare forecasters).

    ``WalkForward(cv=..., n_folds=...).fit_predict(model, X, y, ids=, timestamps=)``
    returns a `PipelineResult` mapping each node's name → its stitched OOF
    `DistributionForecast`. The model owns its own internal structure (a
    `Pipeline` does its lifter/calibrator inner splits itself); WalkForward
    owns only the outer fold loop.
    """

    def __init__(
        self,
        *,
        cv: str = "expanding-window",
        n_folds: int = 5,
        embargo: int = 0,
        refit_on_full: bool = False,
        shuffle: bool = False,
        random_state: int | None = None,
        rolling_window: int | None = None,
    ):
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
        self.refit_on_full = refit_on_full
        self.shuffle = shuffle
        self.random_state = random_state
        self.rolling_window = rolling_window
        # Set by fit_predict when refit_on_full=True; consumed by predict().
        self._nodes: list[dict] | None = None
        self._fitted: list[Any] | None = None

    # ---- run ----

    def fit_predict(
        self,
        model: Any,
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
        N = y.shape[0]
        sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

        nodes = _flatten(model)

        # Time-series CV sorts by timestamp; k-fold leaves rows as-is.
        order = np.arange(N) if self.cv == "kfold" else np.argsort(timestamps, kind="stable")
        Xo, yo, ids_o, ts_o = X[order], y[order], ids[order], timestamps[order]
        sw_o = sw[order] if sw is not None else None

        folds = self._make_folds(N)
        per_node: dict[int, list[tuple[np.ndarray, Any]]] = {i: [] for i in range(len(nodes))}

        for train_idx, test_idx in folds:
            fold_train: dict[int, Any] = {}
            fold_test: dict[int, Any] = {}
            for i, node in enumerate(nodes):
                obj = copy.deepcopy(node["obj"])
                if node["is_meta"]:
                    up_tr = [fold_train[j] for j in node["deps"]]
                    up_te = [fold_test[j] for j in node["deps"]]
                    dist_tr, dist_te = self._fit_meta(
                        obj, Xo, yo, ids_o, ts_o, train_idx, test_idx, up_tr, up_te, sw_o,
                    )
                else:
                    dist_tr, dist_te = self._fit_plain(
                        obj, Xo, yo, ids_o, ts_o, train_idx, test_idx, sw_o,
                    )
                fold_train[i] = dist_tr
                fold_test[i] = dist_te
                per_node[i].append((order[test_idx], dist_te))

        out: dict[str, Any] = {}
        for i, node in enumerate(nodes):
            prov = ProvenanceMeta.placeholder(node["name"], sigma_source="native")
            out[node["name"]] = _stitch_folds(
                per_node[i], timestamps=timestamps, provenance=prov,
            )

        # Refit each node on the full (sorted) training data → canonical models
        # for predict() on truly-unseen rows (sklearn's CV-then-final pattern).
        if self.refit_on_full:
            self._refit_full(nodes, Xo, yo, ids_o, ts_o, sw_o)

        return PipelineResult(forecasts=out)

    # ---- refit-on-full + predict on unseen rows ----

    def _refit_full(self, nodes, X, y, ids, ts, sw) -> None:
        fitted: list[Any] = [None] * len(nodes)
        canonical: dict[int, Any] = {}   # node index → in-sample full-data dist
        for i, node in enumerate(nodes):
            obj = copy.deepcopy(node["obj"])
            if node["is_meta"]:
                up = [canonical[j] for j in node["deps"]]
                _fit_with_optional_weight(
                    obj, X, y, sw, ids=ids, timestamps=ts, upstream=up,
                )
                canonical[i] = _predict_with_extras(obj, X, ids, ts, upstream=up)
            else:
                _fit_with_optional_weight(obj, X, y, sw, ids=ids, timestamps=ts)
                canonical[i] = _predict_with_extras(obj, X, ids, ts)
            fitted[i] = obj
        self._nodes = nodes
        self._fitted = fitted

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, Any]:
        """Predict on unseen rows with the canonical (full-train) models.

        Requires ``refit_on_full=True`` and a prior ``fit_predict``. Returns
        ``{node_name: DistributionForecast}`` — every node addressable, same as
        the leaderboard.
        """
        if self._fitted is None or self._nodes is None:
            raise RuntimeError(
                "predict() requires a prior fit_predict() with refit_on_full=True"
            )
        X = np.asarray(X)
        ids = np.asarray(ids)
        timestamps = np.asarray(timestamps)
        out: dict[str, Any] = {}
        node_dist: dict[int, Any] = {}
        for i, node in enumerate(self._nodes):
            f = self._fitted[i]
            if node["is_meta"]:
                up = [node_dist[j] for j in node["deps"]]
                dist = _predict_with_extras(f, X, ids, timestamps, upstream=up)
            else:
                dist = _predict_with_extras(f, X, ids, timestamps)
            node_dist[i] = dist
            out[node["name"]] = dist
        return out

    # ---- per-node fold fit ----

    @staticmethod
    def _fit_plain(f, X, y, ids, ts, tr, te, sw):
        sw_tr = sw[tr] if sw is not None else None
        _fit_with_optional_weight(
            f, X[tr], y[tr], sw_tr, ids=ids[tr], timestamps=ts[tr],
        )
        dist_tr = _predict_with_extras(f, X[tr], ids[tr], ts[tr])
        dist_te = _predict_with_extras(f, X[te], ids[te], ts[te])
        return dist_tr, dist_te

    @staticmethod
    def _fit_meta(m, X, y, ids, ts, tr, te, up_tr, up_te, sw):
        sw_tr = sw[tr] if sw is not None else None
        _fit_with_optional_weight(
            m, X[tr], y[tr], sw_tr, ids=ids[tr], timestamps=ts[tr], upstream=up_tr,
        )
        dist_tr = _predict_with_extras(m, X[tr], ids[tr], ts[tr], upstream=up_tr)
        dist_te = _predict_with_extras(m, X[te], ids[te], ts[te], upstream=up_te)
        return dist_tr, dist_te

    # ---- folds (mirror ForecastPipeline; shared into _cv when it shims) ----

    def _make_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.cv == "expanding-window":
            return self._expanding_folds(N)
        if self.cv == "rolling-window":
            return self._rolling_folds(N)
        return self._kfold_folds(N)

    def _expanding_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
        chunk_size = N // (self.n_folds + 1)
        if chunk_size < 2:
            raise ValueError(f"N={N} too small for n_folds={self.n_folds}")
        folds = []
        for k in range(self.n_folds):
            train_end = (k + 1) * chunk_size
            test_start = train_end + self.embargo
            is_last = (k == self.n_folds - 1)
            test_end = N if is_last else min(N, test_start + chunk_size)
            if test_start >= N:
                break
            folds.append((np.arange(0, train_end), np.arange(test_start, test_end)))
        return folds

    def _rolling_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
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
            folds.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
        if not folds:
            raise ValueError(
                f"rolling-window CV produced 0 folds (N={N}, w={w}, n_folds={self.n_folds})"
            )
        return folds

    def _kfold_folds(self, N: int) -> list[tuple[np.ndarray, np.ndarray]]:
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
            train_idx = np.sort(
                np.concatenate([chunks[j] for j in range(self.n_folds) if j != k])
            )
            folds.append((train_idx, test_idx))
        return folds
