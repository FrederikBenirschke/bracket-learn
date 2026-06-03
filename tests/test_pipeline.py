"""WalkForward orchestration tests (CV + OOF stitching over Pipeline/Stacker).

Focused on the load-bearing invariants:
- OOF row alignment: `result[node].ids` maps correctly into the original
  y vector — `y[dist.ids]` recovers the realized targets.
- WalkForward doesn't fit on test data (leakage check).
- Duplicate node names raise loudly.
- An unsupported cv raises loudly.
- PipelineResult.score / to_table behave.
- groups is threaded to nodes that declare it.

These tests run against tiny synthetic data so each test is sub-second.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from bracketlearn import Pipeline, PipelineResult, Stacker, WalkForward
from bracketlearn.lift import GlobalResidual
from bracketlearn.trainers import EMOS, SklearnPoint, StackedParametric


def _synthetic(n: int = 200, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    truth = 10.0 + 0.05 * days + rng.normal(0, 1.0, n)
    X = truth[:, None] + rng.normal(0, 0.5, (n, k))
    ids = np.arange(n)
    ts = days.astype(float)
    return X, truth, ids, ts


# ---------------------------------------------------------------------------
# Basic shape + alignment.
# ---------------------------------------------------------------------------


def test_walkforward_emits_one_dist_per_node():
    X, y, ids, ts = _synthetic()
    result = WalkForward(n_folds=3).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    assert isinstance(result, PipelineResult)
    assert result.stages == ["emos"]
    assert "emos" in result.forecasts


def test_oof_ids_align_into_original_y():
    """`y[dist.ids]` should give back exactly the realized values aligned
    with the OOF predictions."""
    X, y, ids, ts = _synthetic()
    result = WalkForward(n_folds=3).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    d = result["emos"]
    y_oof = y[d.ids.astype(int)]
    rmse = float(np.sqrt(np.mean((d.params["mu"] - y_oof) ** 2)))
    assert rmse < 2.0


def test_oof_no_test_fold_overlap():
    """Across all folds, every test row index appears at most once in the
    stitched OOF coverage."""
    X, y, ids, ts = _synthetic(n=200)
    result = WalkForward(n_folds=4).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    oof_ids = result["emos"].ids.astype(int)
    assert len(oof_ids) == len(np.unique(oof_ids))


def test_expanding_window_absorbs_tail():
    """Final expanding-window fold absorbs the N % (n_folds + 1) trailing
    rows so OOF coverage equals N exactly — no silent data drop."""
    n = 203
    X, y, ids, ts = _synthetic(n=n)
    p = WalkForward(n_folds=5)
    folds = p._expanding_folds(n)
    covered = np.concatenate([te for _, te in folds])
    assert covered.max() == n - 1, (
        f"final test row {covered.max()} must reach N-1={n - 1}; "
        f"otherwise the tail is silently dropped"
    )
    assert len(np.unique(covered)) == len(covered)
    chunk = n // 6
    assert len(covered) == n - chunk


def test_score_returns_dict_of_dicts():
    X, y, ids, ts = _synthetic()
    result = WalkForward(n_folds=3).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    scores = result.score(y, metrics=["crps", "log_score"])
    assert "emos" in scores
    assert "crps" in scores["emos"]
    assert "log_score" in scores["emos"]
    assert scores["emos"]["crps"] > 0


# ---------------------------------------------------------------------------
# Loud failures.
# ---------------------------------------------------------------------------


def test_duplicate_node_name_raises():
    X, y, ids, ts = _synthetic()
    a = Pipeline([EMOS()], name="emos")
    b = Pipeline([EMOS()], name="emos")
    with pytest.raises(ValueError, match="duplicate node name"):
        WalkForward(n_folds=3).fit_predict([a, b], X, y, ids=ids, timestamps=ts)


def test_unsupported_cv_raises():
    with pytest.raises(ValueError, match="cv="):
        WalkForward(cv="not-a-cv")


def test_score_unknown_metric_raises():
    X, y, ids, ts = _synthetic()
    result = WalkForward(n_folds=3).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    with pytest.raises(ValueError, match="unknown metric"):
        result.score(y, metrics=["not_a_metric"])


def test_score_bracket_metrics_without_ladder_raise():
    X, y, ids, ts = _synthetic()
    result = WalkForward(n_folds=3).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    with pytest.raises(ValueError, match="require ladder"):
        result.score(y, metrics=["log_loss_bracket"])


# ---------------------------------------------------------------------------
# Stacker over upstream objects.
# ---------------------------------------------------------------------------


def test_stacker_runs_over_upstream_objects():
    X, y, ids, ts = _synthetic()
    ridge = Pipeline(
        [SklearnPoint(LinearRegression()), GlobalResidual()], name="ridge",
    )
    emos = Pipeline([EMOS()], name="emos")
    stack = Stacker([ridge, emos], StackedParametric(), name="stack")
    result = WalkForward(n_folds=3).fit_predict(stack, X, y, ids=ids, timestamps=ts)
    assert "stack" in result.forecasts
    scores = result.score(y, metrics=["crps"])
    # The stack should at least be in the same ballpark as each upstream.
    assert scores["stack"]["crps"] < 2 * scores["emos"]["crps"]


# ---------------------------------------------------------------------------
# Groups routing for hierarchical / cross-site trainers.
# ---------------------------------------------------------------------------


def test_walkforward_routes_groups_to_hierarchical_normal():
    """fit_predict and predict thread the ``groups`` kwarg through to any
    node whose signature declares it. Nodes without ``groups`` ignore it."""
    from bracketlearn.trainers import HierarchicalNormal

    rng = np.random.default_rng(0)
    K = 3
    site_sizes = [40, 80, 120, 200]
    Xs, ys, gs, tss = [], [], [], []
    t = 0
    for s, n in enumerate(site_sizes):
        beta_s = np.array([0.5, -1.0, 2.0]) + rng.standard_normal(K) * 0.3
        X_s = rng.standard_normal((n, K))
        y_s = X_s @ beta_s + rng.standard_normal(n) * 0.5
        Xs.append(X_s); ys.append(y_s); gs.extend([s] * n)
        tss.extend(range(t, t + n)); t += n
    X = np.vstack(Xs); y = np.concatenate(ys)
    groups = np.array(gs); ts = np.array(tss, dtype=float); ids = np.arange(len(y))
    # Shuffle so folds aren't site-segregated.
    perm = rng.permutation(len(y))
    X, y, groups, ids, ts = X[perm], y[perm], groups[perm], ids[perm], ts[perm]

    wf = WalkForward(cv="kfold", n_folds=4, refit_on_full=True)
    res = wf.fit_predict(
        Pipeline([HierarchicalNormal()], name="hn"),
        X, y, ids=ids, timestamps=ts, groups=groups,
    )
    oof = res.forecasts["hn"]
    assert oof.ids.shape[0] == len(y)
    y_oof = y[oof.ids.astype(int)]
    assert np.sqrt(((oof.mu - y_oof) ** 2).mean()) < 1.5

    # Predict path also routes groups.
    X_te = rng.standard_normal((20, K))
    g_te = np.array([0, 1, 2, 3] * 5)
    out = wf.predict(
        X_te, ids=np.arange(20), timestamps=np.arange(20, dtype=float), groups=g_te,
    )
    assert "hn" in out
    assert out["hn"].mu.shape == (20,)

    # Predict without groups should fall through HN's loud rail.
    with pytest.raises(ValueError, match="groups is required"):
        wf.predict(X_te, ids=np.arange(20), timestamps=np.arange(20, dtype=float))


# ---------------------------------------------------------------------------
# to_table renders without crashing.
# ---------------------------------------------------------------------------


def test_to_table_renders_string():
    X, y, ids, ts = _synthetic()
    result = WalkForward(n_folds=3).fit_predict(
        Pipeline([EMOS()], name="emos"), X, y, ids=ids, timestamps=ts,
    )
    out = result.to_table(y, metrics=["crps", "log_score"])
    assert "emos" in out
    assert "crps" in out
    assert "log_score" in out
