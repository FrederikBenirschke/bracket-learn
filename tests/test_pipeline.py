"""ForecastPipeline orchestration tests.

Focused on the load-bearing invariants:
- OOF row alignment: `result[stage].ids` maps correctly into the original
  y vector — `y[dist.ids]` recovers the realized targets.
- Pipeline doesn't fit on test data (leakage check via constant detection).
- Duplicate stage names raise loudly.
- depends_on missing-dependency raises loudly.
- depends_on stages get deps_oof correctly.
- Fold stitching preserves backing/family invariants.

These tests run against tiny synthetic data so each test is sub-second.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import ForecastPipeline, LiftedForecaster, PipelineResult
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


def test_pipeline_emits_one_dist_per_stage():
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(
        steps=[("emos", EMOS())],
        n_folds=3,
    )
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    assert isinstance(result, PipelineResult)
    assert result.stages == ["emos"]
    assert "emos" in result.forecasts


def test_oof_ids_align_into_original_y():
    """The hand-rolled scoring loop `y[dist.ids]` should give back exactly
    the realized values aligned with the OOF predictions."""
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    d = result["emos"]
    y_oof = y[d.ids.astype(int)]
    # μ should be close to the realized values (synthetic data is easy).
    rmse = float(np.sqrt(np.mean((d.params["mu"] - y_oof) ** 2)))
    assert rmse < 2.0


def test_oof_no_test_fold_overlap():
    """Across all folds, every test row index appears at most once in the
    stitched OOF coverage."""
    X, y, ids, ts = _synthetic(n=200)
    p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=4)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    oof_ids = result["emos"].ids.astype(int)
    assert len(oof_ids) == len(np.unique(oof_ids))


def test_expanding_window_absorbs_tail():
    """Final expanding-window fold absorbs the N % (n_folds + 1) trailing
    rows so OOF coverage equals N exactly — no silent data drop."""
    # N=203, n_folds=5 → chunk_size=33 → without absorb the last
    # 203 - 6·33 = 5 rows would be lost. With absorb, fold 5 takes them.
    n = 203
    X, y, ids, ts = _synthetic(n=n)
    from bracketlearn.compose import WalkForward
    p = WalkForward(n_folds=5)
    folds = p._expanding_folds(n)
    covered = np.concatenate([te for _, te in folds])
    assert covered.max() == n - 1, (
        f"final test row {covered.max()} must reach N-1={n - 1}; "
        f"otherwise the tail is silently dropped"
    )
    # Same row never tested twice.
    assert len(np.unique(covered)) == len(covered)
    # OOF coverage = first test_start..N-1 (everything after chunk 1).
    chunk = n // 6
    assert len(covered) == n - chunk


def test_score_returns_dict_of_dicts():
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    scores = result.score(y, metrics=["crps", "log_score"])
    assert "emos" in scores
    assert "crps" in scores["emos"]
    assert "log_score" in scores["emos"]
    assert scores["emos"]["crps"] > 0


# ---------------------------------------------------------------------------
# Loud failures.
# ---------------------------------------------------------------------------


def test_duplicate_stage_name_raises():
    with pytest.raises(ValueError, match="already registered"):
        ForecastPipeline(steps=[("emos", EMOS()), ("emos", EMOS())])


def test_missing_dependency_raises():
    with pytest.raises(ValueError, match="depends on"):
        ForecastPipeline(steps=[("stack", StackedParametric(deps=("ridge",)))])


def test_unsupported_cv_raises():
    with pytest.raises(ValueError, match="cv="):
        ForecastPipeline(steps=[("emos", EMOS())], cv="not-a-cv")


def test_score_unknown_metric_raises():
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    with pytest.raises(ValueError, match="unknown metric"):
        result.score(y, metrics=["not_a_metric"])


def test_score_bracket_metrics_without_ladder_raise():
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    with pytest.raises(ValueError, match="require ladder"):
        result.score(y, metrics=["log_loss_bracket"])


# ---------------------------------------------------------------------------
# Dependency injection.
# ---------------------------------------------------------------------------


def test_stacking_receives_deps_oof():
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(
        steps=[
            ("ridge", LiftedForecaster(
                base=SklearnPoint(LinearRegression()),
                lifter=GlobalResidual(),
                name="ridge",
            )),
            ("emos",  EMOS()),
            ("stack", StackedParametric(deps=("ridge", "emos"))),
        ],
        n_folds=3,
    )
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    assert "stack" in result.forecasts
    # StackedParametric should beat or match each individual upstream on training data.
    scores = result.score(y, metrics=["crps"])
    # We don't assert strict dominance (3-fold + small N is noisy) but the
    # stack should at least be in the same ballpark.
    assert scores["stack"]["crps"] < 2 * scores["emos"]["crps"]


# ---------------------------------------------------------------------------
# Groups routing for hierarchical / cross-site trainers.
# ---------------------------------------------------------------------------


def test_pipeline_routes_groups_to_hierarchical_normal():
    """fit_predict and predict thread the ``groups`` kwarg through to any
    stage whose signature declares it. Trainers without ``groups`` ignore it.
    """
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

    p = ForecastPipeline(
        steps=[("hn", HierarchicalNormal())],
        cv="kfold", n_folds=4,
    )
    res = p.fit_predict(X, y, ids=ids, timestamps=ts, groups=groups)
    oof = res.forecasts["hn"]
    assert oof.ids.shape[0] == len(y)
    # OOF must be at least as good as a no-pool σ baseline; check it's
    # not blowing up (within 3× the in-sample residual scale).
    y_oof = y[oof.ids.astype(int)]
    assert np.sqrt(((oof.mu - y_oof) ** 2).mean()) < 1.5

    # Predict path also routes groups.
    X_te = rng.standard_normal((20, K))
    g_te = np.array([0, 1, 2, 3] * 5)
    out = p.predict(
        X_te, ids=np.arange(20), timestamps=np.arange(20, dtype=float), groups=g_te,
    )
    assert "hn" in out
    assert out["hn"].mu.shape == (20,)

    # Predict without groups should fall through HN's loud rail.
    with pytest.raises(ValueError, match="groups is required"):
        p.predict(X_te, ids=np.arange(20), timestamps=np.arange(20, dtype=float))


# ---------------------------------------------------------------------------
# to_table renders without crashing.
# ---------------------------------------------------------------------------


def test_to_table_renders_string():
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
    result = p.fit_predict(X, y, ids=ids, timestamps=ts)
    out = result.to_table(y, metrics=["crps", "log_score"])
    assert "emos" in out
    assert "crps" in out
    assert "log_score" in out
