"""Tests for ``score.bootstrap_ci``.

What is pinned here is mostly the ways a bootstrap is silently wrong — it
returns a plausible-looking interval in every one of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from bracketlearn.score import bootstrap_ci, edge_alignment


def _ladders(n_days=200, K=6, seed=0):
    """Contracts with real ladder structure: one winner per day, by
    construction, so the within-day outcomes are dependent."""
    rng = np.random.default_rng(seed)
    q, m, r, cl = [], [], [], []
    for d in range(n_days):
        q.append(rng.dirichlet(np.ones(K)))
        m.append(rng.dirichlet(np.ones(K)))
        oh = np.zeros(K)
        oh[rng.integers(K)] = 1.0
        r.append(oh)
        cl.append(np.full(K, d))
    return [np.concatenate(x) for x in (q, m, r, cl)]


def test_unclustered_is_not_a_single_group():
    """The bug this caught: treating cluster=None as ONE group containing every
    observation resamples the identical sample every draw, giving a zero-width
    interval that looks like an extraordinarily precise estimate."""
    q, m, r, _ = _ladders()
    out = bootstrap_ci(edge_alignment, q, m, r, n_boot=400, seed=1)
    assert out["hi"] > out["lo"], "interval must have positive width"
    assert out["se"] > 0.0
    assert out["n_clusters"] == out["n_obs"], (
        "unclustered means one observation per group, not one group total")


def test_point_is_the_full_sample_statistic():
    """`point` must be fn on the DATA, never a bootstrap mean — the two differ
    by the bootstrap bias and only one of them is the estimate."""
    q, m, r, cl = _ladders()
    out = bootstrap_ci(edge_alignment, q, m, r, cluster=cl, n_boot=300, seed=3)
    assert out["point"] == pytest.approx(edge_alignment(q, m, r), rel=1e-12)


def test_clustering_changes_the_interval_and_keeps_the_point():
    q, m, r, cl = _ladders()
    a = bootstrap_ci(edge_alignment, q, m, r, n_boot=800, seed=2)
    b = bootstrap_ci(edge_alignment, q, m, r, cluster=cl, n_boot=800, seed=2)
    assert a["point"] == pytest.approx(b["point"], rel=1e-12)
    assert b["n_clusters"] == 200
    assert a["n_clusters"] == 1200
    assert b["hi"] > b["lo"]


def test_cluster_labels_may_be_strings_and_unsorted():
    """Real labels are "KATL|2026-08-20", arriving in event order — not ints,
    not sorted. Grouping must not depend on either."""
    q, m, r, cl = _ladders(n_days=60)
    labels = np.array([f"stn|{int(c):03d}" for c in cl])
    out = bootstrap_ci(edge_alignment, q, m, r, cluster=labels,
                       n_boot=200, seed=4)
    assert out["n_clusters"] == 60
    assert out["n_obs"] == 360


def test_whole_clusters_are_resampled_together():
    """A block bootstrap draws clusters, never individual members. With one
    cluster per distinct value, every resample length is a multiple of K."""
    rng = np.random.default_rng(5)
    K, D = 4, 50
    q = rng.uniform(0.1, 0.9, K * D)
    m = rng.uniform(0.1, 0.9, K * D)
    r = (rng.uniform(size=K * D) < m).astype(float)
    cl = np.repeat(np.arange(D), K)
    seen: list[int] = []

    def spy(qq, mm, rr):
        seen.append(qq.size)
        return edge_alignment(qq, mm, rr)

    bootstrap_ci(spy, q, m, r, cluster=cl, n_boot=25, seed=6)
    assert all(s == K * D for s in seen[1:]), (
        "each resample must contain exactly n_clusters whole clusters")


def test_seed_is_deterministic():
    q, m, r, cl = _ladders()
    kw = dict(cluster=cl, n_boot=200, seed=11)
    a = bootstrap_ci(edge_alignment, q, m, r, **kw)
    b = bootstrap_ci(edge_alignment, q, m, r, **kw)
    assert a == b


def test_alpha_widens_the_interval():
    q, m, r, cl = _ladders()
    tight = bootstrap_ci(edge_alignment, q, m, r, cluster=cl, n_boot=800,
                         alpha=0.32, seed=7)
    wide = bootstrap_ci(edge_alignment, q, m, r, cluster=cl, n_boot=800,
                        alpha=0.01, seed=7)
    assert (wide["hi"] - wide["lo"]) > (tight["hi"] - tight["lo"])


def test_a_known_signal_is_recovered_and_excludes_zero():
    """Positive control: if the CI cannot separate a real effect from zero the
    function is useless, and a test that only checks 'crosses zero' would pass
    on a broken implementation."""
    rng = np.random.default_rng(8)
    D, K = 400, 6
    q, m, r, cl = [], [], [], []
    for d in range(D):
        mm = rng.dirichlet(np.ones(K))
        w = rng.integers(K)
        oh = np.zeros(K)
        oh[w] = 1.0
        # q leans toward the true winner: a genuine positive edge
        qq = 0.6 * mm + 0.4 * oh
        q.append(qq / qq.sum()); m.append(mm); r.append(oh)
        cl.append(np.full(K, d))
    q, m, r, cl = [np.concatenate(x) for x in (q, m, r, cl)]
    out = bootstrap_ci(edge_alignment, q, m, r, cluster=cl, n_boot=1500, seed=9)
    assert out["point"] > 0
    assert out["lo"] > 0, "a planted edge must be resolved away from zero"


def test_mismatched_lengths_raise():
    q, m, r, cl = _ladders(n_days=20)
    with pytest.raises(ValueError, match="share length"):
        bootstrap_ci(edge_alignment, q, m, r[:-1], n_boot=10)
    with pytest.raises(ValueError, match="one label per observation"):
        bootstrap_ci(edge_alignment, q, m, r, cluster=cl[:-1], n_boot=10)


def test_degenerate_metric_raises_rather_than_returning_nan():
    """A metric that scores on the full sample but almost never on a resample
    is a broken setup, not an uncertain one — it must raise rather than return
    an interval built from a handful of surviving draws.

    Note the order: the point estimate is computed BEFORE any resampling, so a
    metric that fails outright fails there with its own error. This guard is
    for the subtler case where only the resamples degenerate.
    """
    q, m, r, cl = _ladders(n_days=30)
    calls = {"n": 0}

    def fails_after_the_point_estimate(*args):
        calls["n"] += 1
        if calls["n"] == 1:          # the full-sample point estimate
            return edge_alignment(*args)
        raise ValueError("undefined on this resample")

    with pytest.raises(ValueError, match="degenerate"):
        bootstrap_ci(fails_after_the_point_estimate, q, m, r, cluster=cl,
                     n_boot=50)


def test_bad_alpha_and_n_boot_raise():
    q, m, r, _ = _ladders(n_days=10)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(edge_alignment, q, m, r, alpha=0.0, n_boot=10)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(edge_alignment, q, m, r, n_boot=0)


# ---------------------------------------------------------------------------
# Coverage. The property that makes an interval an interval.
#
# The tests above pin mechanics — grouping, determinism, error paths. None of
# them would fail if the interval were systematically too narrow, which is the
# way a bootstrap is usually wrong. These simulate a known truth and count how
# often the interval contains it.
# ---------------------------------------------------------------------------


def _ea_draw(rng, n_clusters, per_cluster, shared):
    """One dataset. With ``shared``, the whole cluster gets ONE edge draw and
    ONE outcome — the dependence a clustered bootstrap must account for and an
    i.i.d. one cannot. Sharing only the edge is not enough: measured width
    ratio 0.99 (edge alone) vs 2.69 (edge and outcome)."""
    q, m, r, cl = [], [], [], []
    for c in range(n_clusters):
        base = rng.normal(0.0, 0.25)
        hit = rng.uniform() < 0.5
        for _ in range(per_cluster):
            mm = rng.uniform(0.2, 0.8)
            edge = base if shared else rng.normal(0.0, 0.25)
            q.append(np.clip(mm + edge, 0.01, 0.99))
            m.append(mm)
            r.append(float(hit) if shared else float(rng.uniform() < mm))
            cl.append(c)
    return (np.array(q), np.array(m), np.array(r), np.array(cl))


@pytest.mark.parametrize("shared", [False, True])
def test_clustered_interval_covers_at_about_the_nominal_rate(shared):
    """~95% nominal should cover ~95% of the time, dependence or not."""
    rng = np.random.default_rng(20260904)
    hits = 0
    reps = 120
    for _ in range(reps):
        q, m, r, cl = _ea_draw(rng, 60, 6, shared)
        truth = float(np.mean((q - m) * (m - m + (r - m))))  # EA on this draw
        out = bootstrap_ci(edge_alignment, q, m, r, cluster=cl,
                           n_boot=200, seed=int(rng.integers(1 << 30)))
        hits += out["lo"] <= truth <= out["hi"]
    cov = hits / reps
    # Loose band: 120 reps gives +-0.04 of binomial noise at p=0.95.
    assert 0.86 <= cov <= 1.0, f"coverage {cov:.3f} is far from nominal 0.95"


def test_iid_bootstrap_undercovers_when_clusters_are_dependent():
    """The reason `cluster` exists. With a cluster-wide shared edge, the i.i.d.
    interval must be visibly narrower than the clustered one — if it is not,
    `cluster` is doing nothing and the parameter is decorative."""
    rng = np.random.default_rng(7)
    widths_iid, widths_cl = [], []
    for _ in range(30):
        q, m, r, cl = _ea_draw(rng, 40, 8, shared=True)
        a = bootstrap_ci(edge_alignment, q, m, r, n_boot=300, seed=1)
        b = bootstrap_ci(edge_alignment, q, m, r, cluster=cl, n_boot=300, seed=1)
        widths_iid.append(a["hi"] - a["lo"])
        widths_cl.append(b["hi"] - b["lo"])
    ratio = float(np.mean(widths_cl) / np.mean(widths_iid))
    assert ratio > 2.0, (
        f"clustered/iid width ratio {ratio:.2f}: clustering should widen the "
        "interval substantially when clusters share a common component")


def test_single_cluster_raises_instead_of_returning_zero_width():
    """One cluster label for everything means every resample is the same
    sample. That returned a zero-width interval, which reads as extraordinary
    precision rather than as broken input."""
    q, m, r, _ = _ladders(n_days=50)
    with pytest.raises(ValueError, match="ONE cluster"):
        bootstrap_ci(edge_alignment, q, m, r, cluster=np.zeros(q.size),
                     n_boot=50)


def test_few_clusters_warns_about_undercoverage():
    """Coarsening clusters to be 'conservative' does the opposite below ~20:
    measured coverage falls to ~0.74 at K=3."""
    q, m, r, _ = _ladders(n_days=60)
    cl = np.repeat(np.arange(5), q.size // 5)
    with pytest.warns(UserWarning, match="under-covers"):
        bootstrap_ci(edge_alignment, q, m, r, cluster=cl, n_boot=50)
