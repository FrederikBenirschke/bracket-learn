"""Shared notebook style — rcParams, palette, and a couple of plot helpers.

Imported at the top of every notebook. Keeps the look consistent and stops
each notebook from re-inventing its own bar-chart-with-two-bars aesthetic.

The headline pattern is sklearn's `plot_stack_predictors` example: a small
grid of scatter panels (predicted vs realized), one panel per model, with
the metrics annotated inside the panel rather than in a separate bar chart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global style — applied as a side effect on import.
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "figure.figsize": (8.5, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "regular",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.4,
    "scatter.edgecolors": "none",
    "font.family": "sans-serif",
})

# ---------------------------------------------------------------------------
# Model-family palette — tab10. One stable color per *family*; individual
# models within a family inherit it. Lookup falls through to a default tab10
# slot via _DEFAULT_CYCLE for anything unmapped.
# ---------------------------------------------------------------------------

_TAB10 = list(mpl.colormaps["tab10"].colors)

# Family → tab10 index. Picked so the most frequently compared families
# (baselines, native dist, lifted point, calibrated) land on adjacent /
# legible hues.
FAMILY_COLORS: dict[str, tuple[float, float, float]] = {
    "baseline":    _TAB10[7],   # gray
    "persistence": _TAB10[8],   # olive
    "point_lift":  _TAB10[0],   # blue
    "native_dist": _TAB10[1],   # orange
    "calibrated":  _TAB10[2],   # green
    "multistage":  _TAB10[4],   # purple
    "bracket":     _TAB10[3],   # red
    "sklearn":     _TAB10[5],   # brown
}

# Per-model overrides — used when a notebook wants a specific model to pop.
MODEL_COLORS: dict[str, tuple[float, float, float]] = {
    "emp":              FAMILY_COLORS["baseline"],
    "Empirical":        FAMILY_COLORS["baseline"],
    "persist1":         FAMILY_COLORS["persistence"],
    "persist24":        FAMILY_COLORS["persistence"],
    "persist168":       FAMILY_COLORS["persistence"],
    "Persist-1":        FAMILY_COLORS["persistence"],
    "Persist-24":       FAMILY_COLORS["persistence"],
    "Persist-168":      FAMILY_COLORS["persistence"],
    "ridge":            FAMILY_COLORS["point_lift"],
    "Ridge+GR":         FAMILY_COLORS["point_lift"],
    "qreg":             FAMILY_COLORS["native_dist"],
    "QuantileReg":      FAMILY_COLORS["native_dist"],
    "QuantileForest":   FAMILY_COLORS["native_dist"],
    "NGBoost":          FAMILY_COLORS["native_dist"],
    "NGBoostNormal":    FAMILY_COLORS["native_dist"],
    "ngb":              FAMILY_COLORS["native_dist"],
    "ngboost":          FAMILY_COLORS["native_dist"],
    "qf":               FAMILY_COLORS["native_dist"],
    "emos_iso":         FAMILY_COLORS["calibrated"],
    "EMOS+Iso":         FAMILY_COLORS["calibrated"],
    "QReg-best":        FAMILY_COLORS["calibrated"],
    "CumBinary":        FAMILY_COLORS["bracket"],
    "CumulativeBinary": FAMILY_COLORS["bracket"],
    "cum":              FAMILY_COLORS["bracket"],
    "sklearn RidgeCV":  FAMILY_COLORS["sklearn"],
    "sklearn Ridge":    FAMILY_COLORS["sklearn"],
    "LightGBM":         FAMILY_COLORS["sklearn"],
}

_DEFAULT_CYCLE = [_TAB10[i] for i in (0, 1, 2, 4, 3, 8, 7, 9, 5, 6)]


def color_for(name: str, fallback_index: int | None = None) -> tuple[float, float, float]:
    """Stable color for a model name. Unknown names cycle through tab10."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    if fallback_index is not None:
        return _DEFAULT_CYCLE[fallback_index % len(_DEFAULT_CYCLE)]
    # Hash to a stable slot so two notebooks pick the same color for the
    # same unknown name.
    idx = abs(hash(name)) % len(_DEFAULT_CYCLE)
    return _DEFAULT_CYCLE[idx]


# ---------------------------------------------------------------------------
# Headline plot: predicted-vs-realized scatter grid.
# ---------------------------------------------------------------------------


def predicted_vs_realized_grid(
    panels: list[tuple[str, np.ndarray, np.ndarray, dict[str, float]]],
    *,
    ncols: int = 3,
    units: str = "",
    title: str | None = None,
    diag_pad: float = 0.05,
    figsize_per_panel: tuple[float, float] = (3.5, 3.5),
) -> plt.Figure:
    """sklearn-`plot_stack_predictors`-style scatter grid.

    Args:
        panels: list of ``(name, mu, y_true, metrics_dict)``. The
            ``metrics_dict`` is rendered as text inside the panel
            (e.g. ``{"MAE": 0.42, "RMSE": 0.55, "CRPS": 0.27}``).
        ncols: panels per row.
        units: appended to axis labels (e.g. ``"$100k"``).
        title: overall figure title.
        diag_pad: fractional padding around the (lo, hi) data range.
        figsize_per_panel: panel size; figure scales by grid shape.
    """
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols,
                 figsize_per_panel[1] * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    # Robust axis range. Anchor on realized y (true observations can't be
    # outliers); ignore predicted outliers so one extrapolation pathology
    # doesn't collapse every panel to a 5%-wide stripe.
    y_all = np.concatenate([p[2] for p in panels])
    mu_all = np.concatenate([p[1] for p in panels])
    lo = float(min(np.nanmin(y_all), np.nanpercentile(mu_all, 1)))
    hi = float(max(np.nanmax(y_all), np.nanpercentile(mu_all, 99)))
    span = hi - lo
    lo, hi = lo - diag_pad * span, hi + diag_pad * span

    for idx, (name, mu, y_true, metrics) in enumerate(panels):
        ax = axes[idx // ncols, idx % ncols]
        color = color_for(name, fallback_index=idx)
        ax.scatter(y_true, mu, s=12, color=color, alpha=0.55)
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, linestyle="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        # Metric annotation inside the panel (sklearn-style).
        lines = [f"{k} = {v:.3f}" for k, v in metrics.items()]
        ax.text(0.04, 0.96, "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, family="monospace",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=2.0))
        # Flag degenerate panels (predictions collapse to a near-constant —
        # most often the marginal-y baseline, by construction).
        pred_std = float(np.nanstd(mu))
        y_std = float(np.nanstd(y_true))
        if y_std > 0 and pred_std / y_std < 0.05:
            ax.text(0.5, 0.04,
                    "constant prediction\n(spread / y-std < 5 %)",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, style="italic", color="dimgray")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="box")

    # Hide unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    # Axis labels only on edge panels.
    xlabel = f"realized{(' (' + units + ')') if units else ''}"
    ylabel = f"predicted{(' (' + units + ')') if units else ''}"
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reliability with per-model probability histograms below.
# ---------------------------------------------------------------------------


def reliability_with_histogram(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str = "Reliability",
) -> plt.Figure:
    """Top: reliability curve per model. Bottom: predicted-probability
    histogram per model (same x-axis). Tells you both calibration AND
    whether the model is ever confident.

    Each entry in ``series`` is ``(name, mean_predicted, hit_rate)`` from
    a binning, plus optionally raw predicted probabilities for the bottom
    panel — we accept ``(name, mp, hr)`` and also infer raw via the
    third return if a 1-D vector matches the bin count.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6.5),
        sharex=True, gridspec_kw={"height_ratios": [2.4, 1]},
    )
    ax_top.plot([0, 1], [0, 1], color="black", lw=0.8, linestyle="--",
                label="perfect")
    for name, mp, hr in series:
        color = color_for(name)
        ax_top.plot(mp, hr, marker="o", color=color, label=name, lw=1.2)
    ax_top.set_ylabel("empirical hit rate")
    ax_top.set_title(title)
    ax_top.legend(loc="upper left")

    # Bottom panel — stacked histograms of the binned mean-predicted-prob
    # values themselves (rough proxy for "where is the model placing mass").
    for name, mp, _ in series:
        color = color_for(name)
        ax_bot.hist(mp, bins=np.linspace(0, 1, 11),
                    histtype="step", color=color, lw=1.4, label=name)
    ax_bot.set_xlabel("mean predicted bracket probability")
    ax_bot.set_ylabel("# bins")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Family-colored horizontal leaderboard bar — used by leaderboard_zoo.
# ---------------------------------------------------------------------------


def leaderboard_bar(
    rows: Iterable[tuple[str, float]],
    *,
    baseline_name: str,
    baseline_value: float,
    skill_label: str = "CRPSS",
    families: Mapping[str, str] | None = None,
    title: str = "Leaderboard",
    fig_height_per_row: float = 0.32,
) -> plt.Figure:
    """Horizontal skill-score bar, sorted within family then by skill.

    Args:
        rows: ``[(model_name, value_or_skill)]``. If ``baseline_value`` is
            passed alongside ``baseline_name``, ``value`` is interpreted as
            raw CRPS and skill is computed as ``1 − v / baseline_value``.
            Pass already-computed skill values by setting
            ``baseline_value=1.0`` (so skill = ``1 − v``) — keep the math
            on the caller side and use this helper only for layout.
        families: optional map ``model_name → family_key`` from
            FAMILY_COLORS. Models without a family use color_for(name).
    """
    rows = [(n, v) for n, v in rows if n != baseline_name]
    # Skill scores.
    pairs = [(n, 1.0 - v / baseline_value) for n, v in rows]
    # Sort: first by family (if provided), then by skill within.
    def fam_of(name: str) -> str:
        return (families or {}).get(name, "_zzz")
    pairs.sort(key=lambda p: (fam_of(p[0]), -p[1]))

    names = [p[0] for p in pairs]
    skills = [p[1] for p in pairs]
    colors = [
        FAMILY_COLORS.get(fam_of(n), color_for(n))
        for n in names
    ]

    fig, ax = plt.subplots(
        figsize=(10, max(4.0, fig_height_per_row * len(names))),
    )
    ax.barh(names, skills, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.6)
    ax.invert_yaxis()
    ax.set_xlabel(f"{skill_label} vs {baseline_name} ({baseline_value:.3f})")
    ax.set_title(title)
    for i, s in enumerate(skills):
        ax.text(s + (0.005 if s > 0 else -0.005), i, f"{s:+.3f}",
                va="center", ha="left" if s > 0 else "right", fontsize=8.5)
    if families:
        # Tiny family legend in the upper-right.
        seen = []
        handles = []
        for n in names:
            fam = fam_of(n)
            if fam in seen or fam == "_zzz":
                continue
            seen.append(fam)
            handles.append(plt.Rectangle((0, 0), 1, 1,
                                          color=FAMILY_COLORS.get(fam, "gray")))
        if handles:
            ax.legend(handles, seen, loc="lower right",
                      title="family", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CDF overlay — replaces the 3-house grouped-bar plot in housing_brackets.
# ---------------------------------------------------------------------------


def cdf_overlay_for_examples(
    dists_by_name: Mapping[str, object],     # mapping name → DistributionForecast
    *,
    row_indices: list[int],
    y_realized: np.ndarray,
    edges: np.ndarray,
    units: str = "",
    title: str = "Predicted CDFs vs realized",
) -> plt.Figure:
    """One panel per example row; on each panel overlay the forecast CDF
    from each model, plus a vertical line at the realized value and
    light dashes at the bracket edges.

    Replaces the 3-grouped-bar plot — same information, immediately
    readable as a probability distribution rather than a stack of bars.
    """
    n = len(row_indices)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.2 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]
    edges = np.asarray(edges, dtype=float)
    x_grid = np.linspace(edges[0], edges[-1], 200)
    for ax_idx, row_i in enumerate(row_indices):
        ax = axes[ax_idx]
        for name, dist in dists_by_name.items():
            color = color_for(name)
            # cdf(x_grid) returns (N, M); pick the entity's row.
            cdf_vals = dist.cdf(x_grid)[row_i]
            ax.plot(x_grid, cdf_vals, color=color, label=name, lw=1.3)
        ax.axvline(y_realized[row_i], color="black",
                   linestyle="--", lw=1.0,
                   label=f"realized = {y_realized[row_i]:.2f}")
        for e in edges:
            ax.axvline(e, color="lightgray", lw=0.4)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"row {row_i}\nP(X ≤ x)")
        if ax_idx == 0:
            ax.legend(loc="lower right", ncol=2, fontsize=8)
    axes[-1].set_xlabel(f"x{(' (' + units + ')') if units else ''}")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig
