"""A calibration suite. var(PIT) alone cannot diagnose a forecast.

var(PIT) is one number summarising a whole histogram, and it is blind to
most of the ways a predictive distribution goes wrong. Three concrete
failures it cannot see:

* **Bias.** A forecast shifted 3°F warm with the right spread has a PIT
  histogram piled at one end. Its variance can sit near 1/12 while every
  interval is centred in the wrong place. Only ``pit_mean`` sees this.
* **Which tail.** Two forecasts, one too heavy on the left and one too
  heavy on the right, produce the same var(PIT). ``pit_skew`` separates
  them, and ``tail_left``/``tail_right`` say by how much.
* **Bimodality.** A mixture that is too wide in the body and too narrow in
  the tails can average out to a neutral variance while fitting neither.
  ``pit_ks`` and the reliability curve catch it, and the variance does not.

var(PIT) therefore belongs in a suite rather than standing alone. This
module computes the suite on a family-agnostic interface. The caller
supplies a CDF, so a Gaussian, Student-t or mixture forecast all report the
same columns.

The three axes, and why all three are needed
--------------------------------------------
A forecast is useful only if it is calibrated, sharp and accurate at once,
and these trade off against each other.

1. **Calibration** (``pit_*``, ``reliability_mae``, ``coverage_*``): are
   the stated probabilities honest? A climatological forecast is perfectly
   calibrated and worthless.
2. **Sharpness** (``rmv``, ``sharpness_iqr``): how concentrated is the
   distribution? Sharpness is measured without reference to the outcome.
   It is a property of the forecast alone and cannot be optimised alone.
3. **Accuracy** (``crps``, ``log_score``): proper scores, which reward
   calibration and sharpness jointly. Report these as the summary, and the
   diagnostics above to explain why a score moved.

Discrete outcomes
-----------------
``grid_step`` handles settlement on a grid. NWS CLI reports an integer °F,
so a continuous CDF evaluated at the realized value is not the Rosenblatt
PIT. Uniformity needs a continuous Y as well as a continuous F. Passing
``grid_step=1.0`` switches to the mid-interval, continuity-corrected form

    F(y − h) + ½·[F(y + h) − F(y − h)],   h = grid_step/2

which is the deterministic analogue of the randomised PIT rather than the
randomised PIT itself. It uses no random number generator and so cannot
smear an outcome across its cell. The distinction matters here. The
randomised form shifted the mean +0.093…+0.111 on every model over 85,250
forecasts (2026-09-02), and this repo bans it for that reason.

Reading the numbers
-------------------

::

    pit_mean      0.5 neutral. Below 0.5 means the forecast runs high
                  (the outcome falls low in its distribution).
    pit_var       Compare against ``pit_var_neutral`` rather than against
                  1/12. On a discrete outcome the calibrated value is
                  strictly below 1/12. Below neutral is overdispersed
                  (intervals too wide, hump-shaped histogram), and above is
                  underdispersed (too narrow, U-shaped). ``pit_var_excess``
                  is the signed gap and is the column to read.
    pit_skew      0 neutral. Signs which tail carries the excess.
    pit_ks        0 is perfect. KS distance of the PIT from uniform, an
                  omnibus check that catches shapes the moments miss.
    reliability_mae
                  0 is perfect. Mean abs(empirical - nominal) coverage over
                  a grid of central intervals. The number to quote when
                  someone asks "are the intervals right".
    coverage_50/90
                  Should be 0.50 / 0.90. Interval coverage at the two
                  levels people actually read off a fan chart.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "calibration_suite",
    "pit_values",
    "neutral_pit_var",
    "NEUTRAL_PIT_VAR",
]

# Neutral var(PIT) for a continuous outcome, Var[U(0,1)] = 1/12.
NEUTRAL_PIT_VAR = 1.0 / 12.0


def neutral_pit_var(cell_probs: np.ndarray | None = None) -> float:
    """The var(PIT) a perfectly calibrated forecast actually attains.

    For a continuous outcome this is 1/12. For an outcome on a grid it is
    strictly less, and comparing a discrete PIT against 1/12 manufactures a
    spurious "overdispersed" verdict.

    The mid-interval PIT of a calibrated forecast is not Uniform(0,1). It can
    only take the value at the centre of each cell's probability mass, so the
    within-cell spread a continuous PIT would have is missing. Write p_i for
    the probability the forecast puts on the cell the outcome landed in. The
    exact result is then

        Var[PIT_mid] = 1/12 - E[p^2]/12

    Verified against simulation (1M rows, integer settlement, Gaussian
    forecast) to within 8e-5 at sigma = 1, 2, 3, 4.

    The naive PIT can therefore look closer to 1/12 than the corrected one
    while being the wrong quantity. The naive form is biased upward and the
    correct target is biased downward, and the two errors are easily confused
    for each other. Measured at grid step 1 degF, against the true continuous
    0.0833:

        sigma   naive    mid-interval   correct target (1/12 - E[p^2]/12)
        1.0    0.0870        0.0762                              0.0763
        2.0    0.0841        0.0815                              0.0815
        3.0    0.0838        0.0825                              0.0825
        4.0    0.0834        0.0827                              0.0829

    The mid-interval column matches its own target to 3 decimal places at
    every sigma. The naive column matches nothing.

    Pass the per-row cell probabilities to get the right reference, or pass
    nothing for the continuous case.
    """
    if cell_probs is None:
        return NEUTRAL_PIT_VAR
    p = np.asarray(cell_probs, dtype=float)
    if p.size == 0:
        raise ValueError("neutral_pit_var: empty cell_probs")
    return float(NEUTRAL_PIT_VAR * (1.0 - np.mean(p ** 2)))

# Central-interval levels the reliability curve is evaluated on. They span
# the range a reader cares about rather than being dense. A finer grid adds
# no information, because neighbouring levels are near-perfectly correlated.
_RELIABILITY_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def pit_values(
    cdf: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    *,
    grid_step: float | None = None,
) -> np.ndarray:
    """PIT values, continuity-corrected when the outcome lives on a grid.

    ``cdf`` maps thresholds to P(Y <= threshold) under the row's predictive
    distribution. The caller closes over μ, σ and the family, so this works
    for any distribution without a family switch here.

    ``grid_step`` is the outcome's resolution, 1.0 for integer °F. ``None``
    means a genuinely continuous outcome and uses the plain F(y).
    """
    y = np.asarray(y, dtype=float)
    if grid_step is None:
        return np.asarray(cdf(y), dtype=float)
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError(
            f"grid_step must be positive and finite, or None; got "
            f"{grid_step!r}. A NaN step NaNs every PIT and an infinite one "
            f"collapses them all to 0.5, either of which reads as a result."
        )
    h = grid_step / 2.0
    lo = np.asarray(cdf(y - h), dtype=float)
    hi = np.asarray(cdf(y + h), dtype=float)
    return lo + 0.5 * (hi - lo)


def _ks_uniform(u: np.ndarray) -> float:
    """KS distance between the empirical CDF of u and Uniform(0,1)."""
    n = u.size
    s = np.sort(u)
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - s, s - (i - 1) / n)))


def calibration_suite(
    cdf: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    *,
    sd: np.ndarray | None = None,
    quantile: Callable[[np.ndarray], np.ndarray] | None = None,
    grid_step: float | None = None,
    crps: np.ndarray | None = None,
) -> dict[str, float]:
    """Calibration, sharpness and accuracy for one set of predictions.

    Parameters
    ----------
    cdf
        ``thresholds -> P(Y <= threshold)``, elementwise over rows.
    y
        Realized outcomes.
    sd
        Per-row predictive standard deviation, for sharpness. For a
        Student-t this must be σ√(ν/(ν−2)) rather than the scale. Passing the
        scale understates dispersion, and the argument is named ``sd`` to
        make that harder to do.
    quantile
        ``levels -> value`` per row, used for coverage and the reliability
        curve. Without it those columns are omitted rather than
        approximated. A Gaussian-shaped guess at a t's quantiles would be a
        silent fallback (Rule #0.5).
    grid_step
        Outcome resolution. See ``pit_values``.
    crps
        Per-row CRPS if the caller has a closed form for its family. Omitted
        rather than approximated when absent.

    Returns
    -------
    dict
        Always: ``n``, ``pit_mean``, ``pit_var``, ``pit_var_neutral``,
        ``pit_var_excess``, ``pit_skew``, ``pit_ks``, ``tail_left``,
        ``tail_right``, ``log_score``.
        With ``sd``: ``rmv``.
        With ``quantile``: ``coverage_50``, ``coverage_90``,
        ``reliability_mae``, ``sharpness_iqr``.
        With ``crps``: ``crps``.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"calibration_suite: y must be 1-D; got {y.shape}")
    if y.size < 2:
        raise ValueError("calibration_suite: need >=2 rows")

    # Cell probabilities are needed twice, for the discrete log score and
    # for the neutral var(PIT) reference. They are computed once.
    cell = None
    if grid_step is not None:
        h = grid_step / 2.0
        lo_c = np.asarray(cdf(y - h), dtype=float)
        hi_c = np.asarray(cdf(y + h), dtype=float)
        cell = hi_c - lo_c
        u = lo_c + 0.5 * cell
    else:
        u = pit_values(cdf, y, grid_step=None)
    if not np.all(np.isfinite(u)):
        raise ValueError(
            "calibration_suite: the CDF returned non-finite PIT values: "
            "check for sigma<=0 or NaN moments before calling."
        )
    n = int(u.size)
    m = float(np.mean(u))
    v = float(np.var(u))
    sdv = float(np.std(u))

    # The calibrated target, which is below 1/12 whenever the outcome is on
    # a grid. Reading pit_var against 1/12 on discrete data invents an
    # overdispersion that is an artifact of the grid.
    neutral = neutral_pit_var(cell)

    out: dict[str, float] = {
        "n": n,
        "pit_mean": m,
        "pit_var": v,
        "pit_var_neutral": neutral,
        # Signed gap. Negative is overdispersed and positive is
        # underdispersed. This is the column to read, since pit_var alone
        # needs its reference.
        "pit_var_excess": v - neutral,
        # Skew of the PIT, saying which side of the distribution carries the
        # excess. It is zero for any symmetric miscalibration, and so adds
        # information var(PIT) does not have.
        "pit_skew": (float(np.mean(((u - m) / sdv) ** 3)) if sdv > 0 else 0.0),
        "pit_ks": _ks_uniform(u),
        # Tail masses. A calibrated forecast puts 5% of outcomes below its
        # 5th percentile and 5% above its 95th. These say which tail is
        # wrong, which the variance cannot.
        "tail_left": float(np.mean(u < 0.05)),
        "tail_right": float(np.mean(u > 0.95)),
    }

    # Log score on the discrete cell probability when the outcome is on a
    # grid. This is comparable across distribution families, unlike a
    # density, whose units depend on the family's parameterisation.
    if cell is not None:
        out["log_score"] = float(-np.mean(np.log(np.clip(cell, 1e-12, None))))
    else:
        out["log_score"] = float("nan")

    if sd is not None:
        s = np.asarray(sd, dtype=float)
        if s.shape != y.shape:
            raise ValueError(
                f"calibration_suite: sd has shape {s.shape}, y has {y.shape}"
            )
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("calibration_suite: sd must be finite and positive")
        # Root mean variance, the paper's sharpness column (Table 11).
        out["rmv"] = float(np.sqrt(np.mean(s ** 2)))

    if quantile is not None:
        errs = []
        for lvl in _RELIABILITY_LEVELS:
            a = (1.0 - lvl) / 2.0
            lo = np.asarray(quantile(np.full(n, a)), dtype=float)
            hi = np.asarray(quantile(np.full(n, 1.0 - a)), dtype=float)
            emp = float(np.mean((y >= lo) & (y <= hi)))
            errs.append(abs(emp - lvl))
            if abs(lvl - 0.50) < 1e-9:
                out["coverage_50"] = emp
            elif abs(lvl - 0.90) < 1e-9:
                out["coverage_90"] = emp
        # Mean |empirical - nominal| over the whole reliability curve. It is
        # one number for "are the intervals right", robust to a single level
        # happening to land well.
        out["reliability_mae"] = float(np.mean(errs))
        q25 = np.asarray(quantile(np.full(n, 0.25)), dtype=float)
        q75 = np.asarray(quantile(np.full(n, 0.75)), dtype=float)
        # Sharpness measured without the outcome, a property of the forecast
        # alone. It is distribution-free, so it compares across families
        # where RMV, which needs a finite variance, cannot.
        out["sharpness_iqr"] = float(np.mean(q75 - q25))

    if crps is not None:
        c = np.asarray(crps, dtype=float)
        if c.shape != y.shape:
            raise ValueError(
                f"calibration_suite: crps has shape {c.shape}, y has {y.shape}"
            )
        out["crps"] = float(np.mean(c))

    return out
