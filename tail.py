"""Tail extrapolation policy (§7).

Asymmetric by default: TailPolicy = (left: TailRule, right: TailRule).
Use TailPolicy.same(rule) when left == right.

TailRule is a spec; FittedTail is per-row, per-side state computed lazily
at first tail-region query and cached on the DistributionForecast.

Required when constructing a DistributionForecast from a finite
representation (quantile, empirical). Parametric backings with full
support (normal, student_t) skip the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# FitContext — passed to custom log-density callables.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitContext:
    """Context handed to TailRule.custom log-density callables."""

    top_quantiles: np.ndarray       # (n,) — quantile values used to fit
    top_taus: np.ndarray            # (n,) — corresponding tau levels
    side: Literal["left", "right"]


# ---------------------------------------------------------------------------
# TailRule — one side of a TailPolicy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailRule:
    """One side of a tail policy. Spec only — fitted state lives in
    FittedTail (cached on the DistributionForecast)."""

    kind: Literal["gaussian_match", "gpd", "exponential", "clip", "custom"]
    params: dict                    # kind-specific parameters
    custom_fn: Callable[[np.ndarray, FitContext], np.ndarray] | None = None

    # ----------------------------------------------------------- factories

    @staticmethod
    def gaussian_match(
        use_top_n_quantiles: int = 3,
        fit_loss: Literal["wls", "ols", "mle"] = "wls",
    ) -> "TailRule":
        """Fit a Gaussian to the top/bottom N quantiles; extrapolate.

        fit_loss=wls is the order-statistic-variance-weighted fit (default).
        n_quantiles >= 2 is a precondition; raises if violated.
        """
        ...

    @staticmethod
    def gpd(threshold_quantile: float = 0.95) -> "TailRule":
        """Generalized Pareto fit above/below the threshold quantile.

        Validates threshold_quantile is reachable from the backing's
        quantile grid at construction; raises TailPolicyError otherwise.
        """
        ...

    @staticmethod
    def exponential(rate_from_top_n: int = 2) -> "TailRule":
        """Exponential decay; rate fit from top N quantiles."""
        ...

    @staticmethod
    def clip() -> "TailRule":
        """Mass beyond outermost quantile is zero. Triggers a loud warning
        when paired with an unbounded adapter that declares
        needs_<side>_tail=True (e.g. VanillaCall on right tail)."""
        ...

    @staticmethod
    def custom(
        log_density_fn: Callable[[np.ndarray, FitContext], np.ndarray],
    ) -> "TailRule":
        """User-supplied log-density (numerical stability for deep tails).

        Signature: (x: ndarray, ctx: FitContext) -> log-density ndarray.
        Must be vectorised over x.
        """
        ...


# ---------------------------------------------------------------------------
# TailPolicy — paired (left, right).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailPolicy:
    left: TailRule
    right: TailRule

    @classmethod
    def same(cls, rule: TailRule) -> "TailPolicy":
        return cls(left=rule, right=rule)

    @classmethod
    def asym(cls, *, left: TailRule, right: TailRule) -> "TailPolicy":
        return cls(left=left, right=right)


# ---------------------------------------------------------------------------
# FittedTail — per-row, per-side fitted state.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedTail:
    """Fitted tail state for one row × one side.

    Cached on DistributionForecast._tail_cache keyed by (side, policy_hash).
    Invalidated implicitly: calibrators return a new immutable forecast
    with empty cache.
    """

    rule: TailRule
    side: Literal["left", "right"]
    fitted_params: dict             # kind-specific


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class TailPolicyError(ValueError):
    """Raised when a tail policy is incompatible with the backing
    (e.g. gpd(threshold=0.95) on a quantile grid that stops at 0.9)."""
