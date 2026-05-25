"""Tail extrapolation policy (§7).

Asymmetric by default: TailPolicy = (left: TailRule, right: TailRule).
Use TailPolicy.same(rule) when left == right.

Required when constructing a DistributionForecast from a finite
representation (quantile, empirical). Parametric backings with full
support (normal, student_t) skip the policy.

v0.1 ships only ``TailRule.clip()`` — mass beyond the outermost quantile
is zero. ``gaussian_match`` / ``gpd`` / ``exponential`` / ``custom`` are
planned for v0.2; see README "Not yet" section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# TailRule — one side of a TailPolicy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailRule:
    """One side of a tail policy."""

    kind: Literal["clip"]
    params: dict                    # kind-specific parameters

    # ----------------------------------------------------------- factories

    @staticmethod
    def clip() -> TailRule:
        """Mass beyond outermost quantile is zero. Triggers a loud warning
        when paired with an unbounded adapter that declares
        needs_<side>_tail=True (e.g. VanillaCall on right tail)."""
        return TailRule(kind="clip", params={})


# ---------------------------------------------------------------------------
# TailPolicy — paired (left, right).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailPolicy:
    left: TailRule
    right: TailRule

    @classmethod
    def same(cls, rule: TailRule) -> TailPolicy:
        return cls(left=rule, right=rule)

    @classmethod
    def asym(cls, *, left: TailRule, right: TailRule) -> TailPolicy:
        return cls(left=left, right=right)


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class TailPolicyError(ValueError):
    """Raised when a tail policy is incompatible with the backing."""
