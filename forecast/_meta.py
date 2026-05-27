"""Metadata + leaf data objects with no dependencies on forecast subclasses.

Lives here so other modules in this package can import these without
worrying about circulars: ``_meta`` is the bottom of the dependency
graph. Contains:

- ``TailRule`` / ``TailPolicy`` / ``TailPolicyError`` — tail extrapolation policy (§7).
- ``ProvenanceMeta`` — audit / reproducibility schema (§5.4).
- ``Backing`` / ``ParametricFamily`` — discriminator enums kept as compat
  shims; subclasses expose them via ``@property``.
- ``PointForecast`` — §5.1 leaf data object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Tail extrapolation policy (§7).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailRule:
    """One side of a tail policy."""

    kind: Literal["clip"]
    params: dict

    @staticmethod
    def clip() -> TailRule:
        return TailRule(kind="clip", params={})


@dataclass(frozen=True)
class TailPolicy:
    left: TailRule
    right: TailRule

    @classmethod
    def same(cls, rule: TailRule) -> TailPolicy:
        return cls(left=rule, right=rule)


class TailPolicyError(ValueError):
    """Raised when a tail policy is incompatible with the backing."""


# ---------------------------------------------------------------------------
# ProvenanceMeta — audit/reproducibility schema (§5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceMeta:
    forecaster_name: str
    forecaster_version: str
    fit_window: tuple[datetime, datetime]
    fold_idx: int | Literal["prequential"] | None
    calibration_set_hash: str | None
    random_seed: int | None
    code_sha: str
    feature_matrix_hash: str
    created_at: datetime
    sigma_source: Literal["native", "lifted", "none"] = "none"
    conversion_chain: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def placeholder(
        cls,
        forecaster_name: str,
        *,
        sigma_source: Literal["native", "lifted", "none"] = "native",
        random_seed: int | None = None,
    ) -> ProvenanceMeta:
        now = datetime.now()
        return cls(
            forecaster_name=forecaster_name,
            forecaster_version="0.1",
            fit_window=(datetime(2024, 1, 1), now),
            fold_idx=None,
            calibration_set_hash=None,
            random_seed=random_seed,
            code_sha="dev",
            feature_matrix_hash="-",
            created_at=now,
            sigma_source=sigma_source,
        )


# ---------------------------------------------------------------------------
# Backing + ParametricFamily enums — compat shims exposed via @property
# on each DistributionForecast subclass.
# ---------------------------------------------------------------------------


class Backing(StrEnum):
    PARAMETRIC = "parametric"
    QUANTILE = "quantile"
    EMPIRICAL = "empirical"
    BRACKET = "bracket"


class ParametricFamily(StrEnum):
    NORMAL = "normal"
    STUDENT_T = "student_t"
    MIXTURE_NORMAL = "mixture_normal"


# ---------------------------------------------------------------------------
# PointForecast (§5.1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointForecast:
    mu: np.ndarray
    ids: np.ndarray
    timestamps: np.ndarray
    provenance: ProvenanceMeta

    def __post_init__(self) -> None:
        for arr in (self.mu, self.ids, self.timestamps):
            arr.setflags(write=False)
        if not (len(self.mu) == len(self.ids) == len(self.timestamps)):
            raise ValueError(
                f"length mismatch: mu={len(self.mu)} ids={len(self.ids)} "
                f"timestamps={len(self.timestamps)}"
            )
