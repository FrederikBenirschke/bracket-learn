"""ContractForecast, output of ContractAdapter.price() (§5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bracketlearn.forecast._meta import ProvenanceMeta


@dataclass(frozen=True)
class ContractSpec:
    """Typed serialisable spec for an adapter.

    ``edges_per_row`` is set by ladder adapters and carries the exact grid each
    entity was priced on. It exists so a scorer can verify the edges it is
    handed rather than trust them. ``brier_bracket`` and ``log_loss_bracket``
    use ``edges`` to decide which bracket the outcome fell in. On a rotating
    ladder, as when Kalshi relists daily, a caller passing one row's vector for
    all rows previously got a wrong number back with no error, because the
    shapes still lined up. It is left ``None`` by non-ladder adapters.
    """

    kind: str
    schema_version: int = 1
    edges_per_row: tuple[tuple[float, ...], ...] | None = None


@dataclass(frozen=True)
class ContractForecast:
    contract_ids: np.ndarray
    entity_ids: np.ndarray
    timestamps: np.ndarray
    fair_price: np.ndarray
    group_id: np.ndarray
    contract_spec: ContractSpec
    provenance: ProvenanceMeta

    def __post_init__(self) -> None:
        for arr in (self.contract_ids, self.entity_ids, self.timestamps,
                    self.fair_price, self.group_id):
            arr.setflags(write=False)
