"""ContractForecast — output of ContractAdapter.price() (§5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bracketlearn.forecast._meta import ProvenanceMeta


@dataclass(frozen=True)
class ContractSpec:
    """Typed serialisable spec for an adapter."""

    kind: str
    schema_version: int = 1


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
