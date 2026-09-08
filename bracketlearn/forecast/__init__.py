"""Data objects, comprising PointForecast, the DistributionForecast hierarchy,
and ContractForecast.

All are frozen dataclasses. The ndarrays inside are set read-only in
__post_init__ to make immutability real, since frozen=True alone freezes only
attribute binding and not buffer contents.

v0.3.0 layout
-------------
``DistributionForecast`` is an ``abc.ABC`` base with one concrete subclass per
backing.

    DistributionForecast (abstract)         → base.py
    ├── NormalForecast                      → parametric.py
    ├── StudentTForecast                    → parametric.py
    ├── MixtureNormalForecast               → parametric.py
    ├── QuantileForecast                    → quantile.py
    └── BracketForecast                     → bracket.py

Each subclass owns its storage (typed, no optional fields) and its math.
``DistributionForecast.from_*`` classmethods are preserved as thin
construction shims that route to the correct subclass.

This file re-exports the public names so callers keep using
``from bracketlearn.forecast import DistributionForecast, NormalForecast, ...``
unchanged.
"""

from __future__ import annotations

from bracketlearn.forecast._helpers import (
    bracket_probs_from_cdf_at_edges,
    normalize_bracket_probs,
)
from bracketlearn.forecast._meta import (
    PointForecast,
    ProvenanceMeta,
    TailPolicy,
    TailPolicyError,
    TailRule,
)
from bracketlearn.forecast.base import DistributionForecast
from bracketlearn.forecast.bracket import BracketForecast
from bracketlearn.forecast.contract import ContractForecast, ContractSpec
from bracketlearn.forecast.parametric import (
    MixtureNormalForecast,
    NormalForecast,
    StudentTForecast,
)
from bracketlearn.forecast.quantile import QuantileForecast

__all__ = [
    # leaf data objects
    "PointForecast",
    "ContractForecast",
    "ContractSpec",
    # distribution hierarchy
    "DistributionForecast",
    "NormalForecast",
    "StudentTForecast",
    "MixtureNormalForecast",
    "QuantileForecast",
    "BracketForecast",
    # tail policy
    "TailPolicy",
    "TailPolicyError",
    "TailRule",
    # provenance
    "ProvenanceMeta",
    # helpers
    "bracket_probs_from_cdf_at_edges",
    "normalize_bracket_probs",
]
