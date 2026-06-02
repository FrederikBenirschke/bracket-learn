"""Shared helper for meta-combiners: resolve upstream forecasts to an
ordered list.

The clean composition contract (``Stacker([p1, p2], meta)``) feeds a
meta-combiner its upstream ``DistributionForecast`` objects **positionally**,
in declared order, via ``upstream=[...]``. Names are not the wiring — they
live only on the leaderboard.

The legacy contract fed the same forecasts by name via
``deps_oof={name: dist}`` keyed by the meta's ``depends_on``. The five parent
weather stacking trainers (and the old ``ForecastPipeline``) still use it,
so this resolver accepts either and returns the canonical ordered list. Once
every caller passes ``upstream=``, the ``deps_oof`` branch (and the metas'
``deps`` field) can be deleted.

Per Rule #0.5: a missing/ambiguous upstream raises loud — never a silent
empty list or partial set.
"""

from __future__ import annotations

from typing import Any


def resolve_upstream(
    depends_on: tuple[str, ...],
    deps_oof: dict[str, Any] | None,
    upstream: list[Any] | None,
    *,
    where: str,
) -> list[Any]:
    """Return upstream forecasts as an ordered list.

    Exactly one of ``upstream`` (new, positional) or ``deps_oof`` (legacy,
    name-keyed) must carry the forecasts. When both are given, ``upstream``
    wins (the caller is on the new contract). ``where`` labels the call site
    in error messages.
    """
    if upstream is not None:
        ups = list(upstream)
        if not ups:
            raise ValueError(f"{where}: upstream=[] is empty — need ≥1 forecast")
        return ups
    if deps_oof:
        missing = set(depends_on) - set(deps_oof)
        if missing:
            raise ValueError(
                f"{where} needs deps_oof for {depends_on}; "
                f"missing {sorted(missing)} (got {list(deps_oof)})"
            )
        return [deps_oof[n] for n in depends_on]
    raise ValueError(
        f"{where}: pass upstream=[dist, ...] (or legacy deps_oof={{name: dist}})"
    )


def upstream_label(depends_on: tuple[str, ...], i: int) -> str:
    """Human label for the i-th upstream in error messages: its declared
    name when known (legacy/named callers), else a positional tag."""
    if i < len(depends_on):
        return repr(depends_on[i])
    return f"upstream[{i}]"
