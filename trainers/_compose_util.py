"""Shared helper for meta-combiners: resolve upstream forecasts to an
ordered list.

The composition contract (``Stacker([p1, p2], meta)``) feeds a meta-combiner
its upstream ``DistributionForecast`` objects **positionally**, in declared
order, via ``upstream=[...]`` when run under ``WalkForward``. Names are not the
wiring — they live only on the leaderboard.

Per Rule #0.5: a missing/empty upstream raises loud — never a silent empty
list or partial set.
"""

from __future__ import annotations

from typing import Any


def resolve_upstream(
    upstream: list[Any] | None,
    *,
    where: str,
) -> list[Any]:
    """Return upstream forecasts as an ordered list.

    ``upstream`` carries the forecasts positionally, in declared order.
    ``where`` labels the call site in error messages.
    """
    if upstream is None:
        raise ValueError(f"{where}: pass upstream=[dist, ...]")
    ups = list(upstream)
    if not ups:
        raise ValueError(f"{where}: upstream=[] is empty — need ≥1 forecast")
    return ups


def upstream_label(i: int) -> str:
    """Positional label for the i-th upstream in error messages."""
    return f"upstream[{i}]"
