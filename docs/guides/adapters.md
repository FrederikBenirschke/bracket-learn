# Contract adapters

A `ContractAdapter` turns a `DistributionForecast` into a `ContractForecast` —
a long-form table of contract IDs with a `fair_price` per row. This is the
last step before the framework hands off to a downstream sizing or
execution layer.

bracketlearn ships one fully implemented adapter (`BracketLadder`) plus
stubs that raise `NotImplementedError` for shapes that aren't on the v0.x
critical path. The stubs are intentional placeholders, not bugs — calling
one tells you precisely which adapter you'd need to fill in.

## `BracketLadder`

The workhorse adapter. Takes `edges` (length `B+1`) and emits one contract
row per bracket per entity. For each interval `[edges[k], edges[k+1])`,
the fair price is `cdf(edges[k+1]) - cdf(edges[k])`.

```python
import numpy as np
from bracketlearn import BracketLadder

ladder = BracketLadder(edges=np.array([-np.inf, 0.0, 0.5, 1.0, np.inf]))
contracts = ladder.price(dist)         # dist: any DistributionForecast
print(contracts.fair_price.shape)      # (N * B,)
```

### Coverage and `strict`

`BracketLadder.price` reads the CDF at the ladder edges and diffs. If the
ladder doesn't span the distribution's effective support, mass falls off
the ends and row sums dip below 1.0. That's almost always a bug — silent
mass loss biases contract prices downward — so the adapter checks every
row and surfaces the failure.

- `strict=False` (default) — emits a `UserWarning` whenever any row's
  missed mass exceeds `coverage_tol` (default `1e-4`). The warning reports
  the worst-row missed mass and how many rows tripped the tolerance.
- `strict=True` — raises `ValueError` with the same payload instead.

Use `strict=True` when downstream code requires coherent simplex
probabilities (e.g. log-loss scoring, isotonic calibration, sizing under
a "probabilities sum to 1" budget).

```python
# Wide ladder absorbs the tails — row sums stay at 1.
edges = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])

# Narrow ladder loses mass — warns at default tolerance.
edges = np.array([-1.0, 0.0, 1.0])
```

The fix is almost always to widen the outer edges (use ±large numbers to
catch tail mass into the outer bins) rather than tweaking `coverage_tol`.

### Edge semantics

`BracketLadder` defaults to closed-left, open-right intervals
(`[edges[k], edges[k+1])`). The `edge_semantics` constructor arg accepts
`BracketEdges.CLOSED_OPEN` (default) or `BracketEdges.OPEN_CLOSED`. The
choice only matters for atoms exactly at the edges — for continuous
distributions the difference is zero-measure.

### Output shape

`BracketLadder.price` returns a `ContractForecast` in **long form**:

| field             | shape  | content |
|-------------------|--------|---------|
| `contract_ids`    | (N·B,) | `[0,1,...,B-1]` tiled N times |
| `entity_ids`      | (N·B,) | `dist.ids` repeated B times |
| `fair_price`      | (N·B,) | `probs.flatten()` |
| `group_id`        | (N·B,) | `dist.ids` repeated B times (one ladder per entity) |

To recover the (N, B) probability matrix:

```python
probs = contracts.fair_price.reshape(N, B)
np.testing.assert_allclose(probs.sum(axis=1), 1.0)  # iff ladder covered
```

## `BinaryAbove` / `BinaryBelow` (stubs)

`BinaryAbove(threshold)` would emit one contract per entity with
`fair_price = P(X > threshold)`. `BinaryBelow` is the symmetric stub.
Both raise `NotImplementedError` today — the standard way to get a binary
contract is to use `BracketLadder` with a 2-bracket edge vector
`[-inf, threshold, +inf]` and read off the matching column.

## `VanillaCall` / `VanillaPut` (stubs)

Would price European-style payoffs (`max(X - K, 0)` for a call) from any
`DistributionForecast`. Closed form for parametric backings, numerical
integration for quantile / bracket. Not yet implemented — the v0.x
focus has been on binary and bracket contracts.

## `ThresholdLadder`, `Twin`, `LinearCombo`, `PerRow`, `Custom` (stubs)

Same story: stub classes that raise `NotImplementedError` with a TODO
comment naming the shape they'd implement. Useful to grep for if you're
extending the framework — each one has a docstring describing its
contract shape.

## Adding a new adapter

Implement the `ContractAdapter` protocol from `bracketlearn.protocols`:

```python
from bracketlearn.forecast import ContractForecast, DistributionForecast


class MyAdapter:
    name: str = "my_adapter"

    def price(self, dist: DistributionForecast) -> ContractForecast:
        ...
```

There's no required base class — duck typing on `.price(dist)` is enough.
Follow the `BracketLadder` example for provenance plumbing (carry forward
`dist.provenance.fit_window`, `fold_idx`, etc.) so the resulting
`ContractForecast` stays auditable.
