# Contract adapters

A `ContractAdapter` turns a `DistributionForecast` into a `ContractForecast` —
a long-form table of contract IDs with a `fair_price` per row. This is the
last step before the framework hands off to a downstream sizing or
execution layer.

bracketlearn ships six adapters covering the contract shapes you'll see on
real prediction-market venues:

| Adapter                | Pricing                            | Maps to (examples)                                          |
|------------------------|------------------------------------|-------------------------------------------------------------|
| `BinaryAbove(k)`       | `P(X > k)`                         | Kalshi "high above 80°F", "S&P > 5000 by Friday"            |
| `BinaryBelow(k)`       | `P(X ≤ k)`                         | Kalshi "GDP ≤ 2.5%", "low below freezing"                   |
| `Twin(k)`              | paired `P(X > k)` / `P(X ≤ k)`     | Polymarket spread (`Eagles -3.5`), total (`Over 47.5`)      |
| `ThresholdLadder(ks)`  | `[P(X > k_i)]` per strike          | Kalshi multi-threshold temperature ladders                  |
| `BracketLadder(edges)` | `[P(lo ≤ X < hi)]` shared edges    | Polymarket weather brackets, fixed weekly contracts         |
| `PerRowBracketLadder`  | per-row edges (each row its own)   | Kalshi daily-rotating brackets (edges shift day-by-day)     |

All six take any `DistributionForecast` (normal / student-t /
mixture-normal / quantile / bracket backings) and emit a long-form
`ContractForecast` with `fair_price`, `entity_ids`, `group_id`,
`contract_spec`, and provenance copied through from the upstream
distribution.

## `BinaryAbove` / `BinaryBelow`

Single-threshold binaries. One contract per entity.

```python
import numpy as np
from bracketlearn import BinaryAbove, BinaryBelow

p_above = BinaryAbove(strike=75.0).price(dist).fair_price   # (N,)
p_below = BinaryBelow(strike=32.0).price(dist).fair_price   # (N,)
```

`fair_price` is clipped to `[0, 1]`. No coverage check — a single CDF
read is unambiguous regardless of tail behaviour.

## `Twin`

Paired YES / NO at one strike. Two rows per entity sharing `group_id`
(so calibrators can enforce `p_yes + p_no = 1`).

```python
from bracketlearn import Twin

contracts = Twin(strike=70.0).price(dist)
yes = contracts.fair_price[contracts.contract_ids == 0]   # P(X > 70)
no  = contracts.fair_price[contracts.contract_ids == 1]   # P(X ≤ 70)
np.testing.assert_allclose(yes + no, 1.0)
```

Convention: `contract_id=0` is YES = `P(X > k)`; `contract_id=1` is
NO = `P(X ≤ k)`. By construction prices sum to exactly 1.0 within each
entity.

## `ThresholdLadder`

One row per `P(X > k_i)`, S strikes total. Prices are survival-function
values at strictly increasing strikes, so they decrease monotonically —
they are **not** required to sum to 1.

```python
from bracketlearn import ThresholdLadder

strikes = np.array([60.0, 70.0, 80.0, 90.0])
contracts = ThresholdLadder(strikes=strikes).price(dist)
# N · S rows; contracts.fair_price.reshape(N, S) is monotone-decreasing
# across axis=1.
```

## `BracketLadder`

The workhorse adapter for shared-edge ladders. Takes `edges` (length
`B+1`) and emits one contract row per bracket per entity. For each
interval `[edges[k], edges[k+1])`, the fair price is
`cdf(edges[k+1]) - cdf(edges[k])`.

```python
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

`BracketLadder` uses closed-left, open-right intervals
(`[edges[k], edges[k+1])`). For continuous distributions the choice is
zero-measure, so no knob is exposed.

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

## `PerRowBracketLadder`

Same idea as `BracketLadder` but each row carries its own edge vector.
Motivated by Kalshi-style daily-rotating temperature brackets: the
five-bucket ladder for NYC max-temp shifts day-by-day around the
forecasted mean.

```python
from bracketlearn import PerRowBracketLadder

# One edge vector per row in `dist`. Lengths may differ (ragged storage).
edges_per_day = [
    np.array([mu - 10, mu - 3, mu, mu + 3, mu + 10])
    for mu in dist.params["mu"]
]
ladder = PerRowBracketLadder(
    edges_per_row=edges_per_day,
    include_tail_buckets=True,    # adds explicit "below edges[0]" and
                                  # "above edges[-1]" rows so per-entity
                                  # prices sum to exactly 1.0
)
contracts = ladder.price(dist)
```

`include_tail_buckets=True` emits two extra contract rows per entity —
the below-min and above-max tail mass — so the per-entity prices form a
true simplex. With `include_tail_buckets=False` the same coverage check
as `BracketLadder` (warn / strict-raise) gates against silent tail
leakage.

Implementation note: per-row edges use `DistributionForecast.cdf_at_grid`
under the hood (vectorised CDF on a per-row evaluation grid), so
parametric backings stay fully vectorised even though the edges are
ragged.

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
