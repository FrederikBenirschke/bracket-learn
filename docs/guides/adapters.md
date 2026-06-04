# Contract adapters

A `ContractAdapter` turns a `DistributionForecast` into a `ContractForecast`:
a long-form table of contract IDs with one `fair_price` per row. This is the
last step before the framework hands off to a downstream sizing or execution
layer.

bracketlearn ships five adapters covering the contract shapes you meet on real
prediction-market venues:

| Adapter                | Pricing                            | Maps to (examples)                                          |
|------------------------|------------------------------------|-------------------------------------------------------------|
| `BinaryAbove(k)`       | `P(X > k)`                         | Kalshi "high above 80°F", "S&P > 5000 by Friday"            |
| `BinaryBelow(k)`       | `P(X ≤ k)`                         | Kalshi "GDP ≤ 2.5%", "low below freezing"                   |
| `Twin(k)`              | paired `P(X > k)` / `P(X ≤ k)`     | Polymarket spread (`Eagles -3.5`), total (`Over 47.5`)      |
| `ThresholdLadder(ks)`  | `[P(X > k_i)]` per strike          | Kalshi multi-threshold temperature ladders                  |
| `BracketLadder(edges_per_row)` | `[P(lo ≤ X < hi)]` per-row edges | Kalshi daily-rotating brackets; Polymarket weather brackets (pass `[edges]*N`) |

All five take any `DistributionForecast` (normal, student-t, mixture-normal,
quantile, or bracket backing) and emit a long-form `ContractForecast` with
`fair_price`, `entity_ids`, `group_id`, `contract_spec`, and provenance copied
through from the upstream distribution.

## `BinaryAbove` / `BinaryBelow`

Single-threshold binaries. One contract per entity.

```python
import numpy as np
from bracketlearn import BinaryAbove, BinaryBelow

p_above = BinaryAbove(strike=75.0).price(dist).fair_price   # (N,)
p_below = BinaryBelow(strike=32.0).price(dist).fair_price   # (N,)
```

`fair_price` clips to `[0, 1]`. A single CDF read needs no coverage check; it
stays unambiguous whatever the tail does.

## `Twin`

Paired YES / NO at one strike. Two rows per entity sharing `group_id`, so
calibrators can enforce `p_yes + p_no = 1`.

```python
from bracketlearn import Twin

contracts = Twin(strike=70.0).price(dist)
yes = contracts.fair_price[contracts.contract_ids == 0]   # P(X > 70)
no  = contracts.fair_price[contracts.contract_ids == 1]   # P(X ≤ 70)
np.testing.assert_allclose(yes + no, 1.0)
```

Convention: `contract_id=0` is YES = `P(X > k)`; `contract_id=1` is NO =
`P(X ≤ k)`. The two prices sum to exactly 1.0 within each entity by
construction.

## `ThresholdLadder`

One row per `P(X > k_i)`, S strikes total. The prices are survival-function
values at strictly increasing strikes, so they decrease monotonically and need
not sum to 1.

```python
from bracketlearn import ThresholdLadder

strikes = np.array([60.0, 70.0, 80.0, 90.0])
contracts = ThresholdLadder(strikes=strikes).price(dist)
# N · S rows; contracts.fair_price.reshape(N, S) is monotone-decreasing
# across axis=1.
```

## `BracketLadder`

The workhorse adapter for bracket ladders. It takes `edges_per_row`, a Python
list of length N where `edges_per_row[i]` has shape `(B_i + 1,)`, and emits one
contract row per bracket per entity. For each interval
`[edges_i[k], edges_i[k+1])`, the fair price is
`cdf(edges_i[k+1]) - cdf(edges_i[k])`.

Storage stays ragged: different rows may carry different `B_i` (Kalshi
occasionally adds an extra bracket for extreme-weather days). For the i.i.d.
case where every row shares the same edges, pass `edges_per_row=[edges] * N`.
The inner list holds N references to the same array, so it costs no extra
memory.

```python
from bracketlearn import BracketLadder

# Kalshi-style: edges shift each row around the forecasted mean.
edges_per_day = [
    np.array([mu - 10, mu - 3, mu, mu + 3, mu + 10])
    for mu in dist.params["mu"]
]
ladder = BracketLadder(
    edges_per_row=edges_per_day,
    include_tail_buckets=True,    # adds explicit "below edges[0]" and
                                  # "above edges[-1]" rows so per-entity
                                  # prices sum to exactly 1.0
)
contracts = ladder.price(dist)

# Shared-edge ladder (Polymarket weekly contracts):
edges = np.array([-np.inf, 0.0, 0.5, 1.0, np.inf])
ladder = BracketLadder(edges_per_row=[edges] * N)
```

### Coverage and `strict`

`BracketLadder.price` reads the CDF at the ladder edges and diffs. When the
ladder fails to span the distribution's effective support, mass falls off the
ends and row sums dip below 1.0. That missed mass biases contract prices
downward and usually signals a bug, so the adapter checks every row and
surfaces the failure.

- `strict=False` (default) emits a `UserWarning` whenever any row's missed mass
  exceeds `coverage_tol` (default `1e-4`). The warning reports the worst-row
  missed mass and how many rows tripped the tolerance.
- `strict=True` raises `ValueError` with the same payload instead.
- `include_tail_buckets=True` emits two extra rows per entity (the below-min
  and above-max tail mass) so per-entity prices sum to 1.0 by construction. The
  coverage check then becomes a no-op.

Use `strict=True` when downstream code requires coherent simplex probabilities
(log-loss scoring, isotonic calibration, sizing under a "probabilities sum to
1" budget).

To clear a coverage warning, widen the outer edges (use ±large numbers to catch
tail mass into the outer bins) or set `include_tail_buckets=True`. Loosening
`coverage_tol` hides the problem rather than fixing it.

### Edge semantics

`BracketLadder` uses closed-left, open-right intervals
(`[edges[k], edges[k+1])`). For continuous distributions the choice carries
zero measure, so the adapter exposes no knob.

### Output shape

`BracketLadder.price` returns a `ContractForecast` in **long form**. For the
shared-edge case where every row has the same `B`:

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

With ragged `edges_per_row` or `include_tail_buckets=True`, the per-row
contract count varies, so index by `entity_ids` instead of reshaping.

Implementation note: per-row edges use `DistributionForecast.cdf_at_grid` under
the hood (a vectorised CDF on a per-row evaluation grid), so parametric
backings stay vectorised even with ragged edges.

## Adding a new adapter

Implement the `ContractAdapter` protocol from `bracketlearn.protocols`:

```python
from bracketlearn.forecast import ContractForecast, DistributionForecast


class MyAdapter:
    name: str = "my_adapter"

    def price(self, dist: DistributionForecast) -> ContractForecast:
        ...
```

No base class required; duck typing on `.price(dist)` is enough. Follow the
`BracketLadder` example for provenance plumbing (carry forward
`dist.provenance.fit_window`, `fold_idx`, and the rest) so the resulting
`ContractForecast` stays auditable.
