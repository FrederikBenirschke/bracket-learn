# Tail policies

A `TailPolicy` declares what the CDF does beyond the outermost stored quantile
of a `DistributionForecast`. The quantile-backed factory `from_quantiles`
requires one. Parametric backings (`normal`, `student_t`, `mixture_normal`)
define their CDFs on all of ℝ and ignore it.

A `TailPolicy` carries one `TailRule` for the left tail and one for the right.
Construct the dataclass directly, or use the `TailPolicy.same(rule)` shorthand
when both sides share a rule.

```python
from bracketlearn.forecast import TailPolicy, TailRule

# Same rule on both tails (the common case).
policy = TailPolicy.same(TailRule.clip())

# Asymmetric: construct the dataclass directly.
policy = TailPolicy(left=TailRule.clip(), right=TailRule.clip())
```

## `TailRule.clip()`

`clip` is the tail rule bracketlearn ships. Mass beyond the outermost stored
quantile is zero: the CDF returns 0 below the leftmost quantile and 1 above the
rightmost.

This suits ladder-priced contracts, where the ladder's outer bins absorb the
would-be tail mass, and quantile-backed dists scored against bracket-shaped
contracts. Because `clip` assigns zero probability past the stored quantiles, a
contract that pays off out there reads as worthless. Keep the priced grid inside
the quantile span.

### Interaction with `BracketLadder`

`BracketLadder` reads the CDF at every edge and diffs. When a quantile-backed
`dist` uses `clip` and the ladder fails to span
`[qvals[:, 0].min(), qvals[:, -1].max()]`, mass leaks. The ladder's `strict`
and `coverage_tol` machinery catches it (see [adapters.md](adapters.md)). Widen
the ladder to fix it; leave the tail rule alone.

## Adapters declare which tails they need

A `ContractAdapter` exposes `needs_left_tail` and `needs_right_tail`. Every
shipped adapter sets both to `False`, since each reads the CDF at finite edges
only. The flags give the framework a hook to check an adapter that integrates
past the stored quantiles against the tail policy it needs.
