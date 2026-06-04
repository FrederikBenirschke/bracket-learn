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

## `TailRule.clip()`, the only rule shipped today

Mass beyond the outermost stored quantile is zero. The CDF returns 0 below the
leftmost quantile and 1 above the rightmost.

This is the safe default for ladder-priced contracts (the ladder's outer bins
absorb the would-be tail mass) and for quantile-backed dists scored against
bracket-shaped contracts. It misfires on any contract that pays off in the
tails. A vanilla call far out of the money, for instance, gets marked as "zero
probability of reaching the strike" when the model in fact has no information
out there.

### Interaction with `BracketLadder`

`BracketLadder` reads the CDF at every edge and diffs. When a quantile-backed
`dist` uses `TailRule.clip()` and the ladder fails to span
`[qvals[:, 0].min(), qvals[:, -1].max()]`, mass leaks. The ladder's `strict`
and `coverage_tol` machinery catches it (see [adapters.md](adapters.md)). Widen
the ladder to fix it; leave the tail rule alone.

## Planned: `gpd`, `gaussian_match`, `exponential`, `custom`

The README "Not yet" section and the `TailRule` docstring list these. They
share one pattern: each fits a tail-shape on the outermost stored quantiles,
then extrapolates analytically.

| Rule | Tail family | Fit from |
|------|-------------|----------|
| `gpd()` | Generalised Pareto | Peaks-over-threshold on training residuals |
| `gaussian_match()` | Half-normal | Matched to the outer two quantiles' slope |
| `exponential()` | Exponential | Matched to the outer quantile's local density |
| `custom(cdf=callable)` | Caller-supplied | You pass a `tau -> q` map per side |

Until those land, a `dist.cdf(x)` call outside `[qvals[:, 0], qvals[:, -1]]`
with a non-`clip` rule raises `NotImplementedError` and names the rule in the
message. That follows Rule #0.5: a loud raise beats returning garbage tail
mass.

## When the policy matters

| Contract shape | Tail rule choice |
|----------------|------------------|
| Bracket / multi-binary with finite ladder | `clip` is fine; widen the ladder if mass leaks |
| Binary on a deep OTM threshold | `clip` understates the risk; switch to `gpd` or `gaussian_match` when available |
| Vanilla call / put with payoff growing in the tail | `clip` marks the option worthless; use a tail-fitting rule |
| Spread (one-sided) | the relevant side needs a non-`clip` rule; the other side stays `clip` |

## Adapters declare which tails they need

A `ContractAdapter` subclass sets `needs_left_tail=True` or
`needs_right_tail=True` to signal that it integrates over the distribution
beyond the stored quantiles. `BracketLadder` defaults both flags to `False` (it
reads the CDF at finite edges only); a future `VanillaCall` will set the
right-tail flag.

Pairing a needs-tail adapter with a `clip` policy is a misconfiguration the
framework can catch at adapter construction. That wiring waits for a later
release, but the flags exist so the check lands without a breaking change.
