# Tail policies

A `TailPolicy` declares what the CDF should do beyond the outermost
stored quantile of a `DistributionForecast`. It's required for the
quantile-backed factory `from_quantiles` and unused for parametric
backings (`normal`, `student_t`, `mixture_normal`) whose CDFs are
defined on all of ℝ.

Tail policies are **asymmetric**: a `TailPolicy` carries one `TailRule`
for the left tail and one for the right. The shorthand `TailPolicy.same(rule)`
duplicates the rule across both sides.

```python
from bracketlearn.tail import TailPolicy, TailRule

# Same rule on both tails.
policy = TailPolicy.same(TailRule.clip())

# Asymmetric — clip on the left, something else on the right.
policy = TailPolicy.asym(left=TailRule.clip(), right=TailRule.clip())
```

## `TailRule.clip()` — the only rule shipped today

Mass beyond the outermost stored quantile is zero. The CDF returns 0
below the leftmost quantile and 1 above the rightmost.

This is the safe default for ladder-priced contracts (the ladder's outer
bins absorb the would-be tail mass) and for quantile-backed dists that
get scored against bracket-shaped contracts. It is **the wrong default**
for any contract that pays off in the tails — e.g. a vanilla call far
out of the money — because the model says "zero probability of reaching
the strike" when really it has no information.

### Interaction with `BracketLadder`

`BracketLadder` reads the CDF at every edge and diffs. If a
quantile-backed `dist` uses `TailRule.clip()` AND the ladder doesn't
span `[qvals[:, 0].min(), qvals[:, -1].max()]`, mass is silently lost.
The ladder's `strict` / `coverage_tol` machinery catches this — see
[adapters.md](adapters.md). The fix is to widen the ladder, not to swap
the tail rule.

## Planned: `gpd`, `gaussian_match`, `exponential`, `custom`

These are listed in the README "Not yet" section and the `TailRule`
docstring. They share the same pattern: each one fits a tail-shape on
the outermost stored quantiles, then extrapolates analytically.

| Rule | Tail family | Fit from |
|------|-------------|----------|
| `gpd()` | Generalised Pareto | Peaks-over-threshold on training residuals |
| `gaussian_match()` | Half-normal | Matched to the outer two quantiles' slope |
| `exponential()` | Exponential | Matched to the outer quantile's local density |
| `custom(cdf=callable)` | Caller-supplied | User passes a `tau -> q` map per side |

Until those land, attempting to call `dist.cdf(x)` outside `[qvals[:, 0],
qvals[:, -1]]` with a non-`clip` rule raises `NotImplementedError` with
the rule name in the message. That's Rule #0.5 — better than silently
returning garbage tail mass.

## When the policy matters

| Contract shape | Tail rule choice |
|----------------|------------------|
| Bracket / multi-binary with finite ladder | `clip` is fine; widen the ladder if mass leaks |
| Binary on a deep OTM threshold | `clip` understates risk — switch to `gpd` / `gaussian_match` when available |
| Vanilla call / put with payoff growing in the tail | Same — `clip` will mark the option worthless |
| Spread (one-sided) | The relevant side needs a non-`clip` rule; the other can stay `clip` |

## Adapters declare which tails they need

`ContractAdapter` subclasses can set `needs_left_tail=True` and / or
`needs_right_tail=True` to signal that they integrate over the
distribution beyond stored quantiles. `BracketLadder` defaults to
`needs_left_tail=False, needs_right_tail=False` (it only reads the CDF
at finite edges); a future `VanillaCall` will set the right tail flag.

Pairing a needs-tail adapter with a `clip` policy is a misconfiguration
the framework can detect at adapter construction time — that wiring
isn't in place today, but the flags exist so the check can be added
without a breaking change.
