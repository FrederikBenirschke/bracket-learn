# Accuracy vs value: scoring a price against a reference

The proper-scoring rules in the [scoring guide](scoring.md) (CRPS, log-score,
Brier, log-loss) all answer one question. Is my price close to the outcome?
That is *accuracy*. A trader on a prediction market asks a second question. Is
my price more valuable than the one already quoted? That is *value*. Value
grades a price against a **reference price** `m`, which may be a market quote,
a consensus, or any baseline forecast, rather than against the truth.

The two questions have different answers. A more accurate forecast can be worth
less to trade. The [`score`](../api/score) module ships the metrics that measure
value, namely `edge_alignment`, `value_report`, and their bracket-ladder
wrappers.

> This sits one step before the [trade-decision layer that bracketlearn leaves
> to you](../index.md). Value is still *scoring*. It grades prices and does not
> size positions. Unlike calibration, it grades them the way a trader cares
> about.

## 1. The PnL of a price

For one binary contract, write `q` for your price of YES, `m` for the reference
price, `r ∈ {0,1}` for the realized outcome, and `π` for the (latent) true
probability, so `E[r] = π`.

Buying YES at price `m` costs `m` and collects `1` if the event occurs. Acting
on the edge `q − m` (buy when positive, sell when negative), sized by the edge,
the expected profit on one contract is

```
E[PnL] = (q − m)(π − m)
```

The profit is the reference's mispricing `(π − m)`. Your price only sets the
direction and size of the bet. If the reference is already correct (`m = π`),
no price `q` earns anything. Summed over many contracts, total PnL is an inner
product.

```
PnL ≈ ⟨ q − m , π − m ⟩
```

## 2. Decomposing the PnL

Let `δ = π − m` be the reference's mispricing, the quantity to be captured, and
let `ε = π − q` be your error against the truth. Substituting `q − m = δ − ε`
gives

```
PnL ≈ ‖δ‖²  −  ⟨ ε , δ ⟩
       │          │
       │          └─ your error projected onto the mispricing
       └─ the inefficiency available in the reference
```

Three consequences follow.

1. **No inefficiency, no profit.** If `‖δ‖² = 0`, so the reference is right, PnL
   is zero for *any* forecast. A correct price cannot be out-predicted.
2. **Most error is free.** Only the part of the error that *aligns with the
   mispricing* is lost. Error where the reference is already right (`δ ≈ 0`)
   costs nothing. Nobody trades there.
3. **Shared bias.** If the error tracks the reference's, `ε → δ`, then
   `q = π − ε → π − δ = m`, and your price collapses onto the reference exactly
   where it is most wrong. A more accurate forecast that shares the reference's
   blind spots is worthless for trading.

> **The price that makes money holds errors orthogonal to the reference's
> mispricing, accurate where the reference is wrong and free to be sloppy where
> it is right.** This is the Grossman–Stiglitz point in microcosm (§7). A price
> aggregates common information, so the only exploitable signal is information
> *orthogonal* to it.

## 3. Calibration ≠ value

Calibration, log-loss, and CRPS all minimize `‖ε‖`, the closeness of `q` to
truth over every direction. Value minimizes `⟨ε, δ⟩`, the error *projected onto
the reference's mispricing*. These coincide only if the residual error happens
to avoid the `δ` direction. Calibrating in a direction the reference *shares*,
or one orthogonal to `δ`, is wasted effort for trading.

## 4. The metric: Edge-Alignment

Replace the latent `π` with the observed `r`, which is unbiased since
`E[r] = π`.

```python
from bracketlearn.score import edge_alignment, value_report

ea = edge_alignment(q, m, r)        # mean over contracts of (q - m)(r - m)
```

`edge_alignment` is the un-thresholded, every-contract expected betting PnL. It
scores *every* contract rather than only those clearing a trade threshold, so it
has far more statistical power than a thresholded, costed PnL. That power helps
on short windows. It is the value sibling of `brier_bracket`. Brier measures
`‖q − r‖`, and EA measures the alignment of `q − m` with `r − m`.

> **EA is frictionless.** It is linear in the edge, so it rewards every more
> over-confident forecast. That is correct for a fee-free, proportional-bet
> world, but it means EA alone will recommend over-tilting. With per-trade fees
> the objective becomes a *deductible* `E[(|δ| − fee)₊]` and grows an interior
> optimum. Use `edge_alignment_costed` for the deploy decision. See
> [value with fees](value_with_fees.md).

### The A − B split: attributing a change in value to its cause

`value_report` returns EA together with its exact additive decomposition, with no
latent `π` required, by the identity `(q−m)(r−m) = (r−m)² − (r−q)(r−m)`.

```python
rep = value_report(q, m, r)
# {'EA', 'A_reference_mse', 'B_non_orthogonality',
#  'align_corr', 'shared_bias_slope', 'n_contracts'}
```

* **`A = mean (r − m)²`** is the reference's mean-squared error, the amount of
  mispricing *available*. It is outside your control.
* **`B = mean (r − q)(r − m)`** is the co-projection of your error onto the
  reference's. It measures how much of the available mispricing you *fail* to
  capture because your errors coincide with the reference's.

`EA = A − B`. When EA moves across models or regimes, `ΔEA = ΔA − ΔB` attributes
the move. A fall in `A` means the reference got more efficient, so there is less
to capture and the model is not at fault. A rise in `B` means the forecast lost
orthogonality, with `q → m` where it is wrong, which is a model problem that can
be fixed. Brier cannot give this attribution, since it sees only `‖ε‖²` and is
blind to both `‖δ‖²` and the alignment.

Two normalized companions come along. `align_corr = corr(q − m, r − m)` is the
cosine between your edge and the reference's realized error, with `→ 0` the
shared-bias limit. `shared_bias_slope` is the OLS slope of your error `q − r` on
the reference's error `m − r`, and a large positive value means edge is
forfeited to blind spots shared with the reference.

## 5. A benign demonstration: accuracy and value disagree

This toy, the `_toy` helper in `tests/test_value_metrics.py`, builds a world with
two independent drivers. The reference sees only the dominant one. One candidate
forecast knows that dominant driver, so it is accurate but its edge sits in
already-priced territory. The other knows only the orthogonal driver, so it is
less accurate but its edge is un-priced.

```python
import numpy as np
from bracketlearn.score import edge_alignment, edge_alignment_corr

sigmoid = lambda x: 1 / (1 + np.exp(-x))
rng = np.random.default_rng(7)
n = 40_000
s1, s2 = rng.normal(0, 1.5, n), rng.normal(0, 1.5, n)   # independent drivers
pi = sigmoid(1.1 * s1 + 0.7 * s2)                        # s1 dominates the truth
r = (rng.uniform(size=n) < pi).astype(float)

m      = np.clip(sigmoid(0.9 * s1), 1e-4, 1 - 1e-4)      # market: sees only s1
q_acc  = np.clip(sigmoid(1.1 * s1 + rng.normal(0, .08, n)), 1e-4, 1 - 1e-4)  # knows s1
q_orth = np.clip(sigmoid(0.7 * s2 + rng.normal(0, .08, n)), 1e-4, 1 - 1e-4)  # knows s2

brier = lambda q: np.mean((q - r) ** 2)
print(f"q_acc :  Brier {brier(q_acc):.4f}   EA {edge_alignment(q_acc, m, r):+.4f}")
print(f"q_orth:  Brier {brier(q_orth):.4f}   EA {edge_alignment(q_orth, m, r):+.4f}")
# q_acc :  Brier 0.1857   EA +0.0001      <- MORE accurate, ~zero value
# q_orth:  Brier 0.2263   EA +0.0347      <- LESS accurate, all the value
```

`q_acc` is more accurate, with a lower Brier, yet carries almost no edge, because
everything it knows the market already priced. `q_orth` is *less* accurate but
holds the information the market lacks, so its edge points where the market is
wrong. An independent thresholded, costed betting strategy agrees with EA here
rather than with Brier. Selecting on accuracy would have shipped the wrong
forecast.

## 5b. Real data: EMOS against a market reference

The construction in §5 is synthetic and chosen to separate the two metrics.
This section applies the same measurement to observed forecasts and observed
prices, where the separation does not reproduce.

[`bracketlearn/examples/value_vs_accuracy_weather.py`](https://github.com/FrederikBenirschke/bracket-learn/blob/main/bracketlearn/examples/value_vs_accuracy_weather.py)
fits EMOS on `bracketlearn/examples/data/weather_value_sample.parquet`, which
holds 5,429 station-days over 2026-03-17 to 2026-09-03 across 18 stations,
carrying multi-model ensemble mean and spread, realized temperatures, per-row
bracket grids, and a normalized reference price per bracket. It prices the fitted
distribution onto each row's grid with `dist.integrate` and scores it against
the reference under both metrics. The split is chronological, 60/40.

```
===== HIGH  (train 1737, test 1158) =====
  forecast                        Brier   EA x100
  reference (market)             0.1066    0.0000
  EMOS (raw)                     0.1257   -0.0487   <- less accurate, EA < 0
  EMOS + mean de-bias            0.1256   -0.0588   <- Brier falls, EA falls
  EMOS + edge-recal              0.1069   +0.0042   <- Brier falls, EA rises
  EA 95% CI (clustered by station-day, 1158 clusters / 6945
                       contracts): [-0.1708, +0.0779]  (crosses zero)
```

Three observations follow from the table.

* EMOS is less accurate than the reference, with Brier 0.126 against 0.107, so
  it is ranked below the market on a calibration criterion.
* Its Edge-Alignment is also negative, at -0.049 with `align_corr = -0.011`. The
  errors are not decorrelated from the reference's in a direction that would
  be exploitable, so this sample does not exhibit the case §2 and §3 describe,
  in which a less accurate forecast retains positive value.
* The two metrics nonetheless order the adjustments differently. The edge
  recalibration improves Brier to 0.1069, near the reference's 0.1066, and
  raises EA. The mean de-bias leaves Brier almost unchanged and lowers EA
  further. Accuracy and value do not covary in either direction, which is the
  property this guide is concerned with, and it is observable here without a
  positive-value instance.

The LOW side gives the same qualitative result, with EMOS raw EA -0.110, 95% CI
[-0.2662, +0.0464], and `align_corr = -0.023`.

### What changed, and why the older numbers are gone

An earlier version of this section reported HIGH EA `+0.4938` and read it as
real-data confirmation of the synthetic result. Those numbers came off a
fixture with two defects, in a file that was hand-built and never committed as
a script, so nothing could re-derive it. The defects were these.

* The ladder's open tails were flattened to finite sentinels.
* The fifth inner edge was written one degree low, collapsing one bracket to
  width 1 and shifting the next boundary.

Prices were intact, so the file looked right, but edges decide which bracket
the realized temperature fell in, and 162 of its 2,168 rows (7.5%) carried the
wrong outcome label. Repairing only the edges, holding rows and split fixed,
moves HIGH from `+0.4938` to `+0.3326` and LOW from `+1.2960` to `+0.5922`.
The rest of the move to today's negative figures is population. That fixture
was a subset, and the current one covers a longer window. The chronological
split, which replaces a random permutation that leaks across an autocorrelated
series, accounts for little, about `-0.199` to `-0.151` on matched data.

The fixture is now generated by a committed script that pulls from the source
pipeline's research API and asserts, per row, that tails stay open and inner
brackets are width-2. A `.provenance.json` sidecar records the source commit
and query.

> **Scope.** EA here is computed against the bid-ask midpoint of a real
> exchange's quotes, frictionless, not a tradeable price net of fees and
> spread. `ens_mean`/`ens_std` are a declared definition, the multi-model spread
> across all available sources, not one recovered from the older fixture, whose
> definition is unrecoverable. The numbers here are a new measurement rather
> than a correction of the old ones. The intervals above are percentile
> bootstraps clustered by station-day, since contracts on one ladder resolve
> off a single realized temperature. Both cross zero, so read these as "not
> distinguishable from zero on this evidence" rather than as measured
> negatives.
>
> An earlier caveat here claimed the *sign* of EMOS's EA was robust across
> splits. It was not. That claim was made on the corrupted fixture and is
> withdrawn.

## 6. Improving value: edge-recalibration

The principle is not to push `q → π`, which is calibration, but to push the
**edge `q − m`** to track the **realized mispricing `r − m`**. On data strictly
prior to the prediction, walk-forward and causal, fit the monotone map

```
h = isotonic regression of (r − m) on (q − m)      # h(e) ≈ E[r − m | edge e]
```

then set `q' = m + h(q − m)`, clipped to `(0,1)` and renormalized per event. `h`
amplifies edges that have historically predicted real mispricing and damps edges
that were noise or shared bias. Contrast this with PIT-recalibration
(`q' = g(CDF)`), which maximizes *calibration* and need not help value.

This step needs the reference prices at fit time and edges toward the trade
layer, so bracketlearn keeps it as a documented recipe rather than a core
pipeline stage. That is the same boundary that puts [trade decisions out of
scope](../index.md). The metrics that *grade* it, `edge_alignment` and
`value_report`, are in the library.

`h` is fit against the same realized outcomes it is scored on, so it overfits
readily, far more than a calibration map, which targets the smoother `π`. On
the small real sample of §5b the isotonic `h` *lowered* test EA rather than
raising it, by amplifying in-sample-only edges. Edge-recalibration earns its
keep only with enough data and strict walk-forward validation. Treat a
recalibration that wins in-sample as unproven until it holds out-of-window
(see the §5b "Honest caveats").

## 7. Relation to known theory

The structure is classical, and recognizing the lineage is the point.

* **Kelly / information theory.** Betting a model `q` against prices `m`, the
  expected log-growth of wealth is `D(π‖m) − D(π‖q)`, that is, how far the
  reference is from truth minus how far you are, in KL divergence. Wealth grows
  if and only if you are closer to truth than the reference. The inner product
  `⟨q−m, π−m⟩` is its second-order Taylor expansion for small mispricings.
  (Kelly 1956; Cover & Thomas 2006, ch. 6.)
* **Active portfolio management.** Grinold's Fundamental Law reads
  `IR ≈ IC · √breadth`, with `IC = corr(forecast − benchmark, realized −
  benchmark)`. Swap the benchmark for the reference price and IC *is* the
  normalized EA, `align_corr`. The *rank* form of IC discards the magnitude and
  sizing the law needs, so EA is the covariance form rather than a rank
  correlation. (Grinold 1989; Grinold & Kahn 2000.)
* **Forecast verification.** Meteorology long ago separated a forecast's
  *quality*, its accuracy under proper scores, from its *value* to a
  decision-maker, defined relative to a reference forecast and a decision
  structure. Here that structure is "relative to the reference price, for a
  bet." (Murphy 1993; Murphy 1977; Richardson 2000.)
* **Market efficiency.** Shared bias is Grossman & Stiglitz (1980). A price
  aggregates common information, so the only exploitable signal is information
  orthogonal to it.

## API summary

| function | grades | input |
|---|---|---|
| `edge_alignment(q, m, r)` | value (scalar EA) | flat arrays over contracts |
| `edge_alignment_corr(q, m, r)` | normalized value (`corr`) | flat arrays |
| `shared_bias_slope(q, m, r)` | shared-bias diagnostic | flat arrays |
| `value_report(q, m, r)` | EA + A/B split + diagnostics | flat arrays |
| `edge_alignment_bracket(contracts, reference, edges, y)` | value of a ladder | `ContractForecast` + reference + edges |
| `value_report_bracket(contracts, reference, edges, y)` | full report for a ladder | `ContractForecast` + reference + edges |

`reference` is the quoted or baseline price for the same contracts, given as a
`ContractForecast` or a raw array matching `contracts.fair_price`.
