# Accuracy vs value: scoring a price against a reference

The proper-scoring rules in the [scoring guide](scoring.md) (CRPS, log-score,
Brier, log-loss) all answer one question: **is my price close to the
outcome?** That is *accuracy*. A trader on a prediction market asks a second
question: **is my price more valuable than the one already quoted?** That is
*value*. Value grades your price against a **reference price** `m` (a market
quote, a consensus, any baseline forecast) instead of against the truth.

The two questions have different answers. **A more accurate forecast can be
worth less to trade.** The [`score`](../api/score) module ships the metrics that
measure value: `edge_alignment`, `value_report`, and their bracket-ladder
wrappers.

> This sits one step before the [trade-decision layer that bracketlearn leaves
> to you](../index.md). Value is still *scoring*: it grades prices, it does not
> size positions. Unlike calibration, it grades them the way a trader cares
> about.

## 1. The PnL of a price

For one binary contract, write `q` for your price of YES, `m` for the reference
price, `r ∈ {0,1}` for the realized outcome, and `π` for the (latent) true
probability, so `E[r] = π`.

Buy YES at price `m`: you pay `m`, collect `1` if it occurs. Acting on your edge
`q − m` (buy when positive, sell when negative), sized by the edge, the expected
profit on one contract is

```
E[PnL] = (q − m)(π − m)
```

Read the formula: **the profit is the reference's mispricing `(π − m)`. Your
price only sets the direction and size of the bet.** If the reference is already
correct (`m = π`), no price `q` earns anything. Summed over many contracts, total
PnL is an inner product:

```
PnL ≈ ⟨ q − m , π − m ⟩
```

## 2. Decomposing the PnL

Let `δ = π − m` (the reference's mispricing, what you want to capture) and
`ε = π − q` (your error vs the truth). Substituting `q − m = δ − ε`:

```
PnL ≈ ‖δ‖²  −  ⟨ ε , δ ⟩
       │          │
       │          └─ your error projected onto the mispricing
       └─ the inefficiency available in the reference
```

Three consequences:

1. **No inefficiency, no profit.** If `‖δ‖² = 0` (the reference is right), PnL is
   zero for *any* forecast. You cannot out-predict a correct price.
2. **Most error is free.** You only lose the part of your error that *aligns
   with the mispricing*. Error where the reference is already right (`δ ≈ 0`)
   costs nothing. Nobody trades there.
3. **The shared-bias trap.** If your error tracks the reference's, `ε → δ`, then
   `q = π − ε → π − δ = m`: your price collapses onto the reference where
   it is most wrong. A more accurate forecast that shares the reference's blind
   spots is worthless for trading.

> **The price that makes money holds errors orthogonal to the reference's
> mispricing: accurate where the reference is wrong, free to be sloppy where it
> is right.** This is the Grossman–Stiglitz point in microcosm (§7): a price
> aggregates common information, so the only exploitable signal is information
> *orthogonal* to it.

## 3. Calibration ≠ value

Calibration, log-loss, CRPS all minimize `‖ε‖`: the closeness of `q` to truth
over every direction. Value minimizes `⟨ε, δ⟩`: your error *projected onto the
reference's mispricing*. These coincide only if your residual error happens to
avoid the `δ` direction. Calibrating in a direction the reference *shares* (or
that is orthogonal to `δ`) is wasted effort for trading.

## 4. The metric: Edge-Alignment

Replace the latent `π` with the observed `r` (unbiased, `E[r] = π`):

```python
from bracketlearn.score import edge_alignment, value_report

ea = edge_alignment(q, m, r)        # mean over contracts of (q - m)(r - m)
```

`edge_alignment` is the un-thresholded, every-contract expected betting PnL. It
scores *every* contract (not just the ones that clear a trade threshold), so it
has far more statistical power than a thresholded, costed PnL. That power helps
on short windows. It is the **value** sibling of `brier_bracket`: Brier measures `‖q − r‖`,
EA measures the alignment of `q − m` with `r − m`.

> **EA is frictionless: linear in your edge, so it rewards every more
> over-confident forecast.** That is correct for a fee-free, proportional-bet
> world, but it means EA alone will tell you to over-tilt. With per-trade fees
> the objective becomes a *deductible* `E[(|δ| − fee)₊]` and grows an interior
> optimum; use `edge_alignment_costed` for the deploy decision. See
> [value with fees](value_with_fees.md).

### The A − B split: attributing a change in value to its cause

`value_report` returns EA together with its exact additive decomposition, with no
latent `π` required, by the identity `(q−m)(r−m) = (r−m)² − (r−q)(r−m)`:

```python
rep = value_report(q, m, r)
# {'EA', 'A_reference_mse', 'B_non_orthogonality',
#  'align_corr', 'shared_bias_slope', 'n_contracts'}
```

* **`A = mean (r − m)²`**: the reference's mean-squared error. How much
  mispricing is *available*. Outside your control.
* **`B = mean (r − q)(r − m)`**: co-projection of your error onto the
  reference's. How much of the available mispricing you *fail* to capture
  because your errors coincide with the reference's.

`EA = A − B`. When EA moves across models or regimes, `ΔEA = ΔA − ΔB` tells you
*why*: `A` fell ⇒ the reference got more efficient (less to capture, not your
fault); `B` rose ⇒ your forecast lost orthogonality (`q → m` where it is wrong,
a model problem you can fix). This attribution is what Brier
cannot give: Brier sees only `‖ε‖²`, blind to `‖δ‖²` and to the alignment.

Two normalized companions come along: `align_corr = corr(q − m, r − m)` (the
cosine between your edge and the reference's realized error; `→ 0` is the
shared-bias limit) and `shared_bias_slope` (the OLS slope of your error `q − r`
on the reference's error `m − r`; a large positive value means you forfeit edge
to blind spots you share with the reference).

## 5. A benign demonstration: accuracy and value disagree

This toy (the `_toy` helper in `tests/test_value_metrics.py`) builds a world with
two independent drivers. The **reference sees only the dominant one**; one
candidate forecast knows that dominant driver (accurate, but its edge sits in
already-priced territory), the other knows only the orthogonal driver (less
accurate, but its edge is un-priced).

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

`q_acc` is more accurate (lower Brier) yet carries almost no edge: everything it
knows, the market already priced. `q_orth` is *less* accurate but holds the
information the market lacks, so its edge points where the market is wrong. An
independent thresholded, costed betting strategy agrees with EA here, not with
Brier. Selecting on accuracy would have shipped the wrong forecast.

## 5b. The same thing on real data: EMOS vs a market

The synthetic toy is rigged to make the point. Real forecasts and real prices
test it. [`bracketlearn/examples/value_vs_accuracy_weather.py`](https://github.com/FrederikBenirschke/bracket-learn/blob/main/bracketlearn/examples/value_vs_accuracy_weather.py)
fits EMOS on an anonymized weather sample
(`bracketlearn/examples/data/weather_value_sample.parquet`: ensemble mean/spread, realized
temperatures, per-row bracket grids, and a normalized reference price per
bracket; no venue, station, or date), prices it onto each row's grid with
`dist.integrate`, and scores it against the reference both ways:

```
===== HIGH  (train 901, test 602) =====
  forecast                        Brier   EA ×100
  reference (market)             0.1121    0.0000
  EMOS (raw)                     0.1220   +0.4938   <- less accurate than market, EA > 0
  EMOS + mean de-bias            0.1232   +0.4586   <- value falls vs raw
  EMOS + edge-recal              0.1109   +0.1288   <- best Brier, value collapses
```

Read the HIGH block top to bottom. It is the whole guide in four rows:

* **EMOS is *less* accurate than the market** (Brier 0.122 vs 0.112). On a
  calibration scoreboard EMOS loses.
* **EMOS is tradeable anyway** (EA `+0.494 > 0`). Its errors are decorrelated
  from the market's, so its edge points where the market is wrong, the case
  §2–3 says calibration cannot see.
* **Calibrating it harder does not help.** A mean de-bias toward the truth
  *lowers* value. An edge-recalibration produces the **best Brier of all
  four rows** (0.111, matching the market) while its value **collapses** (from
  `+0.494` to `+0.129`). Closeness to the outcome and value-vs-a-reference are
  different axes.

(The LOW side lands the other way on accuracy: EMOS there is *more* accurate
than the market *and* positive-EA, yet the same two "fixes" still cut its value.
The robust, side-independent lesson is the last bullet, not the sign of the
accuracy gap.)

> Honest caveats. EA here is on a normalized reference price, not a tradeable
> order book net of fees; a positive EA is necessary, not sufficient, for live
> profit. The exact numbers wobble with the split; what is robust across splits
> is the *sign* of EMOS's EA and that both calibration "fixes" reduce it. An
> earlier version tried σ-recalibration; that "worked" only when EMOS was misfit
> to a constant σ. Fit with `fit_method="crps_nelder_mead"`, EMOS is not
> over-dispersed here and a σ-scale does nothing. Verify a fix is real before
> believing it.

## 6. Improving value: edge-recalibration

The principle says: don't push `q → π` (calibration); push the **edge `q − m`**
to track the **realized mispricing `r − m`**. On data strictly prior to the
prediction (walk-forward, causal), fit the monotone map

```
h = isotonic regression of (r − m) on (q − m)      # h(e) ≈ E[r − m | edge e]
```

then set `q' = m + h(q − m)` (clip to `(0,1)`, renormalize per event). `h`
amplifies edges that have historically predicted real mispricing and damps edges
that were noise or shared bias. Contrast with PIT-recalibration (`q' = g(CDF)`),
which maximizes *calibration* and need not help value.

This step needs the reference prices at fit time and edges toward the trade
layer, so bracketlearn keeps it as a documented recipe rather than a core
pipeline stage, the same boundary that puts [trade decisions out of
scope](../index.md). The metrics that *grade* it (`edge_alignment`,
`value_report`) are in the library.

`h` is fit against the same realized outcomes it is scored on, so it **overfits
readily**, far more than a calibration map, which targets the smoother `π`. On
the small real sample of §5b the isotonic `h` *lowered* test EA rather than
raising it (it amplified in-sample-only edges). Edge-recalibration earns its
keep only with enough data and strict walk-forward validation; treat a
recalibration that "wins" in-sample as unproven until it holds out-of-window
(see the §5b "Honest caveats").

## 7. Relation to known theory

The structure is classical; recognizing the lineage is the point.

* **Kelly / information theory.** Betting your model `q` against prices `m`, the
  expected log-growth of wealth is `D(π‖m) − D(π‖q)`: *(how far the reference is
  from truth) − (how far you are)* in KL divergence. You grow iff you are closer
  to truth than the reference. The inner product `⟨q−m, π−m⟩` is its
  second-order Taylor expansion for small mispricings. (Kelly 1956; Cover &
  Thomas 2006, ch. 6.)
* **Active portfolio management.** Grinold's Fundamental Law, `IR ≈ IC · √breadth`,
  with `IC = corr(forecast − benchmark, realized − benchmark)`. Swap benchmark →
  reference price and IC *is* the normalized EA (`align_corr`). The *rank* form
  of IC discards the magnitude/sizing the law needs, which is why EA is the
  covariance form, not a rank correlation. (Grinold 1989;
  Grinold & Kahn 2000.)
* **Forecast verification.** Meteorology long ago separated a forecast's
  *quality* (accuracy: proper scores) from its *value* to a decision-maker,
  defined relative to a reference forecast and a decision structure:
  "relative to the reference price, for a bet." (Murphy 1993; Murphy 1977;
  Richardson 2000.)
* **Market efficiency.** The shared-bias trap is Grossman & Stiglitz (1980): a
  price aggregates common information, so the only exploitable signal is
  information orthogonal to it.

## API summary

| function | grades | input |
|---|---|---|
| `edge_alignment(q, m, r)` | value (scalar EA) | flat arrays over contracts |
| `edge_alignment_corr(q, m, r)` | normalized value (`corr`) | flat arrays |
| `shared_bias_slope(q, m, r)` | shared-bias diagnostic | flat arrays |
| `value_report(q, m, r)` | EA + A/B split + diagnostics | flat arrays |
| `edge_alignment_bracket(contracts, reference, edges, y)` | value of a ladder | `ContractForecast` + reference + edges |
| `value_report_bracket(contracts, reference, edges, y)` | full report for a ladder | `ContractForecast` + reference + edges |

`reference` is the quoted/baseline price for the same contracts: a
`ContractForecast` or a raw array matching `contracts.fair_price`.
