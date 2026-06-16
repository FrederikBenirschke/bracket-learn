# Value with fees: from an inner product to a deductible

The [value guide](value_vs_accuracy.md) derives Edge-Alignment in a
**frictionless** world: bet a size proportional to your edge `e = q − m`, pay no
fee, and the expected payoff of one contract is `e · δ`, where `δ = E[r − m | x]`
is the true mispricing. Edge-Alignment is the sample estimate:

```
EA = mean (q − m)(r − m)
```

That world has a sharp property: **EA is linear in your
edge.** Write the edge as `k · δ̂` (k = how hard you tilt toward the mispricing).
Then `EA = k · E[δ̂ · δ]`: a straight line in `k`, rising forever. If `q = π`
(calibrated) the edge is `δ` and `EA = ‖δ‖²`; if `q = m + 2δ` (deliberately
*over-confident*) then `EA = 2‖δ‖²`. EA pays you more for exaggerating.
There is no interior optimum: the EA-maximizing forecast is infinitely
over-confident in the `δ` direction.

This is *correct* for the frictionless, proportional-bet strategy, not a bug in
EA. With no fee and proportional sizing, over-stating your edge just leverages a
correct-direction bet, and nothing penalizes leverage. **In a frictionless
market, EA is the whole story and you should tilt as hard as you can.**

Real venues are not frictionless. The moment you add a per-trade cost, the
objective stops being linear. Almost everything above flips.

## A fee turns the inner product into a hinge

Take the realistic strategy: trade **one unit** of a contract when your edge
clears a gate, `|q − m| > τ`, in the direction `sign(q − m)`, and pay a fee `φ`
per contract traded. The realized payoff per contract is

```
  sign(q − m) · (r − m) − φ      if |q − m| > τ
  0                              otherwise
```

(`edge_alignment_costed(q, m, r, fee=φ, tau=τ)` computes exactly this.) Take the
expectation given features, with `δ = E[r − m | x]`. On a traded contract the
expected payoff is `sign(q − m) · δ − φ`. Now ask what the *best possible*
forecast achieves: the one that trades in the right direction when it is
worth it. It trades iff `|δ| > φ`, in direction `sign(δ)`, and collects `|δ| − φ`.
So the best achievable value is

```
  V* = E[ (|δ| − φ)₊ ]          ( x₊ = max(x, 0) )
```

A **deductible on the mispricing.** You are paid only where the true mispricing
exceeds the fee, and only the part above the fee. The inner product `⟨e, δ⟩`
became a hinge `(|δ| − φ)₊`. Three consequences follow, and each one reverses a
frictionless instinct.

### 1. Edge *magnitude* stops mattering; only sign and the gate do

For a unit bet the payoff depends on `q − m` only through (a) whether it clears
the gate and (b) its **sign**. Doubling your stated edge changes neither, *unless*
it (i) pushes a previously-skipped contract over the gate or (ii) flips a sign.
So over-confidence can no longer help you. It can only:

- trade **sub-fee junk**: contracts where `|δ| < φ`, which return `−φ` on
  average once you cross the gate, or
- **flip signs** on near-fair, noisy contracts and bet the wrong way.

The frictionless reward for exaggeration is gone; only downside remains.

### 2. The objective is sparse; volume-chasing loses

`(|δ| − φ)₊` is zero on every contract whose mispricing is smaller than the fee.
Most brackets contribute nothing. Value lives in the **few** brackets mispriced
by more than the fee. A forecast that "finds edge everywhere" is finding mostly
sub-fee edge, and paying `φ` to harvest it. This is why, when you tilt a blended
training objective toward EA (higher `λ`), the costed PnL **peaks at an interior
`λ` and then falls**: past the peak you are buying volume of progressively worse
trades. EA keeps climbing because it never charges you for them.

### 3. The gate should sit at the fee

The deductible says the trade rule is "trade iff `|δ| > φ`". So set `τ ≈ φ`:
trade only when your estimated edge clears the cost. With estimation noise, set
`τ` a little **above** `φ` (a safety margin), because a noisy edge that *just*
clears `φ` is, after shrinkage, probably not really there.

## Sizing with fees: the deadband

The unit bet is bang-bang. If instead you size continuously under a quadratic
risk penalty (`λ`) and a proportional fee, you maximize `s·δ − φ|s| − ½λs²` over
the size `s`. The solution is the **soft-threshold** (the proximal operator of an
L1 penalty):

```
  s* = sign(δ̂) · (|δ̂| − φ)₊ / λ
```

Same deductible, now in the *size*: shrink your edge toward zero by the fee, and
bet zero inside a **deadband** of half-width `φ` around fair. This is the
fee-aware, risk-adjusted version of "bet proportional to conviction." Note `δ̂`
appears as a *magnitude* here. So once you size continuously, the **calibration
of your edge magnitude matters again**: calibration is irrelevant for direction
(which bracket to trade), but it governs sizing (how much). Fees and concave
sizing both re-introduce the calibration EA was free to ignore.

## What this changes in practice

| question | frictionless (EA) | with fees |
|---|---|---|
| objective | `⟨q − m, δ⟩` (linear) | `E[(\|δ\| − φ)₊]` (hinge / deductible) |
| what to maximize | edge **magnitude** in the `δ` direction | **sign** correctness on supra-fee brackets |
| best tilt `λ` | as hard as possible (`λ → ∞`) | **interior**, picked by costed value |
| where value lives | everywhere `δ ≠ 0` | only where `\|δ\| > φ` (sparse) |
| sizing | `∝ edge` | soft-threshold: `∝ sign(δ̂)(\|δ̂\| − φ)₊` |
| calibration | irrelevant (direction only) | **matters** (governs sizing) |

Concretely:

- **Scoring.** Use `edge_alignment` for *research power*: it scores every
  contract, has low variance, and ranks forecasts on short windows where the
  thresholded PnL is too noisy. Use `edge_alignment_costed` for the **deploy
  decision**: it is the metric that reflects what you will earn. A
  forecast can win on EA and lose on costed value; trust the costed one for go/no-go.
- **Training.** Train a blended objective `L = CE − λ·EA`, where **CE is the
  cross-entropy** (categorical log-loss, `−log q` on the realized bracket, the
  same calibration term `log_loss_bracket` reports) and `EA` is the value term.
  CE is strictly convex, so it supplies the curvature the linear EA term lacks
  and keeps the optimum bounded; `λ` is the tilt (`λ = 0` = pure calibration).
  This is implemented in `bracketlearn.value`: `BlendedBracketGBM` (LightGBM)
  and `BlendedBracketNet` (torch); see the [value-trainers
  guide](value_trainers.md). Then **select `λ` by costed value, never by EA**.
  EA, being frictionless, votes for more tilt (maximum over-confidence);
  the costed metric finds the interior
  `λ` where the marginal supra-fee edge you gain equals the marginal junk-trade
  fee you pay. The "more tilt is always better" reading is an artifact of scoring
  a fee'd strategy with a fee-free metric.
- **Gating and sizing.** Gate at `τ ≈ φ` (a touch above, for noise). If you size
  continuously, soft-threshold the edge by `φ` and let a risk budget `λ` set the
  scale.

## Honest caveats

- **Realized costed value is not monotone in the fee.** Raising `φ` (with the
  gate tied to it) drops trades, including *losing* ones, so a higher fee can
  *raise* a fixed forecast's realized PnL by gating out bad trades. Only the
  oracle `E[(|δ| − φ)₊]` is monotone in `φ`. Don't read fee-sensitivity of a
  realized backtest as the deductible curve.
- **`φ` is venue- and bracket-specific** (maker vs taker, half-spread by price
  band, size). Use the fee the contract will actually pay, not a flat constant.
- **The deductible bites hardest exactly where the edge is thin.** Sub-fee
  mispricings are common; a real edge has to clear the fee with margin to be
  worth trading at all. That is why most "positive-EA" forecasts are not
  profitable net of costs. The costed metric, not EA, is the gate.
