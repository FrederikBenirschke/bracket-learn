# Value with fees: from an inner product to a deductible

The [value guide](value_vs_accuracy.md) derives Edge-Alignment in a
**frictionless** world. Bet a size proportional to the edge `e = q − m`, pay no
fee, and the expected payoff of one contract is `e · δ`, where
`δ = E[r − m | x]` is the true mispricing. Edge-Alignment is the sample estimate.

```
EA = mean (q − m)(r − m)
```

That world has a sharp property. EA is linear in the edge. Write the edge as
`k · δ̂`, where `k` measures how hard you tilt toward the mispricing. Then
`EA = k · E[δ̂ · δ]`, a straight line in `k` that rises forever. If `q = π`, so
the forecast is calibrated, the edge is `δ` and `EA = ‖δ‖²`. If `q = m + 2δ`,
deliberately *over-confident*, then `EA = 2‖δ‖²`. EA pays more for exaggerating.
There is no interior optimum, and the EA-maximizing forecast is infinitely
over-confident in the `δ` direction.

This is *correct* for the frictionless, proportional-bet strategy and is not a
defect in EA. With no fee and proportional sizing, over-stating the edge merely
leverages a correct-direction bet, and nothing penalizes leverage. In a
frictionless market, EA is the whole story and the right move is to tilt as hard
as possible.

Real venues are not frictionless. Once a per-trade cost is added, the objective
stops being linear, and almost everything above reverses.

## A fee turns the inner product into a hinge

Take the realistic strategy. Trade **one unit** of a contract when the edge
clears a gate, `|q − m| > τ`, in the direction `sign(q − m)`, and pay a fee `φ`
per contract traded. The realized payoff per contract is

```
  sign(q − m) · (r − m) − φ      if |q − m| > τ
  0                              otherwise
```

`edge_alignment_costed(q, m, r, fee=φ, tau=τ)` computes exactly this. Take the
expectation given features, with `δ = E[r − m | x]`. On a traded contract the
expected payoff is `sign(q − m) · δ − φ`. Now ask what the *best possible*
forecast achieves, namely the one that trades in the right direction when it is
worth it. It trades if and only if `|δ| > φ`, in direction `sign(δ)`, and
collects `|δ| − φ`. So the best achievable value is

```
  V* = E[ (|δ| − φ)₊ ]          ( x₊ = max(x, 0) )
```

This is a deductible on the mispricing. You are paid only where the true
mispricing exceeds the fee, and only for the part above the fee. The inner
product `⟨e, δ⟩` has become a hinge `(|δ| − φ)₊`. Three consequences follow, and
each one reverses a frictionless instinct.

### 1. Edge *magnitude* stops mattering, and only sign and the gate do

For a unit bet the payoff depends on `q − m` only through whether it clears the
gate and through its **sign**. Doubling a stated edge changes neither, unless it
pushes a previously-skipped contract over the gate or flips a sign. So
over-confidence can no longer help. It can only do two things.

- Trade **sub-fee junk**, contracts where `|δ| < φ`, which return `−φ` on
  average once the gate is crossed.
- **Flip signs** on near-fair, noisy contracts and bet the wrong way.

The frictionless reward for exaggeration is gone, and only the downside remains.

### 2. The objective is sparse, and volume-chasing loses

`(|δ| − φ)₊` is zero on every contract whose mispricing is smaller than the fee.
Most brackets contribute nothing. Value lives in the few brackets mispriced
by more than the fee. A forecast that finds edge everywhere is finding mostly
sub-fee edge and paying `φ` to harvest it. For this reason, tilting a blended
training objective toward EA with a higher `λ` makes the costed PnL peak at an
interior `λ` and then fall. Past the peak the tilt buys volume of progressively
worse trades. EA keeps climbing because it never charges for them.

### 3. The gate should sit at the fee

The deductible says the trade rule is "trade iff `|δ| > φ`", so set `τ ≈ φ` and
trade only when the estimated edge clears the cost. With estimation noise, set
`τ` a little above `φ` as a safety margin, because a noisy edge that *just*
clears `φ` is, after shrinkage, probably not really there.

## Sizing with fees: the deadband

The unit bet is bang-bang. If instead the size is chosen continuously under a
quadratic risk penalty (`λ`) and a proportional fee, the objective is to maximize
`s·δ − φ|s| − ½λs²` over the size `s`. The solution is the **soft-threshold**,
the proximal operator of an L1 penalty.

```
  s* = sign(δ̂) · (|δ̂| − φ)₊ / λ
```

The same deductible now appears in the *size*. Shrink the edge toward zero by
the fee and bet zero inside a **deadband** of half-width `φ` around fair. This is
the fee-aware, risk-adjusted version of "bet proportional to conviction". Here
`δ̂` appears as a *magnitude*. Once sizing is continuous, the calibration of the
edge magnitude matters again. Calibration is irrelevant for direction, meaning
which bracket to trade, but it governs sizing, meaning how much. Fees and
concave sizing both re-introduce the calibration EA was free to ignore.

## What this changes in practice

| question | frictionless (EA) | with fees |
|---|---|---|
| objective | `⟨q − m, δ⟩` (linear) | `E[(\|δ\| − φ)₊]` (hinge / deductible) |
| what to maximize | edge **magnitude** in the `δ` direction | **sign** correctness on supra-fee brackets |
| best tilt `λ` | as hard as possible (`λ → ∞`) | **interior**, picked by costed value |
| where value lives | everywhere `δ ≠ 0` | only where `\|δ\| > φ` (sparse) |
| sizing | `∝ edge` | soft-threshold: `∝ sign(δ̂)(\|δ̂\| − φ)₊` |
| calibration | irrelevant (direction only) | **matters** (governs sizing) |

The practical consequences follow.

- **Scoring.** Use `edge_alignment` for *research power*. It scores every
  contract, has low variance, and ranks forecasts on short windows where the
  thresholded PnL is too noisy. Use `edge_alignment_costed` for the deploy
  decision, since it is the metric that reflects realized earnings. A forecast
  can win on EA and lose on costed value. Trust the costed one for go/no-go.
- **Training.** Train a blended objective `L = CE − λ·EA`, where CE is the
  cross-entropy (categorical log-loss, `−log q` on the realized bracket, the
  same calibration term `log_loss_bracket` reports) and `EA` is the value term.
  CE is strictly convex, so it supplies the curvature the linear EA term lacks
  and keeps the optimum bounded. `λ` is the tilt, and `λ = 0` is pure
  calibration. This is implemented in `bracketlearn.value` as
  `BlendedBracketGBM` (LightGBM) and `BlendedBracketNet` (torch). See the
  [value-trainers guide](value_trainers.md). Then select `λ` by costed value
  rather than by EA. EA, being frictionless, votes for more tilt and maximum
  over-confidence. The costed metric finds the interior `λ` where the marginal
  supra-fee edge gained equals the marginal junk-trade fee paid. The reading
  that more tilt is always better is an artifact of scoring a fee'd strategy
  with a fee-free metric.
- **Gating and sizing.** Gate at `τ ≈ φ`, a touch above for noise. If sizing is
  continuous, soft-threshold the edge by `φ` and let a risk budget `λ` set the
  scale.

## Honest caveats

- **Realized costed value is not monotone in the fee.** Raising `φ`, with the
  gate tied to it, drops trades, including *losing* ones, so a higher fee can
  *raise* a fixed forecast's realized PnL by gating out bad trades. Only the
  oracle `E[(|δ| − φ)₊]` is monotone in `φ`. Do not read fee-sensitivity of a
  realized backtest as the deductible curve.
- **`φ` is venue- and bracket-specific**, varying with maker against taker,
  half-spread by price band, and size. Use the fee the contract will actually
  pay rather than a flat constant.
- **The deductible bites hardest exactly where the edge is thin.** Sub-fee
  mispricings are common, and a real edge has to clear the fee with margin to be
  worth trading at all. This is why most positive-EA forecasts are not
  profitable net of costs. The costed metric, not EA, is the gate.
