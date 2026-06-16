# Training for value: the `bracketlearn.value` trainers

Every trainer in [Step 1](../index.md) optimizes a **proper score**: it makes
`q` close to the truth. The [value guide](value_vs_accuracy.md) showed that
closeness to truth is not the same as value-vs-a-reference, and the [fees
guide](value_with_fees.md) showed how to *score* for value. `bracketlearn.value`
closes the loop: two trainers that **optimize value directly**, by the blended
objective

```
L = CE − λ·EA
```

`CE` is the cross-entropy (calibration); `EA = (q − m)(r − m)` is the
reference-relative value (the betting payoff against price `m`). `λ ≥ 0` is the
tilt: `λ = 0` is a pure-calibration bracket model; larger `λ` tilts toward
capturing the reference's mispricing. CE stays in the loss to supply curvature;
the `EA` term alone is linear, and a model trained on it overfits with
over-confidence.

These live in `bracketlearn.value`, not `bracketlearn.trainers`, on purpose:
they need the **reference price `m` at fit time**, which is a step past pure
forecasting. (Prediction stays in the core; "beat *this* price" is its own
layer.) `m` is used **only in the loss**: `predict_dist` needs no market data.

## The two engines

Both are bracket-native (per-`(row, bracket)` binary, like `CumulativeBinary`)
and share the contract `fit(X, y, *, ids, brackets_by_id, reference_by_id)` →
`predict_dist(X, *, ids, timestamps, brackets_by_id)` → a `BracketForecast`.
`brackets_by_id` maps `id → 1-D edge array`; `reference_by_id` maps
`id → 1-D price array` (length `len(edges) − 1`).

### The data contract (construction is hyperparameters only)

The constructor takes **only hyperparameters** (`lam` and engine knobs). The
per-row market data (the bracket ladders and reference prices) flows alongside
`X`/`y` at call time, keyed by id. `fit` and `predict_dist` *select* the subset
they need by the `ids` you hand them, so the dicts may cover more ids than any
one call. `reference_by_id` is needed only at fit (it feeds the loss); `predict`
takes only `brackets_by_id` (to assemble the dist). Under `WalkForward` you pass
the dicts once and they are forwarded **verbatim** to every fold:

```python
from bracketlearn import WalkForward

model = BlendedBracketGBM(lam=2.0)                 # hyperparameters only
result = WalkForward(cv="kfold", n_folds=5).fit_predict(
    model, X, y, ids=ids, timestamps=ts,
    brackets_by_id=bbi, reference_by_id=rbi)       # forwarded to each fold
oof = result.forecasts[model.name]                # out-of-fold value-tilted dist
```

| trainer | engine | objective |
|---|---|---|
| `BlendedBracketGBM` | LightGBM custom objective (`grad`, `hess`) | `L = CE − λ·EA` |
| `BlendedBracketNet` | torch MLP (autograd) | `L = CE − λ·EA` |

```python
from bracketlearn.value import BlendedBracketGBM, value_report_dist

# per-row bracket ladders and the market's price for each bracket
brackets_by_id = {i: edges_i for i, edges_i in enumerate(edges_per_row)}
reference_by_id = {i: market_prob_i for i, market_prob_i in enumerate(market_per_row)}

model = BlendedBracketGBM(lam=2.0)
model.fit(X_tr, y_tr, ids=ids_tr,
          brackets_by_id=brackets_by_id, reference_by_id=reference_by_id)
dist = model.predict_dist(X_te, ids=ids_te, timestamps=ts_te,
                          brackets_by_id=brackets_by_id)          # BracketForecast

# score the implied edge against the market in ONE call: value_report_dist does
# the per-row ragged flatten + renormalization for you (y is an array in dist row
# order, or a dict by id). fee= adds the costed metrics under costed_* keys.
rep = value_report_dist(dist, reference_by_id, y_te, fee=0.02)
print(rep["EA"], rep["costed_mean_pnl"])     # value (fee-free) and value net of fees
```

`edge_alignment_dist(dist, reference_by_id, y)` returns just the scalar EA. If you
already hold flat `(q, m, r)` arrays, the lower-level `edge_alignment` /
`edge_alignment_costed` / `value_report` in `bracketlearn.score` take those
directly.

`BlendedBracketNet` swaps the engine (`hidden`, `epochs`, `lr`, `ea_scale`)
behind the same call. The shared math is exposed too: `make_lgb_objective`
(the LightGBM `(grad, hess)` closure), `blended_grad_hess`, and `blended_loss`.

### `λ` means the same thing in both engines

The two engines treat the EA gradient `λ·(r − m)·q(1−q)` differently: LightGBM's
Newton step divides by the Hessian `q(1−q)`, cancelling it, so its effective EA
update is `≈ λ·(r − m)`; the torch net does plain gradient descent and keeps the
`q(1−q)` factor, which suppresses its EA term by `≈ E[q(1−q)]`. So that `λ`
transfers across engines, `BlendedBracketNet` rescales its EA term by
`ea_scale = 1 / mean(m(1−m))`, a quantity **derived from the reference prices at
fit time** (`ea_scale_for_reference`) rather than a hand-tuned constant, and
records it on `model.ea_scale_`. Pass an explicit `ea_scale=` to override. This matches the
gradient scale at initialization (`q ≈ m`); as `q` moves during training the
match is approximate, so still **select `λ` per-engine by costed value** below.

### Trained edge ≠ traded edge (raw vs renormalized `q`)

The objective acts on each **per-`(row, bracket)`
binary** independently: the EA term tilts the raw `q_b = σ(z_b)` toward the
mispricing `q_b − m_b`. But `predict_dist` **renormalizes** each row to sum to 1
before you trade it (`q_b ← q_b / Σ_b q_b`). So the edge the loss optimized
(`raw q_b − m_b`) is *not* the edge you trade (`renormalized q_b − m_b`).

The gap is small but has two practical consequences:

- The training-time EA you'd compute on raw scores will be a little higher than
  the EA you score on the predicted distribution. Evaluate value on the
  **renormalized** output of `predict_dist` (as the demo does); that is the edge
  you can take.
- Because the renormalizer couples brackets within a row, pushing one bracket's
  raw probability up to chase its mispricing slightly deflates the others. The
  net per-row effect is what the scored EA captures; trust it over the per-binary
  training objective.

## Choosing `λ`: by costed value, never by EA

EA is fee-free and **linear in the edge**, so it rises with `λ`; it will tell
you to tilt harder. With real fees the objective is a deductible (fees guide),
so the **costed** value peaks at an interior `λ` and then falls (you start
trading sub-fee junk). So:

```python
from bracketlearn.value import BlendedBracketGBM, value_report_dist

def costed_value(lam):
    model = BlendedBracketGBM(lam=lam)
    model.fit(Xtr, ytr, ids=itr, brackets_by_id=bbi, reference_by_id=rbi)
    dist = model.predict_dist(Xva, ids=iva, timestamps=tva, brackets_by_id=bbi)
    return value_report_dist(dist, rbi, y_va, fee=your_fee)["costed_mean_pnl"]

best = max([0.0, 1.0, 2.0, 4.0, 8.0], key=costed_value)
```

Validate walk-forward with a day-clustered CI, as for any other model
selection. `λ = 0` (a pure-CE bracket model) is the honest baseline to beat; on
hard markets the costed-optimal `λ` can be `0`, a valid result: there is no
tradeable tilt after costs.

## Where it sits

`bracketlearn.value` is scoring + training for value; it stops at a
value-tilted **price**. Turning that price into a **position** (size, gate,
Kelly, hedging) remains [out of scope](../index.md), your trading layer.
