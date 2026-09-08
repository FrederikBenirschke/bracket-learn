# Scoring & metrics

bracketlearn ships proper-scoring rules per backing, plus a helper that
collapses any distribution to a point forecast for benchmarking against
classical regression metrics.

## Distribution-level metrics

`PipelineResult.score(y, metrics=[...])` dispatches per backing.

| metric | parametric normal | parametric mixture-normal | quantile | bracket |
|---|---|---|---|---|
| `crps` | closed-form (σ·[z(2Φ(z)−1) + 2φ(z) − 1/√π]) | Monte-Carlo energy form (2000 samples by default) | pinball-trapezoid | piecewise-uniform CDF |
| `log_score` | closed-form | closed-form | piecewise-linear CDF → constant density per bin | uniform-in-bin density |
| `pit_mean` / `pit_std` | closed-form | numerical CDF | linear interpolation | linear interpolation |

Every backing in the v0.3+ metric set has a working definition, so the table
never returns `nan`.

## Bracket-contract metrics

When `edges=...` is passed, `score()` builds the per-row bracket ladder
internally per stage.

| metric | what it measures |
|---|---|
| `log_loss_bracket` | mean -log P(realised bracket) under predicted bracket distribution |
| `brier_bracket` | mean squared error between one-hot realised bracket and predicted probs |

`edges` takes any of the three shapes `dist.integrate` accepts, namely a shared
`(B+1,)` vector, a dense `(N, B+1)` grid, or a ragged per-row sequence. On a
venue whose ladder rotates, pass the per-row edges the ladder was priced with.
Both scorers use `edges` to decide which bracket the outcome fell in, so one
row's vector applied to every row assigns most outcomes to a bracket that row
never listed. That produced a plausible wrong number until v0.8, with a measured
Brier of 0.8904 against a correct 0.7343 on a 40-row rotating ladder. It now
raises.

## Confidence intervals

Every metric above returns a point estimate. `score.bootstrap_ci` wraps any of
them.

```python
from bracketlearn.score import bootstrap_ci, edge_alignment

ci = bootstrap_ci(edge_alignment, q, m, r, cluster=day_id, n_boot=2000)
# {'point': ..., 'lo': ..., 'hi': ..., 'se': ..., 'n_clusters': ...}
```

Pass `cluster` whenever contracts share an outcome. Every contract on one
bracket ladder resolves off the same realized value, so resampling contracts
independently treats dependent draws as independent evidence. The label is
usually a station-day or event id. Below about 20 clusters the percentile
interval under-covers materially, with a measured 0.74 actual coverage at K=3
against a nominal 0.95, so coarsening clusters in the name of being conservative
does the opposite. `bootstrap_ci` warns in that regime and raises when every row
falls in one cluster.

## Point-forecast helper

```python
from bracketlearn.score import to_point

mu_mean   = to_point(dist, how="mean")     # E[Y | x]
mu_median = to_point(dist, how="median")   # F⁻¹(0.5)
mu_mode   = to_point(dist, how="mode")     # highest-density / highest-weight
```

This is useful for benchmarking against classical sklearn regressors.

```python
from sklearn.metrics import mean_squared_error

y_oof = y[dist.ids.astype(int)]
mu_hat = to_point(dist, how="mean")
print("RMSE:", np.sqrt(mean_squared_error(y_oof, mu_hat)))
```

All three notebooks under `bracketlearn/notebooks/` end with a
point-forecast leaderboard comparing the probabilistic models' means
against `sklearn.linear_model.Ridge` and `LGBMRegressor`.
