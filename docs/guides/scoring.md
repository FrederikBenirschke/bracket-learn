# Scoring & metrics

bracketlearn ships proper-scoring rules per backing plus a helper to
collapse any distribution to a point forecast (so you can benchmark
against classical regression metrics).

## Distribution-level metrics

`PipelineResult.score(y, metrics=[...])` dispatches per backing:

| metric | parametric normal | parametric mixture-normal | quantile | bracket |
|---|---|---|---|---|
| `crps` | closed-form (σ·[z(2Φ(z)−1) + 2φ(z) − 1/√π]) | Monte-Carlo energy form (2000 samples by default) | pinball-trapezoid | piecewise-uniform CDF |
| `log_score` | closed-form | closed-form | piecewise-linear CDF → constant density per bin | uniform-in-bin density |
| `pit_mean` / `pit_std` | closed-form | numerical CDF | linear interpolation | linear interpolation |

There are **no `nan` returns** for the v0.3+ metric set — every backing
has a definition that works.

## Bracket-contract metrics

When you pass `edges=...` (a shared 1-D edge vector; `score()` builds the
per-row bracket ladder internally per stage):

| metric | what it measures |
|---|---|
| `log_loss_bracket` | mean −log P(realised bracket) under predicted bracket distribution |
| `brier_bracket` | mean squared error between one-hot realised bracket and predicted probs |

## Point-forecast helper

```python
from bracketlearn.score import to_point

mu_mean   = to_point(dist, how="mean")     # E[Y | x]
mu_median = to_point(dist, how="median")   # F⁻¹(0.5)
mu_mode   = to_point(dist, how="mode")     # highest-density / highest-weight
```

Useful for benchmarking against classical sklearn regressors:

```python
from sklearn.metrics import mean_squared_error

y_oof = y[dist.ids.astype(int)]
mu_hat = to_point(dist, how="mean")
print("RMSE:", np.sqrt(mean_squared_error(y_oof, mu_hat)))
```

All three notebooks under `bracketlearn/notebooks/` end with a
point-forecast leaderboard comparing the probabilistic models' means
against `sklearn.linear_model.Ridge` and `LGBMRegressor`.
