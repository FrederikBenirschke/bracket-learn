# BracketExpander — any sklearn estimator on a bracket ladder

`BracketExpander` is the "use any sklearn classifier or regressor"
entry point for bracket-aware learning. It owns the per-row →
per-(row, bracket) reshape and nothing else — model choice, loss
function, and the per-(row, bracket) target stay in caller code.

## Why this exists

The pre-v0.5.0 classes `BracketClassifier` / `BracketRegressor`
conflated two concerns inside `fit`:

1. **Reshape** a per-row design `(N, F)` into a per-(row, bracket)
   design `(M, F+2)` with `[..., lo, hi]` appended.
2. **Fit** an sklearn estimator on that design with a hardcoded
   bracket-hit target `1[y ∈ [lo, hi))`.

Any caller who wanted a different per-(row, bracket) target — a
mispricing residual `hit − market_p`, an importance-weighted hit, a
quantile loss — had to fork the class. v0.5.0 separates the two: the
reshape lives in `BracketExpander`, the fit is plain sklearn.

## Default flow — bracket-hit target

```python
from bracketlearn import BracketExpander
from lightgbm import LGBMClassifier

# brackets_by_id maps each entity id to its 1-D edge array.
# Ragged edges across ids are fine — the expander handles them.
brackets_by_id = {
    "row_0": np.array([0.0, 60.0, 70.0, 80.0, 100.0]),
    "row_1": np.array([0.0, 55.0, 65.0, 75.0, 85.0, 100.0]),
    ...
}

exp = BracketExpander(brackets_by_id=brackets_by_id)

# Train: expand (X, y) and get the default bracket-hit target.
X_exp, y_exp = exp.fit_transform(X_train, y_train, ids=train_ids)
# X_exp is (M, F+2). y_exp is (M,) with 0/1 bracket-hit labels.

clf = LGBMClassifier(n_estimators=400).fit(X_exp, y_exp)

# Predict: expand X only, score per-(row, bracket), assemble back.
X_pred_exp, _ = exp.transform(X_pred, ids=pred_ids)
scores = clf.predict_proba(X_pred_exp)[:, 1]
dist = exp.assemble_dist(scores, ids=pred_ids, timestamps=pred_ts)
# dist is a row-renormalised BracketForecast.
```

`assemble_dist` row-renormalises so each predicted row sums to 1 — the
raw per-bracket scores from `clf.predict_proba` won't, since each
augmented row is scored independently.

## Custom per-(row, bracket) target

Skip the default `y_exp`; build the target on top of `X_exp`:

```python
X_exp, y_hit = exp.fit_transform(X_train, y_train, ids=train_ids)

market_p_exp = build_market_p_per_bracket(...)  # caller-side, (M,)

# Mispricing residual as target; market price as feature.
X_exp = np.column_stack([X_exp, market_p_exp])
y_target = y_hit - market_p_exp

from lightgbm import LGBMRegressor
reg = LGBMRegressor().fit(X_exp, y_target)
```

The expander has no opinion about which estimator class fits the
augmented design or what the target should be. That's the point.

## Comparison with the distribution-first trainers

| Trainer family            | Trains on            | When to reach for it                                                           |
|---------------------------|----------------------|--------------------------------------------------------------------------------|
| `EMOS`, `NGBoostNormal`,  | `(X, y)` directly    | Continuous quantity with reasonable parametric shape; brackets via `.integrate()`. |
| `MixtureNormals`, ...     |                      |                                                                                |
| `CumulativeBinary`,       | bracket-derived      | Brackets baked in; LGBM-monotone cumulative CDF for smooth problems.           |
| `TailSpecialist`,         | indicators           |                                                                                |
| `CDFBoostBracket`         |                      |                                                                                |
| `BracketExpander` + any   | bracket-derived,     | Need a non-default per-(row, bracket) target, or want sklearn estimator        |
| sklearn estimator         | caller-built target  | flexibility (MLP, ElasticNet, custom GAM) outside the built-in families.       |

For straightforward bracket-hit problems on smooth data, prefer
`CumulativeBinary` — its monotone-LGBM cumulative head gives
calibration the unconstrained expander can't. Reach for the expander
when (a) the target isn't a plain hit, or (b) the estimator you want
isn't already wrapped.
