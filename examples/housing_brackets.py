"""California housing → bracket-contract prices.

The pitch in one script: take a regression dataset, predict a *distribution*
over the target, then price a ladder of binary contracts ("will this house
sell above $250k?", "between $300k and $400k?").

Dataset: sklearn's ``fetch_california_housing`` — 20 640 rows, 8 numeric
features, target = median house value in units of $100k (so 2.5 = $250k).

Run::

    conda run -n weathermarkets python -m bracketlearn.examples.housing_brackets

What this script demonstrates:

- ``LiftedForecaster(SklearnPoint(RidgeCV()), GlobalResidual())`` —
  a sklearn regressor lifted to a parametric-normal distribution.
- ``QuantileReg`` — LightGBM per-τ heads; quantile-backed distribution
  that captures heteroscedasticity ridge cannot.
- ``BracketLadder`` prices each distribution on a $0–$500k ladder.
- ``PipelineResult.score`` reports distribution-level metrics (CRPS,
  log-score) and bracket-contract metrics (log loss, Brier) side by side.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import RidgeCV

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names.*",
    category=UserWarning,
)

from bracketlearn.adapters import BracketLadder
from bracketlearn.baselines import EmpiricalDistribution
from bracketlearn.composite import LiftedForecaster
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.trainers import QuantileReg, SklearnPoint


def main() -> None:
    print("loading California housing …")
    data = fetch_california_housing()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=float)        # units of $100k
    n = X.shape[0]
    # Subsample so the demo runs in ~30 s; the conclusions don't change.
    rng = np.random.default_rng(0)
    keep = rng.choice(n, size=4000, replace=False)
    X, y = X[keep], y[keep]
    ids = np.arange(X.shape[0])
    ts = ids.astype(float)                          # synthetic ordering — k-fold

    # Bracket ladder over the realistic price range: 8 buckets from $50k → $500k.
    # In $100k units that's [0.5, 5.0] → 0.5-wide bins.
    edges = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    ladder = BracketLadder(edges=edges)
    print(f"ladder: {len(edges)-1} brackets covering "
          f"${edges[0]*100:.0f}k to ${edges[-1]*100:.0f}k")

    print("fitting pipeline (kfold, 5 folds) …")
    pipeline = ForecastPipeline(
        steps=[
            # Baseline: marginal-y distribution, ignores X.
            ("emp", EmpiricalDistribution()),
            ("ridge", LiftedForecaster(
                SklearnPoint(RidgeCV()), GlobalResidual(), name="ridge",
            )),
            ("qreg", QuantileReg(n_estimators=200, learning_rate=0.05,
                                 random_seed=0)),
        ],
        cv="kfold", n_folds=5, shuffle=True, random_state=0,
        refit_on_full=True,
    )
    result = pipeline.fit_predict(X, y, ids=ids, timestamps=ts)

    print("\n=== distribution-level OOF metrics ===")
    print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

    print("\n=== bracket-contract OOF metrics ===")
    print(result.to_table(
        y, metrics=["log_loss_bracket", "brier_bracket"], ladder=ladder,
    ))

    # Skill score vs the EmpiricalDistribution baseline. CRPSS = 1 - CRPS/CRPS_emp;
    # 0 = matches baseline, positive = beats it, negative = worse.
    print("\n=== skill vs EmpiricalDistribution baseline ===")
    crps_scores = result.score(y, metrics=["crps"])
    base = crps_scores["emp"]["crps"]
    for stage, row in crps_scores.items():
        if stage == "emp":
            continue
        skill = 1.0 - row["crps"] / base
        print(f"  {stage:<8} CRPSS = {skill:+.3f}  "
              f"(CRPS {row['crps']:.4f} vs baseline {base:.4f})")

    # Predict bracket prices for the first 3 houses. ContractForecast stores
    # contracts flat with group_id linking the B rows from one entity, so we
    # reshape back to (N, B) for display.
    print("\n=== example bracket prices for 3 held-out rows ===")
    bracket_labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi
                      in zip(edges[:-1], edges[1:], strict=True)]
    header = "  ".join(f"{lbl:>9}" for lbl in bracket_labels)
    print(f"{'stage / row':<24}{header}")
    pred = pipeline.predict(X[:3], ids=np.arange(3),
                            timestamps=np.arange(3, dtype=float))
    B = edges.shape[0] - 1
    for stage_name, dist in pred.items():
        contracts = ladder.price(dist)
        prices = contracts.fair_price.reshape(-1, B)
        for row in range(3):
            row_str = f"{stage_name} row {row}  y={y[row]:.2f}"
            cells = "  ".join(f"{p:9.2f}" for p in prices[row])
            print(f"{row_str:<24}{cells}")

    print("\ndone.")


if __name__ == "__main__":
    main()
