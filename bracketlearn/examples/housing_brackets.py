"""California housing as a probabilistic-forecasting and pricing problem.

The dataset is sklearn's ``fetch_california_housing``, with 20 640 rows and 8
numeric features. The raw target is the median house value in units of $100k,
so 2.5 means $250k.

Turning a regression target into a market problem
-------------------------------------------------
Predicting a house value is a plain point-regression task, giving one number
per row. To price contracts on it, ranges are traded instead, and "will this
house sell between $300k and $400k?" pays $1 if it does. The script therefore
reframes the value in three ways.

1. The house value becomes the continuous underlying.
2. In place of a single predicted number, a full predictive distribution over
   the value is modelled. A lifted ridge model and QuantileReg each produce
   one, and an EmpiricalDistribution baseline gives a floor.
3. A bracket ladder is laid over the price axis, from $0 to $500k. Each bracket
   is one YES/NO contract, priced as the distribution's mass in that range.

The standard three steps follow. Forecast the distribution, price the brackets,
then score both the distribution, by CRPS and log-score, and the contracts, by
bracket log-loss and Brier.

Run::

    conda run -n weathermarkets python -m bracketlearn.examples.housing_brackets

This script demonstrates the following.

- ``Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()])``, a sklearn
  regressor lifted to a parametric-normal distribution.
- ``QuantileReg``, LightGBM per-τ heads giving a quantile-backed distribution
  that captures the heteroscedasticity ridge cannot.
- ``BracketLadder`` prices each distribution on a $0 to $500k ladder.
- ``PipelineResult.score`` reports distribution-level metrics, CRPS and
  log-score, alongside bracket-contract metrics, log-loss and Brier.
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
from bracketlearn.compose import WalkForward
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import Pipeline
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
    ts = ids.astype(float)                          # synthetic ordering, k-fold

    # Bracket ladder over the realistic price range. The outer edges are set
    # wide, at -100 and 100, so the ladder covers the full distribution support
    # for every row. That includes the high-end tail where qreg's stored
    # quantiles plateau at about 5.0, since the California housing target is
    # capped, and the occasional pathological RidgeCV prediction in the deep
    # negative range. Inner bins are 0.5-wide from $50k to $500k.
    edges = np.array(
        [-100.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 100.0]
    )
    print(f"ladder: {len(edges)-1} brackets covering "
          f"${edges[0]*100:.0f}k to ${edges[-1]*100:.0f}k "
          f"(outer bins absorb tail mass)")

    print("fitting (kfold, 5 folds) …")
    model = [
        # Baseline, the marginal-y distribution, which ignores X.
        Pipeline([EmpiricalDistribution()], name="emp"),
        Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()], name="ridge"),
        Pipeline(
            [QuantileReg(n_estimators=200, learning_rate=0.05, random_seed=0)],
            name="qreg",
        ),
    ]
    wf = WalkForward(
        cv="kfold", n_folds=5, shuffle=True, random_state=0, refit_on_full=True,
    )
    result = wf.fit_predict(model, X, y, ids=ids, timestamps=ts)

    print("\n=== distribution-level OOF metrics ===")
    print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

    print("\n=== bracket-contract OOF metrics ===")
    print(result.to_table(
        y, metrics=["log_loss_bracket", "brier_bracket"], edges=edges,
    ))

    # Skill score against the EmpiricalDistribution baseline, where
    # CRPSS = 1 - CRPS/CRPS_emp. Zero matches the baseline, positive beats it,
    # and negative is worse.
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
    header = "  ".join(f"{lbl:>11}" for lbl in bracket_labels)
    print(f"{'stage / row':<24}{header}")
    pred = wf.predict(X[:3], ids=np.arange(3),
                      timestamps=np.arange(3, dtype=float))
    B = edges.shape[0] - 1
    ladder = BracketLadder(edges_per_row=[edges] * 3)
    for stage_name, dist in pred.items():
        contracts = ladder.price(dist)
        prices = contracts.fair_price.reshape(-1, B)
        for row in range(3):
            row_str = f"{stage_name} row {row}  y={y[row]:.2f}"
            cells = "  ".join(f"{p:11.2f}" for p in prices[row])
            print(f"{row_str:<24}{cells}")

    print("\ndone.")


if __name__ == "__main__":
    main()
