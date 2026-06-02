"""GridSearch over a QuantileReg pipeline on California housing.

Searches a small 2-D grid of LightGBM hyperparameters using bracketlearn's
own time-aware CV inside each grid point (see ``bracketlearn.search`` —
sklearn's ``GridSearchCV`` would silently destroy time ordering).

Run::

    conda run -n weathermarkets python -m bracketlearn.examples.grid_search_demo

What this script demonstrates:

- ``GridSearch`` over a (model graph, WalkForward) pair with two node-level
  params (``qreg__n_estimators`` and ``qreg__learning_rate``) using sklearn's
  ``__``-nested syntax.
- The result table sorted by CRPS so the user can see the whole landscape,
  not just the winner.
- ``best_wf_`` is a fitted ``WalkForward`` (refit_on_full=True) ready for
  ``.predict()`` on new rows — no extra refit needed.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names.*",
    category=UserWarning,
)

from bracketlearn.baselines import EmpiricalDistribution
from bracketlearn.compose import WalkForward
from bracketlearn.pipeline import Pipeline
from bracketlearn.search import GridSearch
from bracketlearn.trainers import QuantileReg


def main() -> None:
    print("loading California housing …")
    data = fetch_california_housing()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=float)
    rng = np.random.default_rng(0)
    keep = rng.choice(X.shape[0], size=3000, replace=False)
    X, y = X[keep], y[keep]
    ids = np.arange(X.shape[0])
    ts = ids.astype(float)

    # Fit the marginal-y baseline once, outside the grid. Its CRPS is the
    # "you must beat this" anchor; we'll report the grid winner's skill
    # score against it. Doing this outside the grid avoids re-fitting
    # an identical baseline at every grid point.
    base_result = WalkForward(
        cv="kfold", n_folds=4, shuffle=True, random_state=0, refit_on_full=False,
    ).fit_predict(
        Pipeline([EmpiricalDistribution()], name="emp"),
        X, y, ids=ids, timestamps=ts,
    )
    baseline_crps = base_result.score(y, metrics=["crps"])["emp"]["crps"]
    print(f"\nbaseline EmpiricalDistribution CRPS = {baseline_crps:.4f}")

    model = Pipeline([QuantileReg(random_seed=0)], name="qreg")
    wf = WalkForward(
        cv="kfold", n_folds=4, shuffle=True, random_state=0, refit_on_full=True,
    )

    grid = {
        "qreg__n_estimators": [50, 150, 400],
        "qreg__learning_rate": [0.03, 0.1],
    }
    n_points = sum(1 for _ in range(len(grid["qreg__n_estimators"])
                                    * len(grid["qreg__learning_rate"])))
    print(f"\nrunning GridSearch over {n_points} grid points "
          f"({list(grid)}) …")
    search = GridSearch(
        model, wf, param_grid=grid,
        scoring="crps", refit_node="qreg",
    )
    search.fit(X, y, ids=ids, timestamps=ts)

    print("\n=== full grid (sorted by CRPS, lower is better) ===")
    print(f"{'n_estimators':>14}  {'learning_rate':>14}  {'crps':>10}")
    for row in sorted(search.results_, key=lambda r: r["crps"]):
        n_est = row["params"]["qreg__n_estimators"]
        lr = row["params"]["qreg__learning_rate"]
        print(f"{n_est:>14d}  {lr:>14.3f}  {row['crps']:>10.4f}")

    print(f"\nbest params : {search.best_params_}")
    print(f"best CRPS   : {search.best_score_:.4f}")
    print(f"baseline    : {baseline_crps:.4f}  (EmpiricalDistribution)")
    print(f"CRPSS       : {1.0 - search.best_score_ / baseline_crps:+.3f}")

    # best_wf_ is already fitted on full data (refit_on_full=True).
    pred = search.best_wf_.predict(
        X[:3], ids=np.arange(3), timestamps=np.arange(3, dtype=float),
    )
    print("\nthree predicted quantile vectors from best_wf_:")
    qvals = pred["qreg"].qvals       # (3, Q)
    taus = pred["qreg"].taus
    print("  τ:    " + "  ".join(f"{t:.2f}" for t in taus))
    for i in range(3):
        print(f"  row{i} y={y[i]:.2f}  " + "  ".join(f"{q:.2f}" for q in qvals[i]))

    print("\ndone.")


if __name__ == "__main__":
    main()
