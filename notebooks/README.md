# bracketlearn notebooks

Jupyter notebooks built on top of the public-dataset examples. Each
notebook produces around 6 plots that make the headline numbers
interpretable. These are PIT histograms, quantile fans, reliability diagrams,
bracket-price bars, and skill-score bars. Each notebook ends with a
**leaderboard** ranking multiple trainers against trivial baselines.

| Notebook | Dataset | Highlights |
|---|---|---|
| [`housing_brackets.ipynb`](housing_brackets.ipynb) | sklearn California housing | k-fold CV, bracket pricing $50k–$500k, 6-model leaderboard |
| [`bike_sharing_timeseries.ipynb`](bike_sharing_timeseries.ipynb) | OpenML Bike_Sharing_Demand (17k rows) | expanding-window CV, **two** baselines (marginal + lag-24 seasonal), 7-model leaderboard |
| [`grid_search_demo.ipynb`](grid_search_demo.ipynb) | sklearn California housing | 3×3 grid heatmap, competing-models leaderboard |
| [`leaderboard_zoo.ipynb`](leaderboard_zoo.ipynb) | both | **Exhaustive zoo**, 16+ models across baselines, single-stage dists, point+lifter combos, calibrated wrappers, and multi-stage DAGs (`StackedParametric`, `DistAsFeatures`, `LinearPoolDist`, `CDFBoostBracket`). Distributional-vs-point skill scatter. |

## Outputs are stripped

The committed `.ipynb` files carry code but no output. Executed notebooks embed
every figure as base64. These four totalled 2.5 MB, of which 2.4 MB was output,
which makes diffs unreadable and exceeds the size GitHub will render. A test
(`tests/test_notebooks_are_stripped.py`) keeps them stripped.

Reading them on GitHub therefore shows the code and not the results. Two
figures are committed separately under [`docs/_static/`](../docs/_static/),
namely the model-zoo CRPS leaderboard, used in the README, and the grid-search
CRPS surface, used in [search.md](../docs/guides/search.md). The remaining plots
are produced by running the notebooks locally.

## Running the notebooks

```bash
# From the bracketlearn root:
pip install -e ".[demo]" jupyter matplotlib
jupyter notebook notebooks/
```

## Editing the notebooks

The notebooks are paired with `.py` source files in
[`_src/`](_src/) via [jupytext](https://jupytext.readthedocs.io).
Edit either side and run `jupytext --sync notebooks/<name>.ipynb` to
keep them in sync. Diffs are much cleaner against the `.py` source.

All four notebooks share a single style module
[`_src/_style.py`](_src/_style.py), which holds rcParams, a `tab10`-based
model-family palette, and four plot helpers. The helpers are
`predicted_vs_realized_grid` (an sklearn-style scatter grid used as the headline
plot), `reliability_with_histogram`, `cdf_overlay_for_examples`, and
`leaderboard_bar`. The convention is to avoid bare bar charts unless they
encode at least a dozen models. For one or two numbers, annotate them
inside a scatter panel instead.

To rebuild the `.ipynb` from scratch after editing the `.py`:

```bash
jupytext --to ipynb _src/housing_brackets.py -o housing_brackets.ipynb
jupyter nbconvert --to notebook --execute housing_brackets.ipynb \
    --output housing_brackets.ipynb
```

## Baselines

Every notebook reports skill scores against trivial baselines from
[`bracketlearn.baselines`](../bracketlearn/baselines.py).

- `EmpiricalDistribution` predicts the marginal CDF of training y and
  ignores features. It is the floor any model must beat.
- `Persistence(lag=k)` predicts `y_{t-k}`. Setting lag=1 gives the naive
  baseline, and lag=24 captures daily seasonality on hourly data.

A model with CRPSS = +0.5 against `Empirical` cuts the baseline CRPS in
half. CRPSS = 0 ties the baseline, and a negative value is worse than the
trivial floor.
