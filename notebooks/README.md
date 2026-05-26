# bracketlearn notebooks

Jupyter notebooks built on top of the public-dataset examples. Each
notebook produces ~6 plots that make the headline numbers
interpretable — PIT histograms, quantile fans, reliability diagrams,
bracket-price bars, skill-score bars — and ends with a **leaderboard**
ranking multiple trainers against trivial baselines.

| Notebook | Dataset | Highlights |
|---|---|---|
| [`housing_brackets.ipynb`](housing_brackets.ipynb) | sklearn California housing | k-fold CV; bracket pricing $50k–$500k; 6-model leaderboard |
| [`bike_sharing_timeseries.ipynb`](bike_sharing_timeseries.ipynb) | OpenML Bike_Sharing_Demand (17k rows) | expanding-window CV; **two** baselines (marginal + lag-24 seasonal); 7-model leaderboard |
| [`grid_search_demo.ipynb`](grid_search_demo.ipynb) | sklearn California housing | 3×3 grid heatmap; competing-models leaderboard |
| [`leaderboard_zoo.ipynb`](leaderboard_zoo.ipynb) | both | **Exhaustive zoo:** 16+ models across baselines, single-stage dists, point+lifter combos, calibrated wrappers, multi-stage DAGs (`Stacking`, `DistAsFeatures`, `LinearPoolDist`, `CDFBoostBracket`). Distributional-vs-point skill scatter. |

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

To rebuild the `.ipynb` from scratch after editing the `.py`:

```bash
jupytext --to ipynb _src/housing_brackets.py -o housing_brackets.ipynb
jupyter nbconvert --to notebook --execute housing_brackets.ipynb \
    --output housing_brackets.ipynb
```

## Baselines

Every notebook reports skill scores against trivial baselines from
[`bracketlearn.baselines`](../baselines.py):

- `EmpiricalDistribution` — predicts the marginal CDF of training y;
  ignores features. The "you must beat this" floor.
- `Persistence(lag=k)` — predicts `y_{t-k}`; lag=1 is the naive
  baseline, lag=24 captures daily seasonality on hourly data.

A model with CRPSS = +0.5 against `Empirical` cuts the baseline CRPS in
half; CRPSS = 0 ties the baseline; negative means worse than the
trivial floor.
