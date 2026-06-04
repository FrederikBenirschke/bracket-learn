# Examples

bracketlearn ships five runnable examples in [`bracketlearn/examples/`](https://github.com/FrederikBenirschke/bracketlearn/tree/main/bracketlearn/examples).
Three use **public sklearn / OpenML datasets** so they run anywhere with
no extra credentials.

## Public-dataset examples (recommended starting point)

### `housing_brackets.py`

California housing (sklearn-bundled, 20k rows). The pitch in one script:
take a regression dataset, predict a *distribution* over house prices,
then price a ladder of 8 binary contracts ($50k–$500k).

```bash
python -m bracketlearn.examples.housing_brackets
```

Shows: `Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()])`,
`QuantileReg`, `BracketLadder`, k-fold CV, side-by-side
distribution-level and contract-level metrics. Runs in ~30 s.

### `bike_sharing_timeseries.py`

Hourly bike-sharing demand from OpenML (17k rows, 2011–2012). A real time
series; the first run downloads the dataset, and sklearn caches it afterward.

```bash
python -m bracketlearn.examples.bike_sharing_timeseries
```

Shows: `cv="expanding-window"` with `embargo`, `Pipeline([EMOS(),
Isotonic(pre_integrate_edges=edges)])` (per-fold tail calibration), a
bracket ladder spanning 0–1000 bikes/hour. OOF alignment stays invisible to
you: `result.score(y)` just works.

### `grid_search_demo.py`

`GridSearch` over a 2-D LightGBM hyperparameter grid on California
housing, again with k-fold CV.

```bash
python -m bracketlearn.examples.grid_search_demo
```

Shows: nested `stage__field` param syntax, full results table sorted by
CRPS, fitted `best_wf_` ready for `.predict()` on new data. Runs
in ~3 min.

## Notebooks (recommended)

Each of the three public-dataset examples also ships as a Jupyter
notebook with plots: PIT histograms, quantile fans, reliability diagrams,
bracket-price bars, skill-score bars, and a **leaderboard** ranking multiple
trainers against baselines.

See [`bracketlearn/notebooks/`](https://github.com/FrederikBenirschke/bracketlearn/tree/main/bracketlearn/notebooks).
Source for each notebook lives as a `.py` file under `notebooks/_src/`
(via [jupytext](https://jupytext.readthedocs.io)) so diffs stay clean.

## Synthetic-data examples

These predate the public-dataset ports and stress more trainers per
script. Useful if you want to see every backing in one place.

### `weather_e2e.py`

All 11 dist-producing trainers on synthetic weather-like data:
`ridge`, `lin_ols`, `emos`, `emos_calibrated`, `ngboost`, `mixture`,
`stack`, `qreg`, `qreg_conformal`, `qforest`, `cumbin`, `tail_specialist`,
`online_agg`. The widest backing/family coverage of any example.

### `weather_rnn_e2e.py`

`RNNHourly` (GRU over a `(N, 24, C)` hourly tensor + station embedding)
lifted to a parametric normal via `GlobalResidual`.
