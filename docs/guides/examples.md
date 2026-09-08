# Examples

bracketlearn ships seven runnable examples in [`bracketlearn/examples/`](https://github.com/FrederikBenirschke/bracket-learn/tree/main/bracketlearn/examples).
Three use **public sklearn / OpenML datasets** so they run anywhere with
no extra credentials. Two use the bundled weather sample.

## Public-dataset examples (recommended starting point)

### `housing_brackets.py`

California housing (sklearn-bundled, 20k rows). The whole idea in one script.
Take a regression dataset, predict a *distribution* over house prices,
then price a ladder of 8 binary contracts ($50k–$500k).

```bash
python -m bracketlearn.examples.housing_brackets
```

It shows `Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()])`,
`QuantileReg`, `BracketLadder`, k-fold CV, and side-by-side
distribution-level and contract-level metrics. Runs in ~30 s.

### `bike_sharing_timeseries.py`

Hourly bike-sharing demand from OpenML (17k rows, 2011–2012), a real time
series. The first run downloads the dataset, and sklearn caches it afterward.

```bash
python -m bracketlearn.examples.bike_sharing_timeseries
```

It shows `cv="expanding-window"` with `embargo`, `Pipeline([EMOS(),
Isotonic(pre_integrate_edges=edges)])` for per-fold tail calibration, and a
bracket ladder spanning 0–1000 bikes/hour. OOF alignment is handled
internally, so `result.score(y)` works without further setup.

### `grid_search_demo.py`

`GridSearch` over a 2-D LightGBM hyperparameter grid on California
housing, again with k-fold CV.

```bash
python -m bracketlearn.examples.grid_search_demo
```

It shows the nested `stage__field` param syntax, a full results table sorted
by CRPS, and a fitted `best_wf_` ready for `.predict()` on new data. Runs
in ~3 min.

## Notebooks (recommended)

Each of the three public-dataset examples also ships as a Jupyter
notebook with plots. These are PIT histograms, quantile fans, reliability
diagrams, bracket-price bars, skill-score bars, and a **leaderboard** ranking
multiple trainers against baselines.

See [`notebooks/`](https://github.com/FrederikBenirschke/bracket-learn/tree/main/notebooks).
Source for each notebook lives as a `.py` file under `notebooks/_src/`,
paired via [jupytext](https://jupytext.readthedocs.io), so diffs stay clean.

## Synthetic-data examples

These predate the public-dataset ports and exercise more trainers per
script. They are useful for seeing every backing in one place.

### `weather_e2e.py`

All 11 dist-producing trainers on synthetic weather-like data, namely
`ridge`, `lin_ols`, `emos`, `emos_calibrated`, `ngboost`, `mixture`,
`stack`, `qreg`, `qreg_conformal`, `qforest`, `cumbin`, `tail_specialist`,
and `online_agg`. The widest backing and family coverage of any example.

### `weather_rnn_e2e.py`

`RNNHourly`, a GRU over a `(N, 24, C)` hourly tensor with a station embedding,
lifted to a parametric normal via `GlobalResidual`.

## Real-data example: accuracy vs value

### `value_vs_accuracy_weather.py`

Fits EMOS on the bundled weather sample at
`bracketlearn/examples/data/weather_value_sample.parquet`, which holds 5,429
station-days of Kalshi contracts over 2026-03-17 to 2026-09-03 across 18
stations, carrying multi-model ensemble mean and spread, realized temperatures,
per-row bracket grids with open tails, and normalized reference prices. It
prices the fitted distribution onto each row's grid and scores it against the
reference price two ways, by Brier for accuracy and by `score.edge_alignment`
for value, with a station-day clustered bootstrap interval on the latter.

```bash
python -m bracketlearn.examples.value_vs_accuracy_weather
```

On this sample EMOS is less accurate than the market and its EA is negative,
with an interval containing zero. The synthetic case in §5 of the
[value guide](value_vs_accuracy.md), a forecast with worse Brier but positive
value, does not reproduce here. The two metrics do still order the calibration
adjustments differently, which is the property the guide is concerned with.
Section 5b gives the numbers and the decomposition of what changed from an
earlier result computed against a fixture whose bracket edges were wrong.

### `value_trainers_demo.py`

Trains *for* value with the `bracketlearn.value` trainers and scores the result.
On the same bundled sample it fits `BlendedBracketGBM` (LightGBM) and
`BlendedBracketNet` (torch) on `L = CE − λ·EA` across several tilts `λ`, then
scores each with `edge_alignment` for value and `edge_alignment_costed` for
value net of fee.

```bash
python -m bracketlearn.examples.value_trainers_demo
```

It shows the trainer `fit` / `predict_dist` contract, the construction of
`brackets_by_id` and `reference_by_id`, the scoring of the implied edge, and the
selection rule, which picks `λ` by *costed* value rather than EA. EA rising with
`λ` shows up cleanly here. The full "costed peaks interior" curve is in
`tests/test_value_trainers.py`.
