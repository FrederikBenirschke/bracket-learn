# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions
follow semver: MAJOR.MINOR.PATCH. Pre-1.0 the public API can break in any
minor release; patch releases are bug-fixes and additive tests.

## [Unreleased]

### Added

- `DistributionForecast.cdf_at(y)` — per-row CDF for any backing.
  Replaces the O(N²) `np.diag(dist.cdf(y))` pattern; drops `score.pit`
  memory from ~800 MB to ~80 KB at N=10k.
- `forecast.bracket_probs_from_cdf_at_edges(cdf_at_edges, source)` —
  shared diff / clip / row-sum-check / normalise helper. Used by
  `CumulativeBinary.predict_dist` and `lift._bracket_probs_from_dist`.
- Top-level re-exports: every estimator, adapter, lifter, calibrator,
  pipeline, search, baselines, base + `clone` (67 names in
  `bracketlearn.__all__`). `from bracketlearn import EMOS, BracketLadder,
  ForecastPipeline, ...` works without digging into submodules.
- `BaseEstimator` now subclasses `sklearn.base.BaseEstimator`. Adds
  `__sklearn_is_fitted__`, `n_features_in_`, `feature_names_in_`,
  auto-fill of `ids=` / `timestamps=` kwargs, sklearn-compatible
  `get_params` / `set_params` / `clone`. `sklearn.base.clone(est)` and
  `sklearn.utils.validation.check_is_fitted(est)` work on bracketlearn
  estimators.
- `BracketLadder` now accepts `strict: bool = False` and
  `coverage_tol: float = 1e-4`. Coverage shortfalls warn (or raise
  under `strict=True`) instead of silently dropping mass.
- `EMOS.sigma_fit_was_constant_` flag — exposed when the fit fell back
  to constant σ because the linear-in-variance MoM regression returned
  a negative coefficient.

### Changed

- `Persistence.predict` tiles `tail_y_` cyclically across the inference
  horizon. `lag=1` unchanged ("predict the last training y everywhere");
  `lag=24` now actually replays yesterday's diurnal cycle instead of
  holding at the last value after row 24.
- `BaseEstimator.__init_subclass__` wraps subclass `fit` / `predict` /
  `predict_dist` so callers may omit `ids=` and `timestamps=`. Explicit
  kwargs still win.
- `PipelineResult.score` now dispatches via a module-level metric
  registry (`_metric_crps`, `_metric_log_score`, `_compute_metric`).
  Adding a new backing means updating one helper, not four if/elif
  blocks.
- Bracket `cdf` and `pdf` now use a single `np.searchsorted` call
  instead of a per-query Python loop. Same for `cdf_at` on the bracket
  branch. Three isotonic-repair loops in `QuantileReg`, `QuantileForest`,
  `CumulativeBinary` collapsed to one `np.maximum.accumulate(..., axis=1)`
  each. `OnlineAggregator.predict` per-row Python loop replaced with
  vectorised mask + sum.

### Fixed

- **B1** `BracketLadder` row sums dropping below 1 for quantile-backed
  dists when the ladder didn't span the distribution's effective
  support. Now surfaced via a `UserWarning` (or raise under
  `strict=True`) instead of silent mass loss.
- **B2** `Stacking.fit` / `Stacking.predict_dist` now require upstream
  `.ids` to align across all deps; mismatch raises. Tolerance-based
  `sigma_` degenerate check catches float-noise-positive cases
  (previously only exact-zero was caught).
- **B3** `CumulativeBinary` now requires `outer_edges=(lo, hi)` as a
  constructor argument with `__post_init__` validation; previously the
  outer edges were invented from a silent pad.
- **B4** `TailSpecialist` `class_weight="balanced"` is applied only
  when caller passes no `sample_weight`. Inner-sum=0 silent uniform
  fallback became a loud `ValueError`. New warning when classifier tail
  probabilities disagree with upstream EMOS by > 0.5 on the edge bins
  (surfaces a "ladder too narrow" misconfiguration).
- **B5** `RNNHourly.predict` raises on unknown station IDs (previously
  silently clipped to 0).
- **B6** `SklearnPoint.fit` and `pipeline._predict_with_deps` now use
  `inspect.signature` introspection to decide whether to pass
  `sample_weight` / `deps_oof`. Genuine `TypeError` from inside `.fit()`
  now propagates instead of being swallowed.
- **B7** `Persistence` baseline now produces a diurnal cycle under
  `lag=24` instead of holding at the last value (see Changed).
- **B8** Stub methods in `forecast.py` (`from_empirical`, `to_quantiles`,
  `to_brackets`, `to_normal`, `is_lossless_to`, `ContractForecast.calibrate`)
  and `adapters.py` (`BinaryAbove`, `BinaryBelow`, `Bracket`,
  `ThresholdLadder`, `Twin`, `VanillaCall`, `VanillaPut`, `LinearCombo`,
  `PerRow`, `Custom`, `to_quote`) now raise `NotImplementedError`
  instead of returning `None`.
- **B9** `score.pit` per-row CDF (see Added).
- **B10** `lift.Isotonic` and `_bracket_probs_from_dist` row-sum guards
  raise `ValueError` instead of silently substituting a uniform
  distribution. `EMOS.fit` MoM negative-variance falls back to constant
  σ at fit time; `EMOS.predict_dist` raises on non-positive variance
  (extrapolation outside training spread range) instead of clipping.

### Tests

- 235 tests pass (was 195 at start of audit).
- New test files: `test_ladder_sum.py` (17 tests), `test_no_silent_fallbacks.py`
  (14 tests), `test_sklearn_compat_v2.py` (11 tests), `test_invariants.py`
  (12 tests).
- `test_trainers.py` extended with positive-path Stacking + TailSpecialist
  integration tests, factory tests for `ridge` / `market_ols` /
  `emos_calibrated`, and `sample_weight` respect tests for
  `SklearnPoint` / `EMOS` / `EmpiricalDistribution`.

### Docs

- New guides: `adapters.md`, `baselines.md`, `tail_policies.md`.
- `AUDIT.md` (audit findings + remediation plan) and `DECISION.md`
  (standalone-PyPI-library decision) added under `bracketlearn/`.

## [0.2.0] — 2026-05

Initial public release; sklearn-style API, four backings (parametric
normal / mixture-normal / quantile / bracket), `ForecastPipeline` with
time-aware CV, `BracketLadder` adapter, CRPS / log-score / PIT / Brier
metrics, three public-dataset examples, Sphinx docs, GitHub Actions CI.
