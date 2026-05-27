# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions
follow semver: MAJOR.MINOR.PATCH. Pre-1.0 the public API can break in any
minor release; patch releases are bug-fixes and additive tests.

## [0.3.0] — 2026-05-26

`DistributionForecast` is now an `abc.ABC` base with five concrete
subclasses; `BracketForecast` stores per-row edges natively;
bracket-aware trainers consume id-keyed dicts so each market/event
can carry its own bracket grid. Motivating use case: Kalshi
temperature contracts list a different bracket ladder every day — a
single forecast needs to price against the row's own grid, not a
shared global ladder.

### Added

- `bracketlearn.forecast.NormalForecast`, `StudentTForecast`,
  `MixtureNormalForecast`, `QuantileForecast`, `BracketForecast` —
  concrete subclasses of `DistributionForecast`. Each owns typed
  storage (no `params: dict[str, ndarray] | None`) and its own math
  (no `if/elif` on `(backing, family)` at every accessor). Both the
  classes and per-subclass `from_arrays` classmethods are re-exported
  at the top level.
- `DistributionForecast.integrate(edges_per_row) → BracketForecast`
  on the abstract base: projects any subclass onto a per-row bracket
  grid via `cdf_at_grid + np.diff`. Accepts 1-D shared edges, 2-D
  dense `(N, B+1)`, or a length-N ragged sequence (NaN-padded
  internally). Renormalises per row and raises if any row gets zero
  total mass.
- `BracketForecast.realized_bin(y) → (N,)` int array: per-row index
  of the bracket containing the realized value. Used by score
  functions and `Isotonic` to look up the realized bin under per-row
  edges.
- `BracketForecast.shared_edges() → (B+1,)`: returns the 1-D edge
  vector if every row's edges are identical and not NaN-padded;
  raises otherwise. Use from legacy callers still assuming a shared
  ladder.

### Changed

- **Breaking.** `DistributionForecast(backing=..., family=..., params=...)`
  union construction is gone. Use a concrete subclass directly
  (`NormalForecast(mu=, sigma=, ids=, timestamps=, provenance=)`) or
  the `DistributionForecast.from_*` classmethods, which now route to
  the matching subclass. The two-level `(backing, family)`
  discriminator collapses to one level — the class itself is the
  backing.
- **Breaking.** `BracketForecast.edges` is now `(N, B+1)` per-row,
  NaN-padded for ragged-length rows. `from_brackets` still accepts
  1-D shared edges and broadcasts them internally, so most existing
  callers keep working; callers that read `dist.edges` directly and
  assumed 1-D need to either consume the 2-D array or call
  `dist.shared_edges()`.
- **Breaking.** `CumulativeBinary` drops `cutpoints` + `outer_edges`
  for `cutpoints_by_id: dict[id → 1-D cutpoint array]` and
  `outer_edges_by_id: dict[id → (lo, hi)]`. `fit()` now requires the
  `ids=` kwarg. Each row contributes its own K_i augmented training
  examples to a single global LGBM classifier; the cutpoint flows in
  as a feature so the model generalises across grids. Predict emits
  a per-row `BracketForecast` (NaN-padded ragged columns when K_i
  varies).
- **Breaking.** `TailSpecialist` drops `edges` for
  `brackets_by_id: dict[id → 1-D edge array]`. Algorithmic change:
  training-time tail indicators become "y in row's first/last
  bracket" (per-row, computed from the row's own edges) instead of
  "y < shared_edges[1]" / "y >= shared_edges[-2]". Predict calls
  `upstream.integrate(per_row_edges)` instead of
  `upstream.cdf(shared_edges)`.
- **Breaking.** `CDFBoostBracket` drops `edges` for `brackets_by_id`
  with a strict uniform-B requirement (edge *values* may vary per
  row, but the bin count must be identical across rows because the
  trainer fits one head per bin). `fit()` requires `ids=` kwarg.
  Featurisation now uses `cdf_at_grid` on the per-row grid.
- **Breaking.** `Isotonic` calibrator drops the `edges` constructor
  arg. Inputs and outputs are `BracketForecast` (any subclass that
  isn't a `BracketForecast` must be `.integrate()`d first). Pass
  `pre_integrate_edges=...` to have `Isotonic` auto-integrate
  non-bracket inputs internally — used by the `emos_calibrated()`
  factory to wrap a parametric forecaster with bracket calibration.
- `score.log_score_bracket`, `crps_bracket`, `to_point`, and
  `_quantile_at` now consume per-row edges (NaN-padded tail aware).
- `pipeline._stitch_folds` for BRACKET folds concatenates per-row
  edges + probs along axis 0; the previous shared-edges sanity check
  is gone (edges are now per-row by construction).
- `Backing` and `ParametricFamily` enums survive as compat
  `@property` shims on each subclass so existing consumers (score,
  pipeline, lift, restrict, downstream tests) keep working
  untouched. Slated for removal once consumers migrate to
  `isinstance` dispatch.
- `DistributionForecast.from_*` classmethods kept as construction
  shims that route to the correct subclass.

### Removed

- `bracketlearn.lift._bracket_probs_from_dist`: redundant with
  `dist.integrate(edges)`, which works on any subclass and returns
  a typed `BracketForecast` instead of raw probs.

## [0.2.0] — 2026-05-26

Initial public release. Sklearn-style API; four backings (parametric
normal / mixture-normal / quantile / bracket); `ForecastPipeline` with
time-aware CV; bracket/binary/twin/threshold adapters; CRPS, log-score,
PIT, Brier metrics; three public-dataset examples; Sphinx docs;
GitHub Actions CI.

### Added

- `bracketlearn.__version__` (top-level + `__all__`) so callers can
  introspect the installed version without parsing package metadata.
- `adapters.BinaryAbove` — `P(X > k)` priced as `1 - dist.cdf(k)`. Maps
  to single-threshold Kalshi / Polymarket contracts.
- `adapters.BinaryBelow` — `P(X ≤ k)` priced as `dist.cdf(k)`.
- `adapters.Twin` — paired YES/NO at one strike. Two rows per entity
  sharing `group_id`, `fair_price` sums to 1.0 by construction. Maps to
  prediction-market spread / total contracts (`Eagles -3.5`,
  `Over 47.5 total points`).
- `adapters.ThresholdLadder` — survival function evaluated at S strikes
  (`[P(X > k_i)]_i`). Maps to single-side Kalshi multi-threshold ladders.

### Changed

- Packaging: `pyproject.toml` now uses
  `[tool.setuptools.package-dir]` to map `bracketlearn = "."`, so wheels
  built from the flat layout actually ship the 15 source modules. The
  previous `packages.find` config produced a metadata-only wheel that
  installed no Python code (release blocker).
- `README.md` and `docs/index.md` install instructions now reflect
  pre-PyPI status (`pip install -e` from a git clone). Will switch back
  to `pip install bracketlearn` once published.
- GitHub URLs in `pyproject.toml` and `docs/conf.py` unified to
  `FrederikBenirschke/bracketlearn` (previous mismatch: docs used
  `frederikbenirschke/...`, pyproject used `fbenirschke/...`).
- `bracketlearn.__all__` extended with `PerRowBracketLadder`, `Twin`,
  `ThresholdLadder` (the per-row ladder was missing from the previous
  release; the new binary/threshold/twin adapters are exported alongside).
- README rewritten with a prediction-market-first pitch, an adapter
  catalogue mapping each adapter to real venue contracts, and a
  synthetic NYC-max-temperature worked example showing all four
  contract shapes priced from one EMOS forecast.

### Removed

- Six unimplemented methods on `DistributionForecast` / `ContractForecast`:
  `from_empirical`, `to_quantiles`, `to_brackets`, `to_normal`,
  `is_lossless_to`, and `ContractForecast.calibrate`. All six raised
  `NotImplementedError` and had no callers outside the no-silent-fallbacks
  tests. The two corresponding stub-tests were dropped. Public class
  surface now matches what actually works.
- Unimplemented adapter stubs that raised `NotImplementedError`:
  `Bracket` (single-bin), `VanillaCall`, `VanillaPut`, `LinearCombo`,
  `CallSpread`, `Butterfly`, `Condor`, `PerRow`, `Custom`, `VenueSpec`,
  `to_quote`. These covered options-style payoffs that don't exist on
  the prediction-market venues this library is built for. Drop net:
  ~250 lines of stubs plus their "raises NotImplementedError" tests.
- `test_no_silent_fallbacks.test_adapter_stubs_raise_not_implemented`
  and `test_to_quote_raises_not_implemented` removed alongside the
  stubs they covered.

### Fixed

- Sphinx `-W` build (the CI gate) was failing on three docstring issues
  in `trainers.py` — `CDFBoostBracket` (definition-list unindent) and
  `DistAsFeatures` (undefined `|taus|` / `|cuts|` substitutions). Both
  rewritten with code-literal formulas.
- `docs/guides/adapters.md` was documenting `BinaryAbove`, `BinaryBelow`,
  `Twin`, `ThresholdLadder`, `LinearCombo`, `PerRow`, `Custom`,
  `VanillaCall`, and `VanillaPut` as stubs. The first four are fully
  implemented and tested; the latter five were deleted earlier this
  cycle. Rewrote the guide to document each shipping adapter with a
  working example.
- `ruff check .` now passes with zero errors (CI lint was red on 31
  warnings). Manual fixes covered nested-`if` collapses in
  `base.py:122-128` and `trainers.py:326-334`, a `contextlib.suppress`
  rewrite in `base.py:141-144`, and a semicolon split in
  `tests/test_trainers.py:231`. `__init__.py` restored to a single
  alphabetical import block.

### Added (continued — earlier in this release cycle)

- `DistributionForecast.cdf_at_grid(y)` — per-row CDF on a *per-row*
  evaluation grid. Input `y` shape `(N, M)` → output `(N, M)`, where row
  `i` uses its own grid `y[i, :]`. NaN entries round-trip as NaN so
  callers can pad ragged grids. Generalises `cdf_at` (which is the M=1
  case in spirit) and avoids the `(N, M_global)` cross-product of `cdf`
  when each row needs different query points.
- `adapters.PerRowBracketLadder` — bracket ladder with a *per-row* edge
  vector. Motivated by Kalshi-style daily-rotating temperature brackets
  (NYC max-temp etc.). Storage is ragged (`edges_per_row: list[ndarray]`,
  per-row `B_i` allowed to vary). `include_tail_buckets=True` emits
  explicit "below edges[0]" and "above edges[-1]" rows so per-entity
  prices sum to exactly 1.0; otherwise the existing coverage check
  (warn / strict-raise) gates against silent tail leakage. Built on
  `cdf_at_grid` so parametric backings stay fully vectorised.
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
