# bracketlearn — audit & remediation plan

Date: 2026-05-25
Scope: full package audit (API, estimators, tests, examples, docs, integration).

---

## 0. Verdict

Framework ships A→D (packaging, sklearn-contract attempt, CV/weights/multi-target,
docs+CI). Two structural problems:

1. **API is not sklearn-shaped.** Custom `BaseEstimator`, `predict_dist` not
   `predict_proba`, mandatory `ids=`/`timestamps=` kwargs, estimators not
   exported. Cannot drop into `sklearn.Pipeline` / `GridSearchCV` /
   `cross_val_score`.
2. **Disconnected from the repo.** No imports outside `bracketlearn/` except
   its own CI. `prediction_market_weather/ml/trainers/*` are shadow
   re-implementations (same names: EMOS, QuantileReg, Stacking, etc.).
   `docs/weather/ml_inference_and_unification.md` explicitly rejects
   unification.

Decision point before any fix: **standalone PyPI lib** OR **repo-internal
framework**. The current "both" is the root of most issues.

---

## 1. Severe bugs

### B1. Quantile-backed bracket prices don't sum to 1 — FIXED 2026-05-25
- **Root cause** (revised after investigation): not a clip-semantics bug
  inside `cdf`. Clip is self-consistent: under clip, the distribution
  lives entirely in `[qvals[0], qvals[-1]]`, and `cdf(x<qvals[0]) = 0`,
  `cdf(x>qvals[-1]) = 1`. The real failure was **ladder coverage**:
  `BracketLadder.price` silently dropped mass when the ladder didn't
  span the distribution's effective support — e.g. `edges[-1]=5.0`
  while `qvals[-1]=5.04` (LightGBM plateau), or ridge predicting
  `mu=-76` with the ladder starting at 0.
- **Fix**: `BracketLadder` now checks row-sum coverage after pricing.
  `strict=False` (default) warns with worst-row missed mass + bad-row
  count; `strict=True` raises `ValueError`. Tolerance via
  `coverage_tol` (default 1e-4). Rule #0.5 compliant.
- **Tests**: `tests/test_ladder_sum.py` covers normal / bracket /
  quantile-clip / mixture-normal with B∈{1,2,5,20}, the
  inside-quantile-range "documented mass loss" case, the strict=True
  raise path, and the regression scenario (qvals plateau).
- **Example update**: `housing_brackets.py` now uses outer edges
  `[-100, ..., 100]` to absorb tail mass on all 3 stages cleanly.

### B2. `Stacking` row-alignment by trust — FIXED 2026-05-25
- `Stacking.fit` now requires every upstream's `.ids` vector to match
  the others; mismatch raises. Optional `ids=` kwarg also checks
  caller alignment.
- `Stacking.predict_dist` requires each upstream's `.ids` to match
  caller `ids` exactly.
- `sigma_` degenerate check upgraded from `<= 0` (only catches exact
  zero) to a tolerance-based check against `np.std(y)` — catches
  float-noise-positive cases that look like upstream-μ-vs-y
  collinearity (data leak).
- Tests: `test_no_silent_fallbacks.py::test_stacking_raises_on_*`.

### B3. `CumulativeBinary` invents outer edges — FIXED 2026-05-25
- `outer_edges: tuple[float, float]` is now a required constructor arg
  (no default; `__post_init__` validates `lo < cuts[0]` and
  `cuts[-1] < hi`).
- Row-sum guard upgraded from `np.where(row_sum > 0, row_sum, 1.0)`
  silent uniform to a loud `ValueError`.
- Callers updated: `tests/test_trainers.py`,
  `examples/weather_e2e.py`, `notebooks/_src/bike_sharing_timeseries.py`.

### B4. `TailSpecialist` hard-codes `class_weight="balanced"` — FIXED 2026-05-25
- `class_weight="balanced"` now applies *only* when caller passes no
  `sample_weight`. With weights, balanced is dropped — user's weights
  rule.
- Inner-sum=0 silent uniform fallback → loud `ValueError`.
- Final row-sum guard upgraded from `np.where(..., 1.0)` to a raise
  (logic error if ever triggered).
- Body rescaling shape (`body_probs[:, 1:-1]` discarding edge bin mass)
  — addressed 2026-05-25 with a consistency warning. The override of
  upstream edge-bin mass by the classifier output is *intentional*
  (the whole point of a tail specialist), but on narrow ladders the
  discarded body mass can be substantial. `predict_dist` now compares
  classifier p_lo/p_hi against upstream `body_probs[:, 0]` /
  `body_probs[:, -1]` and emits a `UserWarning` if they disagree by
  more than 0.5 — surfaces a "ladder too narrow" misconfiguration.

### B5. `RNNHourly` clips unseen station IDs to 0 — FIXED 2026-05-25
- `RNNHourly.predict` now raises `ValueError` on any `station_id`
  outside `[0, n_stations_-1]` — reports the unknown IDs (first 10).
- Test: `test_no_silent_fallbacks.py::test_rnn_hourly_raises_on_unknown_station_ids`.

### B6. `SklearnPoint` bare `except TypeError` — FIXED 2026-05-25
- New `_estimator_accepts_sample_weight()` helper introspects the
  estimator's `fit` signature. If `sample_weight` isn't in the
  parameters, we don't pass it. Genuine `TypeError`s raised inside
  fit now propagate.
- Same fix applied to `pipeline._predict_with_deps` (was also a bare
  `except TypeError` swallow for `deps_oof`).
- Tests: `test_sklearn_point_introspects_sample_weight_signature`,
  `test_sklearn_point_raises_genuine_typeerror_inside_fit`.

### B7. Persistence baseline holds at last value after `lag` rows — FIXED 2026-05-25
- `Persistence.predict` now tiles `tail_y_` across the inference
  horizon via `tail_y_[np.arange(N) % self.lag]`. lag=1 still collapses
  to "predict the last training y everywhere"; lag=24 on hourly data
  replays the last full day repeatedly — actually matches the
  `examples/bike_sharing_timeseries.py` "diurnal cycle" framing.
- Docstring rewritten to spell out the cyclic semantics.
- Tests: `test_baselines.py::test_lag_k_cycles` (replaces the old
  `test_lag_k_peels_then_holds`) and `::test_lag24_diurnal_cycle` pin
  the new behaviour.

### B8. Stubs return `None` silently — FIXED 2026-05-25
- All concrete-class stubs in `forecast.py` (`from_empirical`,
  `to_quantiles`, `to_brackets`, `to_normal`, `is_lossless_to`,
  `ContractForecast.calibrate`) now raise `NotImplementedError`.
- All adapter stubs in `adapters.py` (`BinaryAbove`, `BinaryBelow`,
  `Bracket`, `ThresholdLadder`, `Twin`, `VanillaCall`, `VanillaPut`,
  `LinearCombo`, `PerRow`, `Custom`, `to_quote`) raise
  `NotImplementedError`.
- `from_student_t` already implemented — no change needed.
- Protocol bodies (`ContractAdapter.price` in the Protocol class) keep
  `...` — idiomatic Protocol stub, not a real method.
- Tests: `test_no_silent_fallbacks.py::test_*_stubs_raise*`.

### B9. `pit` builds full (N,N) matrix then `np.diag` — FIXED 2026-05-25
- New `DistributionForecast.cdf_at(y)` returns the per-row CDF in
  O(N) time and O(N) memory. Implemented for all four backings
  (normal, student-t, mixture-normal, bracket, quantile) — vectorised
  except for the quantile branch (per-row `np.interp` is unavoidable
  with non-shared knots).
- `score.pit` rewired to call `cdf_at` instead of `np.diag(dist.cdf(y))`.
  At N=10k this drops from ~800 MB peak to ~80 KB.
- Tests in `test_scores.py`: `cdf_at` matches `np.diag(cdf(y))` for
  normal / bracket / quantile / mixture-normal; raises on a length
  mismatch.

### B10. Other silent fallbacks — FIXED 2026-05-25
- `lift.py` `Isotonic.transform` and `_bracket_probs_from_dist` row-sum
  guards now raise `ValueError` with a row-count diagnostic. No more
  silent uniform substitution.
- `EMOS.fit` MoM negative-variance — fixed 2026-05-25.
  Method-of-moments OLS for variance (`r² ≈ c + d·ens_var`) is
  unconstrained, so it can return `c_<0` or `d_<0` and yield negative
  variance somewhere in the training range. Pre-fix code silently
  clipped at predict time via `np.clip(..., 1e-6, None)` — Rule #0.5
  violation. Fix: at fit, detect `c_<0 ∨ d_<0 ∨ any var_train ≤ 0`
  and fall back to a constant variance (mean of squared residuals),
  recording the choice on `sigma_fit_was_constant_`. At predict, if
  variance still goes non-positive (means inference X is outside the
  training spread range), raise a `ValueError` naming the bad row
  count instead of clipping.
- Tests: `test_bracket_probs_from_dist_raises_on_zero_row_sum`,
  `test_emos_falls_back_to_constant_sigma_when_mom_gives_negative_coef`,
  `test_emos_predict_raises_on_negative_variance_extrapolation`.

---

## 2. API / sklearn-contract gaps

### A1. Not exported from top level
- `bracketlearn/__init__.py:9-25` only re-exports protocols + dataclasses.
- Every example imports estimators from submodules. `from bracketlearn import
  EMOS` → `ImportError`.

### A2. Wrong verbs
- `predict_dist` instead of `predict_proba`. Library *about distributions* —
  this is the central discoverability dead end.
- No `.score()` on estimators; only on `PipelineResult`.

### A3. Mandatory `ids=` / `timestamps=` kwargs
- `protocols.py:49-67`: every `fit`/`predict_dist` requires them as
  keyword-only. Plain `(X, y)` call → `TypeError`.
- Blocks `sklearn.Pipeline`, `cross_val_score`, `check_estimator`.

### A4. Missing sklearn introspection
- No `__sklearn_is_fitted__`, `n_features_in_`, `feature_names_in_`,
  `__sklearn_tags__`, `_validate_data`, `_check_n_features`.
- `clone` exists at `base.py:130` but not exported.

### A5. No Pipeline / GridSearchCV compat
- Lifter/Calibrator have `lift`/`transform(dist)` not `transform(X)`.
- `GridSearch` (`search.py:44`) reimplements sklearn's; no `cv_results_`,
  `refit=`, custom scorers. Scorer is a string whitelist.

### A6. Naming collisions
- Bracket boundaries: `edges` / `cutpoints` / `strikes` / `taus` / `qvals` /
  `quantiles_`. Pick one term.
- `BracketEdges` enum (closed/open semantics) collides with `edges:` array
  field on the same dataclass.

### A7. `fit_predict` returns `PipelineResult`, not ndarray
- Sklearn convention violated. `predict()` on unseen X only works if
  `refit_on_full=True`.

---

## 3. Duplication / simplification

### S1. `bracket_probs_from_cdf` reinvented 3× — FIXED 2026-05-25
- New `forecast.bracket_probs_from_cdf_at_edges(cdf_at_edges, source)`
  helper handles the diff → clip → row-sum validation → normalise
  sequence once. Called from `CumulativeBinary.predict_dist` and
  `lift._bracket_probs_from_dist` (the latter is now a one-liner that
  calls `dist.cdf(edges)` then delegates).
- `TailSpecialist` *deliberately* doesn't use the helper — it diff-clips
  the upstream CDF, then *overrides* the outer bins with the classifier
  output and rescales the inner. The shared helper would normalise
  too early and break the override semantics; this is the intentional
  exception, called out in a code comment.

### S2. Per-row Python loops where vectorized works — FIXED 2026-05-25
- Isotonic-repair in `QuantileReg.predict_dist`,
  `QuantileForest.predict_dist`, `CumulativeBinary.predict_dist` now
  use one vectorised `np.maximum.accumulate(arr, axis=1)` call apiece.
  The `_isotonic_repair_row` helper became unused and was deleted.
- Bracket `cdf` / `pdf` collapsed from a per-x-query Python loop to a
  single `np.searchsorted` call; below/above-support handled by mask
  on the output. New `cdf_at` quantile branch follows the same pattern.
- Quantile `cdf` (and `cdf_at`): dead defensive `np.maximum.accumulate`
  inside the per-row loop removed — `from_quantiles` already enforces
  monotone qvals at construction time.
- `OnlineAggregator.predict`: per-row Python loop replaced with one
  pass of element-wise multiplies (awake mask × weight broadcast) and
  a row-wise sum. The error-on-coverage-hole logic is preserved.

### S3. Score registry hardcoded — FIXED 2026-05-25
- New module-level dispatch helpers `_metric_crps`, `_metric_log_score`,
  `_compute_metric` in `pipeline.py`. `PipelineResult.score` is now a
  3-line outer loop that delegates per-metric work to the registry.
- Adding a new backing means touching one function (`_metric_crps` or
  `_metric_log_score`), not four if/elif blocks.
- A future `make_scorer`-style public API would build on the same
  helpers; out of scope for this pass.

### S4. `GridSearch` is parallel to sklearn's — DECLINED 2026-05-25
- The audit suggested wrapping a custom `cv` object instead of
  rewriting the loop. On closer inspection, our `ForecastPipeline`
  owns its CV internally (expanding/rolling-window/kfold) and does
  not expose a sklearn-compatible splitter object. Passing CV via
  `cv=` would require restructuring the pipeline first — a larger
  change than the duplication it would remove.
- The 198-line `search.py` is self-contained, has its own grid-search
  tests, and the docstring explicitly justifies the choice
  ("sklearn.GridSearchCV would re-split with its own KFold, destroying
  time ordering").
- Keep as-is.

---

## 4. Non-intuitive (sklearn-user trip-ups)

- `from bracketlearn import EMOS` fails.
- `predict_dist` not discoverable.
- `ids=`/`timestamps=` mandatory → plain `(X, y)` raises.
- No `.score(X, y)` on estimator.
- `fit_predict` returns dict-like, not ndarray.
- `tail_policy` required on `from_quantiles`/`from_empirical` (loud, good —
  but surprising).
- `name=` on estimator *and* tuple-name in pipeline can diverge.

---

## 5. Missing features (prediction-market angle)

bracketlearn only goes model → `fair_price`. README claims to "keep going
past predict-a-distribution"; today it stops there. Missing:

### M1. Inverse direction (market → distribution)
- `from_market_prices(probs_per_bracket) -> DistributionForecast`.
- Lets users score model dist against market-implied dist.

### M2. Edge / sizing
- `edge(model_dist, market_prices, ladder)` per-bracket.
- Kelly / fractional-Kelly sizing helper consuming `ContractForecast`.

### M3. Calibration to market
- `calibrate_to_market(dist, market_prices, method='isotonic')` — pin model
  toward observed market in regions where market is informed.

### M4. Spread markets
- README mentions spread repeatedly; no `SpreadLadder` adapter, no
  asymmetric handling.

### M5. Tail policies
- Only `clip` implemented. README advertises `gpd`, `gaussian_match`.

### M6. Calibration plotting
- Scalars `pit_mean`/`pit_std` exist; no PIT histogram, no per-bracket
  reliability diagram.

### M7. `to_quote` stub
- `adapters.py:354`. `VenueSpec` (tick/multiplier/min_size) defined but
  unused.

### M8. Real prediction-market examples
- All three examples are toy regression. None uses spread or weather bracket
  data from the repo, none shows implied-price/edge.

---

## 6. Test coverage gaps

### T1. Untested estimators
- `Stacking` (only negative-path tests), `TailSpecialist` (none),
  `market_ols` / `emos_calibrated` factories (none).

### T2. Missing edge cases (every trainer)
- B=1, B=2 ladders.
- Single-row fit.
- NaN-in-X (only OnlineAggregator has a NaN test, failure path).
- `sample_weight` on baselines + 8 trainers.
- Multi-target × quantile-backing × bracket-scoring.

### T3. Property tests missing
- `sum(BracketLadder.price(dist)) == 1.0` for every (backing, tail_policy).
- Monotonicity of quantiles after fit.
- `clone(est).get_params() == est.get_params()`.

---

## 7. Docs gaps

- No guide for `score.py` metrics (CRPS/log-score/PIT/Brier semantics).
- No guide for adapters (`BracketLadder`, `BinaryAbove`, `VanillaCall`).
- No guide for baselines.
- No tail-policy guide.
- `docs/guides/examples.md` claims "5 runnable examples"; 2 are synthetic-only.
- No CHANGELOG.

---

## 8. Decision required: standalone vs internal

The biggest fork. Picking matters because remediation diverges:

| | Standalone PyPI lib | Repo-internal framework |
|---|---|---|
| Subclass `sklearn.base.BaseEstimator` | yes | optional |
| `ids`/`timestamps` mandatory | no — optional, inferred | keep, repo needs them |
| Rename `predict_dist` → `predict_proba` | yes | no |
| Deduplicate vs `prediction_market_weather/ml/trainers/*` | no (independent) | yes (cut shadow re-impl) |
| Wire into `ml/inference.py` | no | yes |
| Examples target | public datasets only | weather + spread + public |

Both is the current state and is unsustainable.

---

# Remediation plan — items 1–5

Numbered to match user request.

## 1. Fix qreg ladder-sum bug (B1)  — CORRECTNESS, ship first

**What.** `BracketLadder.price(dist)` rows must sum to 1.0 for every
backing (`bracket`, `quantile`, `normal`, `mixture_normal`) crossed with
every tail policy.

**Cause to nail down before patching.** Two candidates:
- `quantile`-backed `cdf` returns 0 below `q_min(τ_min)` and 1 above
  `q_max(τ_max)`; with `clip` policy mass outside outer τ is *clipped to
  point masses* but `from_quantiles` interprets τ-range as `[τ_min,τ_max]`,
  so `BracketLadder.price` reads CDF over ladder edges including outside
  the τ-range and gets `cdf(edges[k+1]) - cdf(edges[k]) = 0` for edges past
  `q_max`. Net: outer-bin mass is lost, not redistributed.
- Or `BracketLadder.price` itself integrates over `edges[0]..edges[-1]`
  ignoring tails — would give the same symptom.

**Approach.**
1. Add failing test `tests/test_ladder_sum.py`:
   - parametrize over backing × tail_policy × ladder shape (B∈{1,2,5,20}).
   - assert `np.allclose(ContractForecast.fair_price.reshape(N,B).sum(1), 1.0)`.
2. Trace the qreg case to confirm which integration is dropping mass.
3. Fix: either (a) `BracketLadder.price` adds explicit
   `tail_mass_below = 1 - cdf(edges[0])`, `tail_mass_above = cdf(edges[-1])`
   and folds them into outermost bins **iff** outermost edges are `±inf` or
   ladder is declared "closed"; otherwise raise; or (b) `from_quantiles`
   with `clip` policy reports tail mass that `BracketLadder.price`
   consumes; **no silent renormalization** (Rule #0.5).
4. Document semantics in `BracketLadder` docstring + `docs/guides/concepts.md`.

**Files.** `bracketlearn/adapters.py`, `bracketlearn/forecast.py`,
`bracketlearn/tail.py`, `bracketlearn/tests/test_ladder_sum.py` (new),
`bracketlearn/examples/housing_brackets.py` (update interpretation).

**Risk.** Touches the core math path. Test first, then patch.

## 2. Decision: standalone vs internal — DECIDED 2026-05-25 → STANDALONE

See `bracketlearn/DECISION.md` for the full write-up. Summary:

- bracketlearn ships as a standalone PyPI library.
- The repo's `prediction_market_weather/ml/trainers/*` stay independent;
  the duplication is permanent.
- API shape is judged by sklearn-user expectations.
- Audit items A1–A7 become actionable (full sklearn-contract target),
  not hedged.
- Items M1–M8 become library features justifying the
  "prediction-markets" keyword, not repo-integration tasks.

Item 4 in the remediation plan is upgraded accordingly (see end of
document).

## 3. Kill silent fallbacks (B2, B3, B5, B6, B10, plus stubs B8)

**What.** Replace every silent-fallback with a loud raise per Rule #0.5.

Concrete edits:
- `trainers.py:82-85` (`SklearnPoint`): replace bare `except TypeError`
  with `inspect.signature(self.estimator.fit).parameters` check.
- `trainers.py:240-243` (`Stacking`): assert
  `np.array_equal(ids, deps_oof[name].ids)`; raise on mismatch.
- `trainers.py:262-265` (`Stacking.sigma_`): drop `<=0 → 1e-3` fallback;
  fit `sigma_` on held-out slice (matches docstring intent).
- `trainers.py:838` (`CumulativeBinary` pad): require explicit outer edges.
- `trainers.py:946-1005` (`TailSpecialist`): drop `class_weight="balanced"`
  when user passes `sample_weight`; document the choice.
- `trainers.py:1296-1299` (`RNNHourly`): raise on unknown station ID.
- `lift.py:151-153, 176-178` (`Isotonic` row-sum guard): raise on all-zero
  row.
- `forecast.py` stubs + `adapters.py:354 to_quote`: every `return ...` →
  `raise NotImplementedError("…")` with TODO comment.

**Tests.** One negative test per fix asserting the raise.

**Files.** `bracketlearn/trainers.py`, `bracketlearn/lift.py`,
`bracketlearn/forecast.py`, `bracketlearn/adapters.py`,
`bracketlearn/tests/test_no_silent_fallbacks.py` (new).

## 4. Sklearn-contract upgrade — DONE 2026-05-25

Per `DECISION.md` (standalone library), upgraded scope from "minimal
pass" to full sklearn-shaped public API. Items 4a–4f below.

### 4a — Re-export from `__init__.py` ✅
- 67 names in `__all__`. Every estimator, lifter, calibrator, adapter,
  pipeline, search, multi-target wrapper, base + clone, all dataclasses.
- `from bracketlearn import EMOS, BracketLadder, ForecastPipeline, ...`
  works without digging into submodules.

### 4b — `predict_proba` alias — SKIPPED (per user)
- Adding `predict_proba` requires a ladder to know which "classes" to
  expose; bracketlearn's natural verb is `predict_dist` already.
  Documented in README that `predict_dist` is the entry point.

### 4c — Optional `ids=` / `timestamps=` ✅
- New `_auto_fill_ids_ts` decorator in `base.py`. `BaseEstimator.__init_subclass__`
  auto-wraps every subclass's `fit` / `predict` / `predict_dist` so
  callers may omit `ids=` / `timestamps=`; they auto-fill to
  `np.arange(N)` / `np.arange(N, dtype=float)`. Explicit kwargs still
  win.
- Plain `est.fit(X, y)` and `est.predict(X)` work everywhere now.

### 4d — sklearn introspection ✅
- `BaseEstimator.__sklearn_is_fitted__` walks `_`-suffixed attributes
  and returns True iff any is non-None. `sklearn.utils.validation.
  check_is_fitted(est)` works on every bracketlearn estimator.
- `_record_input_signature(X)` helper sets `n_features_in_` (and
  `feature_names_in_` when X is a pandas DataFrame). Wired into
  `SklearnPoint.fit`; other trainers will adopt it as the contract
  hardens.

### 4e — Inherit from `sklearn.base.BaseEstimator` ✅
- `bracketlearn.base.BaseEstimator` now subclasses sklearn's
  `BaseEstimator`. `isinstance(est, sklearn.base.BaseEstimator)`
  returns True; `sklearn.base.clone(est)` works.
- Caveat: `sklearn.utils.estimator_checks.check_estimator` will NOT
  pass — our `predict` returns `PointForecast`, `predict_dist`
  returns `DistributionForecast`, neither is an ndarray. The
  isinstance interop is the win; check_estimator compliance is a
  separate (likely infeasible) workstream.

### 4f — Sklearn-compat tests ✅
- `tests/test_sklearn_compat_v2.py` (11 tests):
  - top-level imports for every estimator
  - `issubclass(BaseEstimator, sklearn.base.BaseEstimator)`
  - `isinstance` checks for EMOS / EmpiricalDistribution / QuantileReg
  - `sklearn.base.clone` works
  - SklearnPoint / EmpiricalDistribution fit+predict without ids/ts
  - explicit ids still honored
  - `__sklearn_is_fitted__` + `check_is_fitted` flip after fit
  - `n_features_in_` set; `feature_names_in_` set when X is DataFrame

Suite: 206 pass (was 195, +11 new). Zero regressions.

## 5. Property tests + ladder-sum invariant (T2, T3)

**What.** Pin invariants so future refactors don't regress.

### DONE 2026-05-25 — `tests/test_invariants.py` (12 tests)

Lock-in tests for the audit-fixed state:

- `test_every_baseestimator_subclass_clones_with_equal_params` —
  walks every `BaseEstimator` subclass; for each constructable one,
  asserts `clone(est).get_params(deep=False)` has the same keys as
  the original.
- `test_clone_does_not_share_fitted_state` —
  fitted attribute on the original does not survive `clone()`.
- `test_clone_deep_copies_nested_estimators` —
  `CalibratedForecaster(EMOS(), Isotonic())` cloned → both nested
  estimators are fresh instances.
- `test_bracket_ladder_b1_single_bracket` / `test_bracket_ladder_b2_*` —
  degenerate ladder shapes (B=1, B=2) work; row sums == 1.
- `test_quantile_backing_qvals_monotone` — `from_quantiles` raises on
  qval crossings.
- `test_normal_dist_cdf_monotone_in_x` — parametric-normal CDF
  monotone non-decreasing.
- `test_bracket_dist_cumulative_probs_monotone` — bracket cumulative
  bin probs monotone.
- `test_fit_does_not_mutate_clone_source` — fitting a clone doesn't
  flip the original's `__sklearn_is_fitted__`.
- `test_empirical_distribution_clone_fits_independently` — two
  clones fit on disjoint data produce disjoint qvals.
- `test_persistence_predict_shape_matches_X` /
  `test_empirical_dist_predict_emits_same_qvals_per_row` —
  baseline shape contracts.

**Deferred** (documented in the test file header):
- `sample_weight` invariance (doubling a row weight ≈ duplicating
  the row). Linear/OLS estimators honor it exactly; tree-based
  trainers approximate. Needs dedicated tolerance-tuned suite.
- Single-row fit. Most trainers fail (ddof=1 σ, k-fold k≥2 …).
  Document as a known limitation rather than test.
- NaN-in-X. Behaviour varies by trainer; pin per-trainer in
  `test_trainers.py` rather than as a cross-cutting invariant.

Ladder-sum invariant (`sum(BracketLadder.price(dist)) == 1.0`) lives
in `tests/test_ladder_sum.py` from item 1 — not duplicated here.

---

## Sequencing

1. **B1 / item 1** — correctness, blocks everything that uses qreg→ladder.
2. **Decision / item 2** — branches the rest of the plan.
3. **Silent fallbacks / item 3** — Rule #0.5 cleanup, low risk, high signal.
4. **Export + minimal sklearn / item 4** — discoverability.
5. **Invariant tests / item 5** — lock the work in.

Items 6+ (the rest of §1–§7 above) re-prioritized after item 2 decision.
