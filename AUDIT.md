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
- Body rescaling shape concern (`body_probs[:, 1:-1]` discarding edge
  bin mass on narrow ladders) — DEFERRED. Documented as a known
  modelling limitation; the rescaling matches the existing trainer's
  semantics. Revisit when a narrow-ladder test case actually exists.

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

### B7. Persistence baseline holds at last value after `lag` rows
- `baselines.py:174-182`: emits `tail_y_[:lag]` then `tail_y_[-1]` repeated.
- `examples/bike_sharing_timeseries.py:99` markets it as "diurnal cycle";
  output shows identical `persist24` across rows.
- Either fix predict semantics (rotate by test-row index) or fix the example
  claim.

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

### B9. `pit` builds full (N,N) matrix then `np.diag`
- `score.py:84`: 800 MB at N=10k. Should be per-row `cdf(y_i)`.

### B10. Other silent fallbacks — PARTIALLY FIXED 2026-05-25
- `lift.py` `Isotonic.transform` and `_bracket_probs_from_dist` row-sum
  guards now raise `ValueError` with a row-count diagnostic. No more
  silent uniform substitution.
- `EMOS.fit` MoM negative-variance issue — DEFERRED. The clip at
  predict time is *visible* (uses `np.maximum(var, floor)`) and the
  fix is independent. Track separately.
- Test: `test_bracket_probs_from_dist_raises_on_zero_row_sum`.

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

### S1. `bracket_probs_from_cdf` reinvented 3×
- `CumulativeBinary` (trainers.py:847-850), `TailSpecialist` (985-988),
  `lift.Isotonic` (lift.py:151-153, 176-178). Extract one helper.

### S2. Per-row Python loops where vectorized works
- Isotonic-repair: `trainers.py:628-629, 713-714, 824-825` →
  `np.maximum.accumulate(arr, axis=1)`.
- Bracket `cdf`/`pdf`: `forecast.py:339-350, 426-432` → `searchsorted`.
- Quantile `cdf`: `forecast.py:361-373` (defensive monotone check inside the
  loop is dead — `from_quantiles` already enforces it).
- `OnlineAggregator.predict`: `trainers.py:1114-1122`.

### S3. Score registry hardcoded
- `PipelineResult.score` (`pipeline.py:129-167`) is if/elif over metric
  names. `bracketlearn.score` has CRPS/log-score/PIT/Brier as raw functions
  but no `make_scorer`-style registry.

### S4. `GridSearch` is parallel to sklearn's
- Rationale (search.py:1-9) is real but partial: wrap a custom `cv` object
  instead of rewriting the loop.

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

## 4. Export estimators + minimal sklearn-ism (A1, A2, A4 partial)

**What.** Fix import surface and the most embarrassing sklearn-isms. Full
sklearn-contract migration depends on item 2's decision.

Minimal pass:
- Re-export from `__init__.py`: every estimator class, `BaseEstimator`,
  `clone`, `ForecastPipeline`, `GridSearch`, `BracketLadder`, all metrics.
- Add `__sklearn_is_fitted__` to `BaseEstimator` (delegates to
  `hasattr(self, "<sentinel>_")`).
- Store `n_features_in_` and `feature_names_in_` after fit (one helper in
  `base.py`, call from every estimator's `fit`).
- Add `predict_proba` *alias* to `DistForecaster` that calls `predict_dist`
  and returns the `(N, B)` bracket-prob array directly — minimal change,
  big discoverability win. Original `predict_dist` stays for callers that
  want the full `DistributionForecast` object.

Defer (depends on item 2):
- Subclassing `sklearn.base.BaseEstimator`.
- Dropping `ids=`/`timestamps=` kwargs.
- `Pipeline` / `cross_val_score` compat.

**Files.** `bracketlearn/__init__.py`, `bracketlearn/base.py`,
`bracketlearn/protocols.py`, every estimator file (for `n_features_in_`),
new test `tests/test_export_surface.py`.

## 5. Property tests + ladder-sum invariant (T2, T3)

**What.** Pin invariants so future refactors don't regress.

- `tests/test_invariants.py`:
  - For every (estimator, backing) combo: `predict_dist(...).probs` (or
    derived) rows sum to ~1, are non-negative, quantiles monotone.
  - `clone(est).get_params() == est.get_params()`, no shared mutable state.
  - `sum(BracketLadder.price(dist)) == 1.0` (covered by item 1, lifted here
    as the canonical invariant test).
- `tests/test_edges.py`:
  - B=1, B=2 ladders.
  - Single-row fit.
  - NaN-in-X for tree-based trainers (tolerate) vs linear (raise).
- `tests/test_sample_weight.py`:
  - Each estimator: doubling a row's weight ≈ duplicating the row (within
    tolerance) for fit-time outputs.

**Files.** `bracketlearn/tests/test_invariants.py` (new),
`bracketlearn/tests/test_edges.py` (new),
`bracketlearn/tests/test_sample_weight.py` (new).

---

## Sequencing

1. **B1 / item 1** — correctness, blocks everything that uses qreg→ladder.
2. **Decision / item 2** — branches the rest of the plan.
3. **Silent fallbacks / item 3** — Rule #0.5 cleanup, low risk, high signal.
4. **Export + minimal sklearn / item 4** — discoverability.
5. **Invariant tests / item 5** — lock the work in.

Items 6+ (the rest of §1–§7 above) re-prioritized after item 2 decision.
