# bracketlearn — direction decision

Date: 2026-05-25
Status: **decided — standalone PyPI library**
Author: audit + remediation discussion, item 2 of AUDIT.md plan.

---

## The question

bracketlearn shipped phases A→D as if it were a standalone PyPI package
(MIT-licensed pyproject, public GitHub URL in docs, "pip install
bracketlearn", Read the Docs config, isolated CI workflow). At the same
time, its 14 trainer classes are *name-for-name* re-implementations of
`prediction_market_weather/ml/trainers/*` (EMOS, QuantileReg, Stacking,
TailSpecialist, OnlineAggregator, RNNHourly, MixtureNormals,
NGBoostNormal, CumulativeBinary, ConformalCalibrate, MarketOLS,
EMOSCalibrated, …).

Two coherent stories. Either:

- **(A) Standalone PyPI library.** bracketlearn is an external,
  general-purpose probabilistic-forecasting + bracket-contract-pricing
  package. The repo's production trainers are independent and stay
  where they are. Shared *ideas*, not shared code.
- **(B) Repo-internal framework.** bracketlearn is the next generation
  of `prediction_market_weather/ml/trainers/_framework.py`. The
  duplication is a port in progress; eventually the production
  trainers either import bracketlearn or get retired.

"Both" is what we have today, and it's the source of most structural
audit findings (mandatory `ids`/`timestamps` kwargs that block sklearn
interop, predict_dist instead of predict_proba, custom `BaseEstimator`
not subclassing sklearn's, name collisions, untested factory functions
matching trainers nobody calls).

---

## Evidence

### Signals pointing standalone

1. **Packaging is PyPI-shaped.**
   `bracketlearn/pyproject.toml` declares `name = "bracketlearn"`,
   `version = "0.2.0"`, MIT license, Trove classifiers ("Development
   Status :: 3 - Alpha", "Intended Audience :: Science/Research"),
   keywords (`forecasting`, `probabilistic-forecasting`,
   `prediction-markets`, …), and split extras
   (`[boosting]`, `[quantile-forest]`, `[rnn]`, `[demo]`). The build
   backend is configured. There is no setuptools-find-packages
   exclusion linking it to the rest of the monorepo.

2. **Docs target an external audience.**
   `README.md` opens with `pip install bracketlearn`.
   `docs/conf.py:37` sets
   `source_repository = "https://github.com/frederikbenirschke/bracketlearn"`
   — a *separate* GitHub repo URL, not this one. Sphinx + Furo + Read
   the Docs config are configured as if for a public package.

3. **CI is isolated.**
   `.github/workflows/bracketlearn-ci.yml` runs *only* against
   `bracketlearn/`. The repo's main `ci.yml` does not invoke
   bracketlearn's test suite.

4. **Recent commit trajectory is package-maturation work**, not
   integration work:
   ```
   67a2217 v0.1 forecasting + bracket-pricing framework
   bd7aedd tier-1 trainers
   10c52b7 tier-2 trainers
   f4ecf4c tier-3 trainers
   7963d44 phase A — packaging, test suite, stub pruning
   f9723d2 phase B — sklearn contract + predict-on-unseen
   76318e1 phase C — CV variants, sample weights, multi-target, grid search
   bd33460 Sphinx + Read the Docs
   d265434 phase D — GitHub Actions CI + pickling persistence
   b8d42f9 docs — 3 public-dataset examples + examples guide
   ```
   Every phase is "make it a better library", none is "integrate with
   the repo".

5. **Examples target public datasets.**
   `examples/{housing_brackets,bike_sharing_timeseries,grid_search_demo}.py`
   use sklearn datasets and synthetic data. None pulls from
   `experiments/ml/feature_matrix_*.parquet` or any repo-internal
   source. There is no weather or spread-market example.

6. **The repo just explicitly rejected unification.**
   `docs/weather/ml_inference_and_unification.md:1-13` (dated
   2026-05-24, one day before this audit) writes:

   > Status: design v3, 2026-05-24. Scope cut after honest audit:
   > **no framework unification**, just inference. […]
   > Audit conclusion: those trainers work, OOF parquets land correctly,
   > eval is registry-driven. Refactoring them to add inference risks
   > breaking 17 working models for a stylistic win. **Inference can be
   > added without touching the existing trainers' fit logic.**

   The repo has consciously decided that the production trainers stay
   as-is. That kills story (B).

### Signals pointing internal

1. **Class-name overlap with production trainers** (14 names match
   1:1). Suggests a port was once intended.
2. **`Stacking.fit` consumes `deps_oof`** in the same shape that
   `_framework.run_trainer` produces. Strongly hints at copy-port.
3. **No imports of bracketlearn outside the package.** This is a
   *neutral* signal: consistent with both "early standalone, not yet
   used internally" and "stalled internal port that never got wired
   up".

### Signals against internal

1. **Production trainer signatures don't match bracketlearn's.**
   bracketlearn `fit(X, y, *, ids, timestamps, sample_weight=…,
   deps_oof=…)` vs production `fit_one_target(...)` /
   `run_trainer(...)` — totally different shapes, both already
   load-bearing in their own ecosystem. Porting would mean breaking
   the production CLI flow.
2. **Production output is parquet to `experiments/ml/`, consumed by
   `scripts/paper_trader/`.** bracketlearn produces
   `DistributionForecast` Python objects with no parquet
   serialisation. Wiring bracketlearn into the production loop would
   require a full IO layer that doesn't exist.
3. **The repo's CLAUDE.md and the production trainers' eval code are
   built around a parquet + registry pattern, not a Python-object
   pipeline.** bracketlearn would need to learn that pattern, not the
   reverse.

---

## Decision

**Standalone PyPI library.** Story (A).

Reasoning:

- The evidence is asymmetric. Five distinct standalone signals (1–5
  above) plus an explicit "no unification" decision by the repo
  itself one day before this audit. The only signal *for* internal —
  the class-name overlap — is consistent with "ideas borrowed from
  production trainers as starting point" without implying a port.
- The migration cost of (B) is high (parquet IO, registry, planner,
  CLI integration, ~2000 LOC churn per the unification doc) and the
  repo already weighed and rejected that cost.
- The migration cost of (A) is zero: bracketlearn keeps doing what
  it's doing. The audit's structural findings then get a clear
  target — sklearn-shaped public API — instead of trying to satisfy
  two competing standards.

---

## What this means concretely

### Things that change

1. **API shape is judged by sklearn-user expectations**, not by
   "what does the repo's production code already do". This makes the
   audit's items A1–A7 actionable:
   - `__init__.py` re-exports every estimator (item A1).
   - Add `predict_proba` alias to `DistForecaster` returning bracket
     probabilities directly (item A2).
   - Make `ids=` / `timestamps=` *optional* with sensible defaults
     (`np.arange(N)`, `np.arange(N, dtype=float)`) so plain
     `(X, y)` calls work — required by `sklearn.Pipeline` and
     `cross_val_score` (item A3).
   - Add `__sklearn_is_fitted__`, `n_features_in_`,
     `feature_names_in_` (item A4).
   - Eventually subclass `sklearn.base.BaseEstimator` directly; the
     custom `BaseEstimator` shim becomes a thin wrapper or is
     deleted (item A4 cont.).

2. **Examples diversify away from the repo's data.** Three public
   examples already exist. No weather/spread example needed inside
   bracketlearn — those belong in the consumer repo, not the library.

3. **The duplication with `prediction_market_weather/ml/trainers/*`
   is permanent.** Audit findings about the production trainers
   (Stacking row-alignment, TailSpecialist class_weight,
   CumulativeBinary pad, RNNHourly clip) are independent issues to
   fix in *that* package, not in bracketlearn. The two suites
   diverge from here.

4. **Audit items M1–M8** (market-edge helpers, calibrate-to-market,
   spread-market adapter, real prediction-market examples) become
   *library features that justify the "prediction-markets" keyword
   in pyproject*, not repo-integration tasks.

### Things that stay the same

- Sticky audit items B2–B10 (silent fallbacks, Stacking sigma,
  TailSpecialist body rescaling, RNNHourly station clip) — all
  Rule #0.5 fixes inside bracketlearn. Same fixes either way.
- All test-coverage and docs items (T1–T3, §7) — same either way.
- B1 (already fixed) — same either way.

### Things bracketlearn explicitly does NOT promise

- **Not a port target for `prediction_market_weather/ml/trainers`.**
  The repo's production trainers stay where they are. If a future
  decision flips this, that's a fresh design doc.
- **Not a market-data client.** bracketlearn does not fetch
  Kalshi/Polymarket prices. Market-facing helpers (M1–M3) take
  arrays in / arrays out; the caller fetches.
- **Not a backtester.** Pipeline scoring is OOF distribution + bracket
  scoring; PnL backtesting belongs to the consuming repo.

---

## Implications for remediation plan items 3–5

**Item 3 (kill silent fallbacks).** Unchanged. All flagged
silent-fallback sites are in bracketlearn-internal code. Standalone
makes the Rule #0.5 alignment with the repo a *style choice imported
from the host project* — bracketlearn could in principle drop it, but
since we wrote it that way, keep it. Document it in the bracketlearn
contributing guide.

**Item 4 (export + minimal sklearn-ism).** *Becomes more ambitious*
under standalone. The "minimal pass" in the plan was hedged because of
the unresolved direction. Now we can target the full sklearn contract
without worrying about repo-side callers. Plan upgrade:

- Re-export every estimator + `BaseEstimator` + `clone`.
- Add `predict_proba` (alias) — discoverability win.
- Add `__sklearn_is_fitted__`, `n_features_in_`, `feature_names_in_`.
- **Make `ids=` / `timestamps=` optional** (auto-fill with `arange`
  if not provided) so `est.fit(X, y).predict(X)` works.
- **Subclass `sklearn.base.BaseEstimator`** at the bottom of the
  inheritance chain. The custom `BaseEstimator` becomes a thin
  bracketlearn-specific layer adding `clone` semantics for nested
  estimators.
- Run `sklearn.utils.estimator_checks.check_estimator` on each
  exported estimator in CI. Failures are bugs to fix, not warnings.

**Item 5 (invariant tests).** Unchanged. The ladder-sum invariant (B1)
is already in place; the rest is sklearn-clone, monotonicity, B=1/B=2
edge cases.

---

## Pivot trigger

If a future need emerges to wire bracketlearn into the repo's
production loop, the trigger is: a new model #18 or #19 that's easier
to express in bracketlearn's pipeline than in the production
framework. At that point we revisit (B) — but with a much smaller
scope (one model, not a port of 17). Until then, bracketlearn is an
external library that the repo happens to live next to.
