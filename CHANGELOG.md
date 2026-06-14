# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions
follow semver: MAJOR.MINOR.PATCH. Pre-1.0 the public API can break in any
minor release; patch releases are bug-fixes and additive tests.

## [Unreleased]

**Breaking — composition API unified.** The two old composition syntaxes
(inside-out `LiftedForecaster`/`CalibratedForecaster` wrappers for chains;
a name-keyed `ForecastPipeline(steps=[(name, fc)], cv=…)` for CV + stacking)
collapse into three homogeneous, object-composed primitives:

- `Pipeline([stage, …], *, name=…)` — a sequential chain wired left→right by
  protocol type. Subsumes `LiftedForecaster` and `CalibratedForecaster`: a
  lifter or calibrator is just another stage in the list. A `Pipeline` *is* a
  `DistForecaster`.
- `Stacker([upstream, …], meta, *, name=…)` — a parallel combiner over
  upstream `Pipeline`/`Stacker` **objects**. The dependency *is* the nesting;
  there is no name-string `deps`/`deps_oof`. Meta-combiners receive their
  upstreams' OOF distributions positionally via `upstream=[…]`.
- `WalkForward(*, cv, n_folds, embargo, refit_on_full, …)` — the CV/OOF
  driver, split out of the old `ForecastPipeline`. `fit_predict(model, X, y,
  ids=…, timestamps=…)` returns a `PipelineResult`; `predict(…)` requires
  `refit_on_full=True`.

Names are now leaderboard labels only, never wiring.

Migration::

    # Old:
    from bracketlearn import (
        ForecastPipeline, LiftedForecaster, CalibratedForecaster,
    )
    pipe = ForecastPipeline(
        steps=[
            ("ridge", LiftedForecaster(SklearnPoint(RidgeCV()), GlobalResidual())),
            ("emos",  CalibratedForecaster(EMOS(), Isotonic(edges=edges))),
        ],
        cv="expanding-window", n_folds=5,
    )
    result = pipe.fit_predict(X, y, ids=ids, timestamps=ts)
    result.to_table(y, metrics=["log_loss_bracket"], ladder=BracketLadder(edges=edges))
    pipe.predict(X_new, ids=…, timestamps=…)

    # New:
    from bracketlearn import Pipeline, WalkForward
    ridge = Pipeline([SklearnPoint(RidgeCV()), GlobalResidual()], name="ridge")
    emos  = Pipeline([EMOS(), Isotonic(pre_integrate_edges=edges)], name="emos")
    wf = WalkForward(cv="expanding-window", n_folds=5, refit_on_full=True)
    result = wf.fit_predict([ridge, emos], X, y, ids=ids, timestamps=ts)
    result.to_table(y, metrics=["log_loss_bracket"], edges=edges)  # shared 1-D vector
    wf.predict(X_new, ids=…, timestamps=…)

Stacker migration — upstreams are objects, weights arrive positionally::

    # Old:  StackedParametric(deps=("ridge", "ngboost"))  + pipe injects deps_oof
    # New:
    stack = Stacker([ridge, ngboost], StackedParametric(), name="stack")
    wf.fit_predict(stack, X, y, ids=ids, timestamps=ts)

Multi-target / search migrate the same way: `MultiOutput(model, WalkForward(…))`
replaces `MultiOutputForecastPipeline(proto)`; `GridSearch(model, wf, …)` takes
the model and driver separately, with `refit_node=` (was `refit_stage=`) and
`edges=` (was `ladder=`) for bracket scoring.

### Added
- **Reference-relative value metrics** in `score` (Step-3 scoring extension):
  `edge_alignment(q, m, r)` (Edge-Alignment — the expected betting payoff
  `(q−m)(r−m)` of a price `q` against a reference `m`), `edge_alignment_corr`,
  `shared_bias_slope`, and `value_report` (EA plus its exact `EA = A − B`
  market-mispricing / non-orthogonality split). Bracket-ladder wrappers
  `edge_alignment_bracket` / `value_report_bracket` take a `ContractForecast`
  plus a reference ladder. These grade whether a price is more *valuable* than a
  quoted one (vs `brier_bracket`, which grades calibration); a more accurate
  price is not always a more valuable one. New guide
  `docs/guides/value_vs_accuracy.md`; tests in `tests/test_value_metrics.py`.

### Removed
- `bracketlearn.pipeline.ForecastPipeline`, `LiftedForecaster`,
  `CalibratedForecaster`
- the name-keyed `deps` / `deps_oof` stacker contract (and the dead
  `Forecaster.depends_on` field)
- `Stacking` (legacy alias for `StackedParametric` — use the canonical name)
- `BracketForecast.shared_edges()` (consume per-row `self.edges` directly)
- `Isotonic(edges=…)` constructor arg → `Isotonic(pre_integrate_edges=…)`
- bracket-metric `ladder=BracketLadder(edges=…)` → `edges=` (a shared 1-D
  edge vector; the score path builds the per-row ladder internally)

## [0.6.0] — 2026-05-28

**Breaking**: ``Backing`` and ``ParametricFamily`` enums removed, along
with the ``DistributionForecast.backing`` / ``.family`` properties on
the base class and all five subclasses. The enums were carried as
compat shims from v0.3.0 when ``DistributionForecast`` became an
``abc.ABC`` base; the abstract-class hierarchy made them redundant
the day they shipped. ``isinstance`` dispatch on the concrete subclass
is the supported API.

Migration::

    # Old:
    from bracketlearn.forecast import Backing, ParametricFamily
    if dist.backing == Backing.PARAMETRIC and dist.family == ParametricFamily.NORMAL:
        ...
    if dist.backing == Backing.BRACKET:
        ...

    # New:
    from bracketlearn import (
        BracketForecast, MixtureNormalForecast,
        NormalForecast, QuantileForecast, StudentTForecast,
    )
    if isinstance(dist, NormalForecast):
        ...
    if isinstance(dist, BracketForecast):
        ...

For checks that previously asked "is this any parametric backing?",
use a tuple of all three parametric subclasses::

    _PARAMETRIC = (NormalForecast, StudentTForecast, MixtureNormalForecast)
    if isinstance(dist, _PARAMETRIC):
        ...

### Removed
- ``bracketlearn.forecast.Backing``
- ``bracketlearn.forecast.ParametricFamily``
- ``DistributionForecast.backing`` (abstract property)
- ``DistributionForecast.family`` (default ``None``)
- ``.backing`` / ``.family`` ``@property`` overrides on
  ``NormalForecast``, ``StudentTForecast``, ``MixtureNormalForecast``,
  ``QuantileForecast``, ``BracketForecast``

### Internal
- ``StackedParametric``, ``BMAStacking``, ``BracketStacking`` upstream
  type checks switched from ``d.backing.value == "..."`` /
  ``d.backing != Backing.BRACKET`` to ``isinstance(d, ...)``.
- Test assertions updated to ``isinstance`` checks.

All 343 tests pass.

## [0.5.0] — 2026-05-27

**Breaking**: ``BracketClassifier`` and ``BracketRegressor`` removed.
Both classes conflated two concerns inside ``fit`` — the per-row to
per-(row, bracket) reshape and the model fit — and hardcoded the target
as a bracket-hit indicator. That made the regressor inflexible: any
caller wanting a different per-(row, bracket) target (e.g. mispricing
residual ``hit - market_p``) had to fork the class.

The two concerns now live separately:

- New ``bracketlearn.BracketExpander`` (in ``bracketlearn.transformers``)
  owns the reshape. ``fit_transform(X, y, ids=...)`` returns
  ``(X_expanded, y_expanded)`` where ``X_expanded`` is
  ``(M, F+2)`` with ``[..., lo, hi]`` appended, and ``y_expanded`` is
  the default bracket-hit target ``(M,)``. Pass ``y=None`` to skip
  target construction. ``transform(X, ids=...)`` is the predict-side
  counterpart (X-only). ``assemble_dist(predictions, ids=..., timestamps=...)``
  packs raw per-(row, bracket) predictions into a row-renormalised
  ``BracketForecast``.

- Fitting is plain sklearn: callers pick any classifier / regressor and
  call its ``fit(X_expanded, y_expanded)`` directly. The expander has
  no opinion about which model fits the augmented design.

Migration recipe::

    # Old:
    # bc = BracketClassifier(estimator=LGBMClassifier(...),
    #                        brackets_by_id=bbi).fit(X, y, ids=ids)
    # d = bc.predict_dist(X_pred, ids=pred_ids, timestamps=ts)

    # New:
    exp = BracketExpander(brackets_by_id=bbi)
    X_exp, y_exp = exp.fit_transform(X, y, ids=ids)
    clf = LGBMClassifier(...).fit(X_exp, y_exp)
    X_pred_exp, _ = exp.transform(X_pred, ids=pred_ids)
    scores = clf.predict_proba(X_pred_exp)[:, 1]
    d = exp.assemble_dist(scores, ids=pred_ids, timestamps=ts)

Callers wanting a custom target build it on top of the expansion::

    X_exp, y_hit = exp.fit_transform(X, y, ids=ids)
    market_p_exp = build_market_p_per_bracket(...)  # caller-side, (M,)
    X_exp = np.column_stack([X_exp, market_p_exp])
    y_target = y_hit - market_p_exp                  # mispricing residual
    LGBMRegressor(...).fit(X_exp, y_target)

The 16 ``BracketClassifier`` / ``BracketRegressor`` tests have been
replaced by 9 ``BracketExpander`` tests covering the same surface
(ragged brackets, default + custom targets, predict-time id mismatch,
length-mismatch guards, end-to-end logistic-regression composition).

### Removed
- ``bracketlearn.trainers.BracketClassifier``
- ``bracketlearn.trainers.BracketRegressor``
- ``bracketlearn.trainers.bracket._assemble_bracket_forecast`` (subsumed
  by ``BracketExpander.assemble_dist``)

### Added
- ``bracketlearn.transformers.BracketExpander``
- ``bracketlearn.BracketExpander`` top-level re-export

## [0.4.0] — 2026-05-27

Three Bayesian trainers added — one with empirical wins on this repo's
domain, one that ties the existing baseline, one that didn't justify
its structural pitch but ships as an alternative. Pipeline grows a
``groups`` kwarg so site-aware trainers compose with the existing CV
machinery. New ``BracketClassifier`` / ``BracketRegressor`` pair
unifies "predict bracket-resolves-YES with any sklearn classifier or
regressor". Two internal refactors: monolithic ``forecast.py`` split
into the ``forecast/`` subpackage (typed subclasses + helpers);
monolithic ``trainers.py`` split into the ``trainers/`` subpackage
(grouped by output shape). Both are pure code-organisation changes;
all public imports (``from bracketlearn import …``, ``from
bracketlearn.forecast import …``, ``from bracketlearn.trainers import
…``) are unchanged.

### Added

- ``bracketlearn.trainers.BracketClassifier`` — single binary
  classifier on ``[X, lo, hi]`` features with target
  ``1[y ∈ [lo, hi))``. Any sklearn-style classifier with
  ``predict_proba`` works (Logistic, GradientBoosting, LGBM, RF, MLP).
  Augments one training example per (row, bracket) pair; row-
  renormalises predict-time probabilities across the row's bin grid
  → BracketForecast. Supports ragged per-row bracket counts via
  ``brackets_by_id`` (id → 1-D edge array). Loud rails: estimator
  without ``predict_proba`` raises at construction; all-zero
  augmented labels raise at fit; unregistered ids raise at predict;
  non-monotonic edges raise at construction. Empirical: with a
  tree-based classifier matches ``CumulativeBinary`` within ~10%
  CRPS on a synthetic nonlinear benchmark, beats EMOS-discretised
  and ``CDFBoostBracket(EMOS dep)``; with a linear classifier acts
  as a Gaussian-ish floor. Sells flexibility — same trainer, any
  classifier — rather than peak accuracy. See bench
  ``/tmp/bracket_classifier_bench.py`` for numbers.
- ``bracketlearn.trainers.BracketRegressor`` — regressor-sibling of
  ``BracketClassifier``. Same augmentation (``[X_i, lo_b, hi_b]``)
  and same target ``1[y_i ∈ [lo_b, hi_b))``, but fits any sklearn-
  style regressor (``fit`` + ``predict``) instead of a classifier.
  Raw scores are clipped to ``[clip_eps, 1-clip_eps]`` and row-
  renormalised across the row's bin grid → BracketForecast. Useful
  when the estimator family ships only ``predict`` (Ridge,
  ElasticNet, GradientBoostingRegressor, LGBMRegressor, MLPRegressor,
  custom GAMs), or when squared-error loss on the bracket-hit target
  is preferable to cross-entropy. Trade-off: regressor outputs aren't
  constrained to ``[0, 1]`` — clipping + row-normalisation lose the
  calibration logistic-style classifiers get for free. Same loud
  rails as ``BracketClassifier``. Shares the ``_augment_with_bracket_
  bounds`` and ``_assemble_bracket_forecast`` helpers with the
  classifier so behaviour stays in lockstep.
- ``bracketlearn.trainers.BayesianRidge`` — conjugate
  Normal-Inverse-Gamma Bayesian linear regression. Predictive per row
  is Student-t with ``ν = 2·a_n``; predictive σ inflates via
  ``(1 + xᵀ V_n x)`` so rows far from training data automatically get
  wider intervals. Closed-form fit, no sampler dependency. Raises on
  zero-variance columns, singular posterior precision, degenerate
  ``b_n`` (data leak / over-tight prior). Standardisation on by
  default; intercept fitted with a near-flat prior
  (``prior_precision_intercept=1e-6``).
- ``bracketlearn.trainers.BMAStacking`` — meta-learner alternative to
  ``StackedParametric``. Dirichlet-prior weights on the
  K-component mixture of upstream Normals (moment-matched from
  any parametric backing). EM with Dirichlet pseudo-counts;
  convergence on Δ log-likelihood. Output is a true
  ``MixtureNormalForecast``, so per-row σ inflates when upstream μ̂'s
  disagree. Empirical caveat: on this repo's regression scenarios
  BMA ties ``StackedParametric(sigma_method='geometric_mean_upstream')``
  and loses to ``StackedParametric`` when upstream disagreement is
  structural bias correctable by an OLS negative coefficient — ships
  as an option, not a replacement.
- ``bracketlearn.trainers.HierarchicalNormal`` — cross-site
  partial-pooling regression. Per-site coefficients ``β_s`` shrunk
  toward a common ``β₀`` with variance components ``(σ², τ²)``
  estimated by empirical-Bayes (Type-II marginal likelihood; Nelder-
  Mead over log-σ², log-τ²; ``β₀`` profiled out by GLS). Per-site
  posterior on ``β_s`` closed-form; predictive at row i in site s is
  Normal with mean ``xᵀ E[β_s]`` and variance ``σ² + xᵀ Cov(β_s) x``.
  Unseen sites raise by default (``allow_unseen_sites=False``); when
  enabled, predictive uses ``β₀`` with ``V_β₀ + τ² I`` added to
  reflect the missing per-site data. Empirical wins on imbalanced-N
  scenarios — see commit benchmark for paired-bootstrap CRPS CIs.
  Woodbury identity keeps per-site fit O(K³) regardless of n_s.
- ``ForecastPipeline.fit_predict(...)`` and ``.predict(...)`` accept
  ``groups: np.ndarray | None``, threaded through fold slicing and
  the canonical-refit path to any stage whose ``fit`` / ``predict_dist``
  signature declares it. Existing trainers (``EMOS``,
  ``StackedParametric``, ``BayesianRidge``, …) ignore ``groups``
  via the same signature-introspection routing that already handles
  ``sample_weight`` / ``deps_oof`` / ``ids`` / ``timestamps``. No
  behaviour change for callers that don't pass ``groups``.
- Internal helper ``bracketlearn.pipeline._predict_with_extras`` —
  generalises the deprecated ``_predict_with_deps`` to thread any
  predict-time kwarg (``deps_oof``, ``groups``, future ones) through
  signature introspection. ``_predict_with_deps`` kept as a
  back-compat wrapper.

### Changed

- Internal: ``bracketlearn/forecast.py`` split into the
  ``bracketlearn/forecast/`` subpackage (``base``, ``parametric``,
  ``quantile``, ``bracket``, ``contract``, ``_meta``, ``_helpers``).
  Pure code-organisation; public re-exports unchanged. See commit
  70b8231.
- Internal: ``bracketlearn/trainers.py`` split into the
  ``bracketlearn/trainers/`` subpackage grouped by output shape
  (``point``, ``parametric``, ``quantile``, ``bracket``, ``meta``)
  with shared utilities in ``_common`` and convenience factories in
  ``_factories``. Pure code-organisation; public re-exports
  unchanged. See commit c29abdc.
- ``BracketClassifier`` refactored to share its augmentation +
  output-assembly helpers with the new ``BracketRegressor``
  (``_augment_with_bracket_bounds`` and ``_assemble_bracket_forecast``
  in ``bracketlearn.trainers._common`` / ``bracketlearn.trainers.
  bracket``). Behaviour unchanged; tests still green.

### Tests

- 7 ``BracketClassifier`` unit tests (BracketForecast shape under
  LR estimator, ragged per-row brackets, in-sample mode accuracy
  with LGBM on tight signal, rejects regressor estimator without
  ``predict_proba``, rejects non-monotonic edges, raises on
  missing-id predict, raises when no y lands in any bracket,
  raises on predict-before-fit).
- 7 ``BracketRegressor`` unit tests mirroring the classifier suite
  (BracketForecast shape under Ridge, ragged per-row brackets,
  in-sample mode accuracy with LGBMRegressor on tight signal,
  rejects estimator without ``.predict``, rejects non-monotonic
  edges, raises on missing-id predict, raises when no y lands in
  any bracket, raises on predict-before-fit).
- 6 ``BayesianRidge`` unit tests (parametric/student_t shape,
  coefficient recovery, distance-based σ inflation, zero-variance /
  collinearity / predict-before-fit raises).
- 4 ``BMAStacking`` unit tests (round-trip + mixture-normal shape,
  σ inflation under upstream disagreement, misaligned-ids guard,
  invalid α prior raise).
- 7 ``HierarchicalNormal`` unit tests (variance-component recovery,
  Normal-predictive shape, unseen-site raise, σ inflation on unseen
  site, beats per-site Ridge on imbalanced-N benchmark, requires
  groups, requires ≥2 sites).
- 1 ``ForecastPipeline`` integration test for ``groups`` routing
  (fit_predict + predict + missing-groups raise).

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
