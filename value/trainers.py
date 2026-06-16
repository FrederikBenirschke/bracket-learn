"""Value-tilted bracket trainers: optimize ``L = CE − λ·EA`` against a reference
price, instead of calibration alone.

Both are **bracket-native** (per-(row, bracket) binary, like
``CumulativeBinary`` / ``BracketExpander``) and both need, in addition to the
usual per-row grids, a **reference price per bracket** at fit time — the price
``m`` whose mispricing the tilt chases. That reference is the one thing that
separates these from every other trainer, and the reason they live in
``bracketlearn.value`` rather than ``bracketlearn.trainers``: training toward
value-vs-a-market is a step past pure forecasting.

The reference is used **only in the loss**, so ``predict_dist`` needs no market
data — the fitted model maps features → a value-tilted bracket distribution. The
implied edge ``q − m`` and its costed value are computed at scoring time with
``bracketlearn.score`` (``edge_alignment``, ``edge_alignment_costed``).

Data contract — construction is hyperparameters only
----------------------------------------------------
The constructor takes **only hyperparameters** (``lam`` and engine knobs). The
per-row market data — the bracket ladders ``brackets_by_id`` and the reference
prices ``reference_by_id`` — flows alongside ``X`` / ``y`` at call time::

    model = BlendedBracketGBM(lam=2.0)
    model.fit(X_tr, y_tr, ids=ids_tr,
              brackets_by_id=bbi, reference_by_id=rbi)
    dist = model.predict_dist(X_te, ids=ids_te, timestamps=ts_te,
                              brackets_by_id=bbi)

The dicts are keyed by id and may cover *more* ids than any single call — ``fit``
/ ``predict`` select the subset they need by the ``ids`` you hand them.
``reference_by_id`` is needed only at fit (the loss); ``predict`` needs only
``brackets_by_id`` (to assemble the dist). Under ``WalkForward`` you pass the
full dicts once and they are forwarded **verbatim** to every fold::

    WalkForward(n_folds=5).fit_predict(
        model, X, y, ids=ids, timestamps=ts,
        brackets_by_id=bbi, reference_by_id=rbi)

Select ``lam`` by costed value, not EA. See ``docs/guides/value_with_fees.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import DistributionForecast
from bracketlearn.trainers._common import _validate_brackets_by_id
from bracketlearn.transformers import BracketExpander
from bracketlearn.value.objective import (
    _HESS_FLOOR,
    _sigmoid,
    ea_scale_for_reference,
    make_lgb_objective,
)


def _validate_value_inputs(
    brackets_by_id: dict[Any, np.ndarray],
    reference_by_id: dict[Any, np.ndarray],
    *,
    owner: str,
) -> None:
    """Validate the fit-time grids/references, loudly and early.

    The grids and reference prices arrive as ``fit`` keyword arguments (the
    constructor is hyperparameters only), so wrong lengths or non-finite prices
    surface here at the top of ``fit`` rather than deep in the LightGBM / torch
    loop. The dicts may cover more ids than the call uses; ``fit`` selects its
    subset by the ``ids`` it is handed.
    """
    _validate_brackets_by_id(brackets_by_id, owner=owner)
    if not isinstance(reference_by_id, dict) or not reference_by_id:
        raise ValueError(
            f"{owner} needs a non-empty reference_by_id dict "
            "(id → 1-D reference-price array)"
        )
    for k, ref in reference_by_id.items():
        ref_arr = np.asarray(ref, dtype=float)
        if ref_arr.ndim != 1 or ref_arr.size == 0:
            raise ValueError(
                f"reference_by_id[{k!r}] must be 1-D non-empty; got shape "
                f"{ref_arr.shape}"
            )
        if not np.all(np.isfinite(ref_arr)):
            raise ValueError(
                f"reference_by_id[{k!r}] has non-finite prices; a NaN reference "
                f"would silently produce NaN gradients (mask unquoted brackets "
                f"out before constructing the trainer)"
            )
        if k in brackets_by_id:
            n_bins = np.asarray(brackets_by_id[k], dtype=float).size - 1
            if ref_arr.size != n_bins:
                raise ValueError(
                    f"reference_by_id[{k!r}] has {ref_arr.size} prices but its "
                    f"ladder has {n_bins} brackets"
                )


def _import_torch():
    """Import torch with the libomp-duplicate guard, mirroring trainers/point.py
    (the macOS abort this avoids fires before any user code runs otherwise)."""
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch

    return torch


def _aligned_reference(
    expander: BracketExpander, ids: np.ndarray, reference_by_id: dict[Any, np.ndarray],
) -> np.ndarray:
    """Flatten ``reference_by_id`` into the expander's (row, bracket) order using
    the offsets recorded by the most-recent ``fit_transform`` / ``transform``."""
    if expander.offsets_ is None or expander.per_row_edges_ is None:
        raise RuntimeError("expander has no offsets; call fit_transform first")
    M = int(expander.offsets_[-1])
    m_exp = np.empty(M, dtype=float)
    missing = []
    for i, rid in enumerate(ids):
        if rid not in reference_by_id:
            missing.append(rid)
            continue
        ref = np.asarray(reference_by_id[rid], dtype=float)
        B_i = expander.per_row_edges_[i].size - 1
        if ref.shape[0] != B_i:
            raise ValueError(
                f"reference_by_id[{rid!r}] has {ref.shape[0]} prices but row has "
                f"{B_i} brackets"
            )
        if not np.all(np.isfinite(ref)):
            raise ValueError(
                f"reference_by_id[{rid!r}] has non-finite prices; a NaN reference "
                f"would silently produce NaN gradients (mask unquoted brackets out "
                f"before fitting)"
            )
        s = int(expander.offsets_[i])
        m_exp[s : s + B_i] = ref
    if missing:
        raise KeyError(
            f"reference_by_id missing {len(missing)} id(s); first: {missing[:3]}"
        )
    return m_exp


@dataclass
class BlendedBracketGBM(BaseEstimator):
    """LightGBM bracket model trained on ``L = CE − λ·EA`` via a custom objective.

    Construction takes **hyperparameters only** — ``lam`` (the value tilt; ``0``
    = pure CE) plus LightGBM knobs mirroring ``bl_bracket_classifier``'s
    regularized low-N defaults. The per-row market data is passed at call time
    (see the module docstring's data contract)::

        fit(X, y, *, ids, brackets_by_id, reference_by_id)
        predict_dist(X, *, ids, timestamps, brackets_by_id)

    ``brackets_by_id`` maps ``id -> 1-D edge array`` (the bracket ladder);
    ``reference_by_id`` maps ``id -> 1-D price array`` (length ``len(edges) - 1``,
    the reference price per bracket, used only in the fit loss). Both may cover
    more ids than a single call uses; ``fit`` / ``predict`` subset by ``ids``. A
    ``WalkForward`` forwards these dicts for you.
    """

    lam: float = 1.0
    n_estimators: int = 120
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 100
    reg_lambda: float = 20.0
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    hess_floor: float = _HESS_FLOOR
    name: str = "BlendedBracketGBM"

    booster_: Any = field(default=None, init=False, repr=False)
    _requires_explicit_ids = True   # grids keyed by id; never auto-fill arange(N)

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(f"{self.name}: lam must be non-negative; got {self.lam}")

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, ids: np.ndarray,
        brackets_by_id: dict[Any, np.ndarray],
        reference_by_id: dict[Any, np.ndarray],
    ) -> Self:
        import lightgbm as lgb

        _validate_value_inputs(brackets_by_id, reference_by_id, owner=self.name)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        exp = BracketExpander(brackets_by_id=brackets_by_id)
        X_exp, y_exp = exp.fit_transform(X, y, ids=ids)
        m_exp = _aligned_reference(exp, ids, reference_by_id)
        dset = lgb.Dataset(X_exp, label=y_exp, free_raw_data=False)
        params = dict(
            num_leaves=self.num_leaves, learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples, reg_lambda=self.reg_lambda,
            feature_fraction=self.feature_fraction, bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq, verbose=-1,
            objective=make_lgb_objective(m_exp, self.lam, hess_floor=self.hess_floor),
        )
        self.booster_ = lgb.train(params, dset, num_boost_round=self.n_estimators)
        return self

    def predict_dist(
        self, X: np.ndarray, *, ids: np.ndarray, timestamps: np.ndarray,
        brackets_by_id: dict[Any, np.ndarray],
    ) -> DistributionForecast:
        if self.booster_ is None:
            raise RuntimeError(f"{self.name}.predict_dist called before fit")
        exp = BracketExpander(brackets_by_id=brackets_by_id)
        X_exp, _ = exp.transform(np.asarray(X, dtype=float), ids=np.asarray(ids))
        q = _sigmoid(self.booster_.predict(X_exp))
        return exp.assemble_dist(q, ids=ids, timestamps=timestamps, name=self.name)


@dataclass
class BlendedBracketNet(BaseEstimator):
    """Torch MLP bracket model trained on ``L = CE − λ·EA``.

    Same data contract as :class:`BlendedBracketGBM` — construction is
    hyperparameters only; grids/references are passed to ``fit`` /
    ``predict_dist``. Inputs are standardized with train-set statistics.
    ``ea_scale`` rescales the (small-magnitude) per-contract EA term so ``lam``
    spans a range comparable to the GBM — left ``None``, it is derived from the
    fit-set references by :func:`ea_scale_for_reference` and recorded on
    ``ea_scale_``.
    """

    lam: float = 1.0
    hidden: tuple[int, ...] = (64, 64)
    epochs: int = 500
    lr: float = 3e-3
    weight_decay: float = 1e-5
    ea_scale: float | None = None
    random_state: int = 0
    name: str = "BlendedBracketNet"

    net_: Any = field(default=None, init=False, repr=False)
    mu_: Any = field(default=None, init=False, repr=False)
    sd_: Any = field(default=None, init=False, repr=False)
    ea_scale_: float | None = field(default=None, init=False, repr=False)
    _requires_explicit_ids = True   # grids keyed by id; never auto-fill arange(N)

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(f"{self.name}: lam must be non-negative; got {self.lam}")

    def _build(self, n_feat: int):
        torch = _import_torch()

        layers: list[Any] = []
        prev = n_feat
        for h in self.hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, ids: np.ndarray,
        brackets_by_id: dict[Any, np.ndarray],
        reference_by_id: dict[Any, np.ndarray],
    ) -> Self:
        torch = _import_torch()

        _validate_value_inputs(brackets_by_id, reference_by_id, owner=self.name)
        ids = np.asarray(ids)
        exp = BracketExpander(brackets_by_id=brackets_by_id)
        X_exp, y_exp = exp.fit_transform(np.asarray(X, dtype=float),
                                         np.asarray(y, dtype=float), ids=ids)
        m_exp = _aligned_reference(exp, ids, reference_by_id)
        # Derive the EA rescaling from the fit-set references so `lam` matches the
        # GBM engine, unless the user pinned it explicitly. See
        # objective.ea_scale_for_reference for the gradient-parity argument.
        self.ea_scale_ = (
            self.ea_scale if self.ea_scale is not None
            else ea_scale_for_reference(m_exp)
        )

        torch.manual_seed(self.random_state)
        Xt = torch.tensor(X_exp, dtype=torch.float32)
        self.mu_ = Xt.mean(0)
        self.sd_ = Xt.std(0) + 1e-6
        Xn = (Xt - self.mu_) / self.sd_
        yt = torch.tensor(y_exp, dtype=torch.float32)
        mt = torch.tensor(m_exp, dtype=torch.float32)

        self.net_ = self._build(X_exp.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            opt.zero_grad()
            q = torch.sigmoid(self.net_(Xn).squeeze(1)).clamp(1e-6, 1 - 1e-6)
            ce = torch.nn.functional.binary_cross_entropy(q, yt)
            ea = ((q - mt) * (yt - mt)).mean()
            (ce - self.lam * self.ea_scale_ * ea).backward()
            opt.step()
        return self

    def predict_dist(
        self, X: np.ndarray, *, ids: np.ndarray, timestamps: np.ndarray,
        brackets_by_id: dict[Any, np.ndarray],
    ) -> DistributionForecast:
        torch = _import_torch()

        if self.net_ is None:
            raise RuntimeError(f"{self.name}.predict_dist called before fit")
        exp = BracketExpander(brackets_by_id=brackets_by_id)
        X_exp, _ = exp.transform(np.asarray(X, dtype=float), ids=np.asarray(ids))
        with torch.no_grad():
            Xn = (torch.tensor(X_exp, dtype=torch.float32) - self.mu_) / self.sd_
            q = torch.sigmoid(self.net_(Xn).squeeze(1)).numpy()
        return exp.assemble_dist(q, ids=ids, timestamps=timestamps, name=self.name)
