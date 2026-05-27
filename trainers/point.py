"""Point forecasters (output: PointForecast).

SklearnPoint, OnlineAggregator, RNNHourly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from bracketlearn.base import BaseEstimator
from bracketlearn.forecast import (
    PointForecast,
    ProvenanceMeta,
)
from bracketlearn.trainers._common import (
    _estimator_accepts_sample_weight,
)

# ---------------------------------------------------------------------------
# SklearnPoint — wrap any sklearn-style regressor as a PointForecaster.
# ---------------------------------------------------------------------------


@dataclass
class SklearnPoint(BaseEstimator):
    """Adapter: any object with sklearn's fit(X, y) + predict(X) is a
    PointForecaster.

    Works with sklearn.linear_model.{Ridge, Lasso, LinearRegression, ...},
    LightGBM/XGBoost regressors, sklearn ensembles, custom estimators —
    anything matching the sklearn contract.

    Examples:
        SklearnPoint(sklearn.linear_model.Ridge(alpha=1.0))
        SklearnPoint(sklearn.ensemble.GradientBoostingRegressor())
        SklearnPoint(lightgbm.LGBMRegressor(n_estimators=200))
    """

    estimator: Any
    name: str | None = None
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = type(self.estimator).__name__

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        # Record input signature BEFORE np.asarray strips the columns
        # attribute (sklearn convention: feature_names_in_ from DataFrame).
        self._record_input_signature(X)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Forward sample_weight only if the estimator accepts it. We
        # introspect the signature (no silent TypeError swallow).
        if sample_weight is not None and _estimator_accepts_sample_weight(self.estimator):
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y)
        self.fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> PointForecast:
        mu = np.asarray(self.estimator.predict(np.asarray(X, dtype=float)), dtype=float)
        prov = ProvenanceMeta.placeholder(self.name)
        return PointForecast(
            mu=mu,
            ids=np.asarray(ids),
            timestamps=np.asarray(timestamps),
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# OnlineAggregator — sleeping-experts AdaHedge (PointForecaster).
# ---------------------------------------------------------------------------


@dataclass
class OnlineAggregator(BaseEstimator):
    """AdaHedge over forecast experts (columns of X).

    Walks rows in order, treats each column of X as an expert's point
    prediction (NaN = asleep on that row), accumulates per-expert squared
    losses, updates the mixability-gap learning rate, and produces an
    aggregated prediction per row.

    Predict-time behavior mirrors the original's `predict_inference_side`
    path: at fit time the final weight vector is snapshotted; at predict
    time we compute weighted mean over awake experts, renormalising the
    snapshot weights to the active subset. This is what the original ships
    to inference — pure online behavior during fit, snapshot-and-apply at
    predict.

    Output: PointForecaster — pair with GlobalResidual (or other Lifter)
    for distribution coverage. Composition is explicit, not baked in.
    """

    min_experts: int = 2
    name: str = "OnlineAggregator"
    depends_on: tuple[str, ...] = ()
    final_w_: np.ndarray | None = field(default=None, init=False)
    K_: int | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
    ) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"OnlineAggregator expects 2-D X (rows × experts); got {X.shape}")
        T, K = X.shape
        L = np.zeros(K)
        delta = 0.0
        eta = float("inf")
        log_K = float(np.log(max(K, 2)))
        last_w_per_expert = np.zeros(K)
        seen_per_expert = np.zeros(K, dtype=int)

        for t in range(T):
            f_t = X[t]
            y_t = y[t]
            awake = ~np.isnan(f_t)
            n_awake = int(awake.sum())
            if n_awake < self.min_experts:
                continue
            awake_idx = np.where(awake)[0]
            L_awake = L[awake_idx]
            w_awake = self._softmin(eta, L_awake)
            last_w_per_expert[awake_idx] = w_awake
            seen_per_expert[awake_idx] += 1
            f_awake = f_t[awake_idx]
            ell_awake = (f_awake - y_t) ** 2
            hedge_loss_t = float(np.dot(w_awake, ell_awake))
            mix_loss_t = self._mix_loss(eta, w_awake, ell_awake)
            delta += max(0.0, hedge_loss_t - mix_loss_t)
            if delta > 0:
                eta = log_K / delta
            L[awake_idx] += ell_awake

        if seen_per_expert.sum() == 0:
            raise RuntimeError(
                f"OnlineAggregator: no rows had ≥{self.min_experts} awake experts"
            )
        # Final weights: per AdaHedge semantics, take the *current* posterior
        # over all experts (those never awake get 0). Renormalise.
        w_final = self._softmin(eta, L)
        # Zero out experts never seen — guards against giving cold-start
        # vendors any weight at predict time.
        w_final[seen_per_expert == 0] = 0.0
        s = w_final.sum()
        if s <= 0:
            raise RuntimeError("OnlineAggregator: final weight vector sums to 0")
        self.final_w_ = w_final / s
        self.K_ = K
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> PointForecast:
        if self.final_w_ is None:
            raise RuntimeError("OnlineAggregator.predict called before fit")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.K_:
            raise ValueError(
                f"OnlineAggregator: predict X has K={X.shape[1]}, train had K={self.K_}"
            )
        N = X.shape[0]
        awake = ~np.isnan(X)                      # (N, K) bool
        # Weight matrix: final_w_ broadcast against awake mask.
        w_mat = self.final_w_[None, :] * awake    # (N, K) — zeroes on asleep
        x_mat = np.where(awake, X, 0.0)
        num = (w_mat * x_mat).sum(axis=1)         # (N,)
        denom = w_mat.sum(axis=1)                 # (N,)
        awake_counts = awake.sum(axis=1)          # (N,)
        ok = (awake_counts >= self.min_experts) & (denom > 0)
        mu = np.full(N, np.nan)
        mu[ok] = num[ok] / denom[ok]
        # Leftover NaNs are a real coverage hole — raise.
        if np.isnan(mu).any():
            n_miss = int(np.isnan(mu).sum())
            raise RuntimeError(
                f"OnlineAggregator.predict: {n_miss}/{N} rows had < {self.min_experts} awake experts"
            )
        prov = ProvenanceMeta.placeholder(self.name)
        return PointForecast(
            mu=mu, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )

    @staticmethod
    def _softmin(eta: float, losses: np.ndarray) -> np.ndarray:
        if not np.isfinite(eta):
            w = np.ones_like(losses)
            return w / w.sum()
        scaled = -eta * losses
        scaled = scaled - scaled.max()
        w = np.exp(scaled)
        return w / w.sum()

    @staticmethod
    def _mix_loss(eta: float, weights: np.ndarray, losses: np.ndarray) -> float:
        if not np.isfinite(eta):
            return float(losses.min())
        z = -eta * losses
        z_max = z.max()
        return float(-(np.log(np.sum(weights * np.exp(z - z_max))) + z_max) / eta)


# ---------------------------------------------------------------------------
# RNNHourly — GRU on (24, C) hourly tensor (PointForecaster).
# ---------------------------------------------------------------------------


@dataclass
class RNNHourly(BaseEstimator):
    """Tiny GRU on a (24, C) hourly tensor → residual-corrected point forecast.

    GRU reads the 24-hour sequence, concatenates a station embedding (if station_ids
    is passed via the `station_ids` argument at fit), MLP head outputs a
    scalar residual to the channel-0 max (HRRR's max-T baseline). Final
    prediction = channel_0_max + residual.

    Expects X.ndim == 3 with shape (N, T, C). For weather: T=24 hours,
    C=6 (temperature, dewpoint, RH, wind, cloud, CAPE).

    `baseline_channel`: which channel's max provides the residual anchor
    (default 0 = temperature, matching the original trainer).

    `station_ids` (optional, passed at fit/predict via `meta=...` arg):
    integer-encoded station for the embedding. If absent, embedding is
    skipped and the model uses GRU only.

    Output: PointForecaster — pair with GlobalResidual (or other Lifter)
    for distribution coverage.
    """

    hidden: int = 32
    embed: int = 4
    dropout: float = 0.3
    epochs: int = 200
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-4
    baseline_channel: int = 0
    seed: int = 17
    name: str = "RNNHourly"
    depends_on: tuple[str, ...] = ()
    model_: Any = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    std_: np.ndarray | None = field(default=None, init=False)
    n_stations_: int | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        deps_oof: dict[str, Any] | None = None,
        station_ids: np.ndarray | None = None,
    ) -> Self:
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"RNNHourly expects 3-D X (N, T, C); got {X.shape}")
        N, T, C = X.shape
        # Residual target = realized - baseline_channel_max.
        baseline = X[:, :, self.baseline_channel].max(axis=1)
        residual = y - baseline

        # Per-channel normaliser fit on train only.
        flat = X.reshape(-1, C)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.mean_, self.std_ = mean.astype(np.float32), std

        if station_ids is not None:
            sid = np.asarray(station_ids, dtype=np.int64)
            if sid.shape[0] != N:
                raise ValueError(f"station_ids length {sid.shape[0]} != N={N}")
            n_stations = int(sid.max()) + 1
        else:
            sid = np.zeros(N, dtype=np.int64)
            n_stations = 1
        self.n_stations_ = n_stations

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model_ = _HourlyGRU(
            n_channels=C, n_stations=n_stations,
            hidden=self.hidden, embed=self.embed, dropout=self.dropout,
        )
        opt = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.SmoothL1Loss(beta=1.0)

        Xn = (X - self.mean_) / self.std_
        Xt = torch.from_numpy(Xn.astype(np.float32))
        yt = torch.from_numpy(residual.astype(np.float32))
        st = torch.from_numpy(sid)

        for _ in range(self.epochs):
            perm = torch.randperm(N)
            self.model_.train()
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self.model_(Xt[idx], st[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        ids: np.ndarray,
        timestamps: np.ndarray,
        station_ids: np.ndarray | None = None,
    ) -> PointForecast:
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch

        if self.model_ is None:
            raise RuntimeError("RNNHourly.predict called before fit")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"RNNHourly.predict expects 3-D X; got {X.shape}")
        N = X.shape[0]
        baseline = X[:, :, self.baseline_channel].max(axis=1)
        Xn = (X - self.mean_) / self.std_
        if station_ids is not None:
            sid = np.asarray(station_ids, dtype=np.int64)
            # Raise on unknown station IDs instead of silently
            # clamping them onto station 0's embedding. Cold-start is a
            # real failure mode that needs caller-level handling (drop the
            # row, pick a fallback embedding policy explicitly, or extend
            # the training set), not a silent map-to-zero.
            unknown_mask = (sid < 0) | (sid >= self.n_stations_)
            if np.any(unknown_mask):
                bad = np.unique(sid[unknown_mask]).tolist()
                raise ValueError(
                    f"RNNHourly.predict: {int(unknown_mask.sum())} rows have "
                    f"station_ids outside the trained range "
                    f"[0, {self.n_stations_ - 1}]; unknown IDs={bad[:10]}"
                )
        else:
            sid = np.zeros(N, dtype=np.int64)
        self.model_.eval()
        with torch.no_grad():
            pred_resid = self.model_(
                torch.from_numpy(Xn.astype(np.float32)),
                torch.from_numpy(sid),
            ).numpy()
        mu = (baseline + pred_resid).astype(float)
        prov = ProvenanceMeta.placeholder(self.name, random_seed=self.seed)
        return PointForecast(
            mu=mu, ids=np.asarray(ids), timestamps=np.asarray(timestamps),
            provenance=prov,
        )


class _HourlyGRU:
    """Inner torch module (built lazily via __new__ trick to avoid eager
    torch import at module import time). Mirrors weather/rnn_hourly.HourlyGRU.
    """

    def __new__(cls, n_channels: int, n_stations: int, hidden: int, embed: int, dropout: float):
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch
        from torch import nn

        class HourlyGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.station_embed = nn.Embedding(n_stations, embed)
                self.gru = nn.GRU(input_size=n_channels, hidden_size=hidden, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Sequential(
                    nn.Linear(hidden + embed, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x, sid_idx):
                _, h_n = self.gru(x)
                h = h_n[-1]
                emb = self.station_embed(sid_idx)
                z = self.dropout(torch.cat([h, emb], dim=-1))
                return self.head(z).squeeze(-1)

        return HourlyGRU()


