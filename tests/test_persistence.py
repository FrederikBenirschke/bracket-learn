"""Persistence tests: save → load round-trip + version envelope.

What's guaranteed:

- A fitted ``ForecastPipeline`` round-trips through ``save``/``load`` and
  produces identical predictions to the original on the same X.
- The envelope carries a bracketlearn version and a user-supplied note.
- ``load`` warns on version mismatch (RuntimeWarning) and raises when
  ``strict_version=True`` is set.
- Loading something that isn't a bracketlearn envelope raises with a
  clear message rather than silently returning whatever ``pickle`` saw.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from bracketlearn import persistence
from bracketlearn.composite import LiftedForecaster
from bracketlearn.lift import GlobalResidual
from bracketlearn.persistence import envelope_info, load, save
from bracketlearn.pipeline import ForecastPipeline, PipelineResult
from bracketlearn.trainers import EMOS, SklearnPoint


def _synthetic(n: int = 150, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, k))
    y = X.mean(axis=1) + rng.normal(0, 0.5, n)
    return X, y, np.arange(n), np.arange(n, dtype=float)


def _build_fitted_pipeline():
    X, y, ids, ts = _synthetic()
    p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3)
    p.fit_predict(X, y, ids=ids, timestamps=ts)
    return p, X, ids, ts


class TestRoundTrip:
    def test_pipeline_round_trip(self, tmp_path: Path):
        p, X, ids, ts = _build_fitted_pipeline()
        path = tmp_path / "p.pkl"
        save(p, path, note="unit test")
        loaded = load(path)
        # Predictions match bit-for-bit on the same data.
        pred_a = p.predict(X[:5], ids=np.arange(5),
                           timestamps=np.arange(5, dtype=float))
        pred_b = loaded.predict(X[:5], ids=np.arange(5),
                                timestamps=np.arange(5, dtype=float))
        np.testing.assert_array_equal(
            pred_a["emos"].params["mu"],
            pred_b["emos"].params["mu"],
        )

    def test_lifted_round_trip(self, tmp_path: Path):
        from sklearn.linear_model import LinearRegression

        X, y, ids, ts = _synthetic()
        p = ForecastPipeline(
            steps=[("ridge", LiftedForecaster(
                SklearnPoint(LinearRegression()), GlobalResidual(), name="ridge",
            ))],
            n_folds=3,
        )
        p.fit_predict(X, y, ids=ids, timestamps=ts)
        path = tmp_path / "ridge.pkl"
        save(p, path)
        loaded = load(path)
        pred_a = p.predict(X[:5], ids=np.arange(5),
                           timestamps=np.arange(5, dtype=float))
        pred_b = loaded.predict(X[:5], ids=np.arange(5),
                                timestamps=np.arange(5, dtype=float))
        np.testing.assert_allclose(
            pred_a["ridge"].params["mu"],
            pred_b["ridge"].params["mu"],
        )

    def test_result_round_trip(self, tmp_path: Path):
        X, y, ids, ts = _synthetic()
        p = ForecastPipeline(steps=[("emos", EMOS())], n_folds=3,
                             refit_on_full=False)
        result = p.fit_predict(X, y, ids=ids, timestamps=ts)
        path = tmp_path / "result.pkl"
        save(result, path)
        loaded = load(path)
        assert isinstance(loaded, PipelineResult)
        np.testing.assert_array_equal(
            result["emos"].params["mu"],
            loaded["emos"].params["mu"],
        )


class TestEnvelope:
    def test_envelope_carries_note(self, tmp_path: Path):
        p, _, _, _ = _build_fitted_pipeline()
        path = tmp_path / "p.pkl"
        save(p, path, note="prod-2026-05")
        info = envelope_info(path)
        assert info["note"] == "prod-2026-05"
        assert info["payload_type"] == "ForecastPipeline"
        assert info["bracketlearn_version"] is not None

    def test_envelope_info_does_not_require_payload_class(self, tmp_path: Path):
        """envelope_info should return metadata fields the user can read
        without needing payload-construction succeeds. (We don't fully verify
        lazy unpickling here — pickle reads the whole file — but we pin the
        fields the helper returns.)"""
        p, _, _, _ = _build_fitted_pipeline()
        path = tmp_path / "p.pkl"
        save(p, path)
        info = envelope_info(path)
        assert set(info.keys()) == {
            "format_version", "bracketlearn_version", "note", "payload_type",
        }


class TestVersionGate:
    def test_warns_on_version_mismatch(self, tmp_path: Path, monkeypatch):
        p, _, _, _ = _build_fitted_pipeline()
        path = tmp_path / "p.pkl"
        save(p, path)
        # Simulate an upgrade by patching the module-level version constant.
        monkeypatch.setattr(persistence, "__BL_VERSION__", "999.0.0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load(path)
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        assert any("version mismatch" in m for m in msgs)

    def test_strict_version_raises(self, tmp_path: Path, monkeypatch):
        p, _, _, _ = _build_fitted_pipeline()
        path = tmp_path / "p.pkl"
        save(p, path)
        monkeypatch.setattr(persistence, "__BL_VERSION__", "999.0.0")
        with pytest.raises(ValueError, match="version mismatch"):
            load(path, strict_version=True)

    def test_rejects_non_envelope(self, tmp_path: Path):
        path = tmp_path / "raw.pkl"
        with open(path, "wb") as f:
            pickle.dump({"random": "dict"}, f)
        with pytest.raises(ValueError, match="not a bracketlearn artefact"):
            load(path)
