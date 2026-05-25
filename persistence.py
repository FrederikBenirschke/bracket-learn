"""Save and load fitted ``ForecastPipeline`` / ``PipelineResult`` instances.

bracketlearn objects pickle natively — every trainer's fitted state is
plain numpy arrays plus picklable third-party models (sklearn,
LightGBM, NGBoost, torch). This module wraps ``pickle`` with a small
envelope so that:

- The bracketlearn version that produced the artefact is stored alongside
  it (``__bracketlearn_version__``). ``load()`` warns loudly when loading
  an artefact built with a different version — drift between
  algorithm-internal representations is the most common cause of
  silently-wrong predictions after a library upgrade.
- The artefact is self-describing: a single ``.pkl`` carries the
  pipeline, the metadata version, and an optional user note.
- The on-disk format is a plain pickle, not a custom serialisation. If
  you ever need to inspect it without bracketlearn installed, raw
  ``pickle.load(open(path,'rb'))['payload']`` works.

Usage::

    from bracketlearn.persistence import save, load

    save(pipeline, "ridge_emos_qreg.pkl", note="prod-2026-05")
    loaded = load("ridge_emos_qreg.pkl")
    new_dists = loaded.predict(X_new, ids=..., timestamps=...)

Both ``ForecastPipeline`` and ``PipelineResult`` are accepted — anything
picklable, really. The envelope is identical.

Security note: ``pickle`` executes arbitrary code on load. Only load
artefacts you produced yourself or that came from a trusted source.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __BL_VERSION__ = version("bracketlearn")
    except PackageNotFoundError:
        __BL_VERSION__ = "dev"
except ImportError:  # pragma: no cover - py<3.8 not supported
    __BL_VERSION__ = "dev"


_FORMAT_VERSION = 1


def save(obj: Any, path: str | Path, *, note: str | None = None) -> None:
    """Pickle ``obj`` to ``path`` with a bracketlearn-version envelope.

    Args:
        obj: any picklable object — typically a fitted ``ForecastPipeline``
            or a ``PipelineResult``.
        path: filesystem path. Parent directories are not created.
        note: optional free-text note stored in the envelope (e.g. the
            git SHA, training-data snapshot id, prod cohort name).
    """
    envelope = {
        "format_version": _FORMAT_VERSION,
        "bracketlearn_version": __BL_VERSION__,
        "note": note,
        "payload": obj,
    }
    with open(path, "wb") as f:
        pickle.dump(envelope, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str | Path, *, strict_version: bool = False) -> Any:
    """Load a ``save()``-produced pickle. Returns the payload.

    Args:
        path: filesystem path written by ``save()``.
        strict_version: if True, raise on version mismatch instead of
            warning. Use when running in production where silently
            accepting a stale artefact would be unsafe.

    Raises:
        ValueError: if the file is not a bracketlearn envelope, or if
            ``strict_version`` is set and versions don't match.
    """
    with open(path, "rb") as f:
        envelope = pickle.load(f)
    if not isinstance(envelope, dict) or "format_version" not in envelope:
        raise ValueError(
            f"{path}: not a bracketlearn artefact (no format_version field). "
            "Use plain `pickle.load` if you saved without `bracketlearn.persistence.save`."
        )
    fmt = envelope["format_version"]
    if fmt != _FORMAT_VERSION:
        raise ValueError(
            f"{path}: envelope format_version={fmt} "
            f"(expected {_FORMAT_VERSION}); the file was written by a "
            "future bracketlearn release"
        )
    saved_version = envelope.get("bracketlearn_version", "<unknown>")
    if saved_version != __BL_VERSION__:
        msg = (
            f"{path}: bracketlearn version mismatch — "
            f"artefact was saved with {saved_version}, "
            f"current install is {__BL_VERSION__}. "
            "Predictions may differ from the original training run "
            "if trainer internals changed."
        )
        if strict_version:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return envelope["payload"]


def envelope_info(path: str | Path) -> dict[str, Any]:
    """Return the envelope metadata without instantiating the payload.

    Cheaper than ``load`` because we stop after reading the wrapper.
    Useful for "what's in this file" tooling.
    """
    with open(path, "rb") as f:
        envelope = pickle.load(f)
    if not isinstance(envelope, dict) or "format_version" not in envelope:
        raise ValueError(f"{path}: not a bracketlearn artefact")
    return {
        "format_version": envelope["format_version"],
        "bracketlearn_version": envelope.get("bracketlearn_version"),
        "note": envelope.get("note"),
        "payload_type": type(envelope.get("payload")).__name__,
    }
