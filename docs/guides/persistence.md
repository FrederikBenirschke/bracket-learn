# Persistence

bracketlearn objects pickle natively. The `persistence` module wraps
`pickle` with a small envelope so each artefact carries the bracketlearn
version that produced it.

```python
from bracketlearn.persistence import save, load

# Save a fitted pipeline with an optional note.
save(pipeline, "ridge_emos_qreg.pkl", note="prod-2026-05")

# Load it back. Warns (loudly) if the bracketlearn version differs.
loaded = load("ridge_emos_qreg.pkl")
new_dists = loaded.predict(X_new, ids=..., timestamps=...)
```

## Why an envelope and not raw pickle

A raw `pickle.dump(pipeline)` works — but on a future upgrade, a trainer's
internal representation might change. The artefact would load silently
and start producing wrong predictions. The envelope stores the
bracketlearn version alongside the payload so `load()` can warn (or, in
production, raise) on mismatch:

```python
load("old.pkl", strict_version=True)   # raises ValueError on mismatch
```

## Inspecting without loading the payload

```python
from bracketlearn.persistence import envelope_info

info = envelope_info("ridge.pkl")
# {'format_version': 1,
#  'bracketlearn_version': '0.2.0',
#  'note': 'prod-2026-05',
#  'payload_type': 'WalkForward'}
```

## Security note

`pickle` executes arbitrary code on load. Only load artefacts you produced
yourself or that came from a trusted source.
