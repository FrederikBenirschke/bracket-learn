# Persistence

bracketlearn objects pickle natively. The `persistence` module wraps
`pickle` with a small envelope so each artefact carries the bracketlearn
version that produced it.

```python
from bracketlearn.persistence import save, load

# Save a fitted pipeline with an optional note.
save(pipeline, "ridge_emos_qreg.pkl", note="prod-2026-05")

# Load it back. Warns if the bracketlearn version differs.
loaded = load("ridge_emos_qreg.pkl")
new_dists = loaded.predict(X_new, ids=..., timestamps=...)
```

## The version envelope

A raw `pickle.dump(pipeline)` works today. On a future upgrade, though, a
trainer's internal representation can change, and the artefact loads without
complaint and starts producing wrong predictions. The envelope stores the
bracketlearn version alongside the payload, so `load()` warns (or, in
production, raises) on a mismatch:

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
