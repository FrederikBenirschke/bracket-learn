"""Reference-relative **value** layer: scoring, training, and theory for "is my
price more valuable to trade than the one already quoted?" — a step past pure
forecasting (these tools take a reference price ``m``).

- Trainers — :class:`BlendedBracketGBM`, :class:`BlendedBracketNet`: bracket
  models trained on ``L = CE − λ·EA`` (calibration tilted toward capturing the
  reference's mispricing), plus the shared objective helpers.
- Metrics (re-exported from :mod:`bracketlearn.score` for one namespace):
  :func:`edge_alignment`, :func:`edge_alignment_costed`, :func:`value_report`,
  and the bracket-ladder wrappers.

Guides: ``docs/guides/value_vs_accuracy.md`` (the principle) and
``docs/guides/value_with_fees.md`` (why fees make you select the tilt by costed
value, not EA).
"""

from __future__ import annotations

from bracketlearn.score import (
    edge_alignment,
    edge_alignment_bracket,
    edge_alignment_corr,
    edge_alignment_costed,
    edge_alignment_dist,
    shared_bias_slope,
    value_report,
    value_report_bracket,
    value_report_dist,
)
from bracketlearn.value.objective import (
    blended_grad_hess,
    blended_loss,
    ea_scale_for_reference,
    make_lgb_objective,
)
from bracketlearn.value.trainers import BlendedBracketGBM, BlendedBracketNet

__all__ = [
    "BlendedBracketGBM",
    "BlendedBracketNet",
    "make_lgb_objective",
    "ea_scale_for_reference",
    "blended_grad_hess",
    "blended_loss",
    "edge_alignment",
    "edge_alignment_costed",
    "edge_alignment_corr",
    "shared_bias_slope",
    "value_report",
    "edge_alignment_bracket",
    "value_report_bracket",
    "edge_alignment_dist",
    "value_report_dist",
]
