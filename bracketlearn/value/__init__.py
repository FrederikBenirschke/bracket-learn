"""Reference-relative value layer, covering scoring, training, and theory.

The question is whether a price is more valuable to trade than the one already
quoted. This is a step past pure forecasting, and these tools take a reference
price ``m``.

- Trainers, :class:`BlendedBracketGBM` and :class:`BlendedBracketNet`, are
  bracket models trained on ``L = CE − λ·EA``, calibration tilted toward
  capturing the reference's mispricing, together with the shared objective
  helpers.
- Metrics are re-exported from :mod:`bracketlearn.score` into one namespace,
  namely :func:`edge_alignment`, :func:`edge_alignment_costed`,
  :func:`value_report`, and the bracket-ladder wrappers.

The guide ``docs/guides/value_vs_accuracy.md`` gives the principle.
``docs/guides/value_with_fees.md`` explains why fees make the tilt a selection
by costed value rather than by EA.
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
