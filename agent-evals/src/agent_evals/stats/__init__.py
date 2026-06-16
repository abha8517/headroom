"""agent_evals.stats subpackage."""

from __future__ import annotations

from .bootstrap import paired_bootstrap
from .pairing import PairedOutcomes, pair_arms
from .power import is_underpowered, recommended_runs
from .tost import equivalence_verdict, non_inferiority_verdict
from .wilson import wilson_half_width_pp, wilson_interval

__all__ = [
    "PairedOutcomes",
    "pair_arms",
    "paired_bootstrap",
    "equivalence_verdict",
    "non_inferiority_verdict",
    "wilson_interval",
    "wilson_half_width_pp",
    "recommended_runs",
    "is_underpowered",
]
