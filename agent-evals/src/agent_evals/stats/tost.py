"""Verdicts from a delta CI: two one-sided tests (TOST) equivalence and non-inferiority.

We never read a p-value here — the bootstrap already gave us a ``(1 - alpha)`` CI, and the CI-based
TOST verdict is equivalent to the two one-sided tests at level ``alpha``. Working in PERCENTAGE
POINTS throughout: a negative ``delta_pp`` means B resolved fewer tasks than A1 (B is worse).
"""

from __future__ import annotations

from typing import Literal

from ..logging import get_logger
from ..models import DeltaEstimate, EquivalenceVerdict

Verdict = Literal["equivalent", "inferior", "inconclusive", "superior"]

_log = get_logger("stats.tost")


def equivalence_verdict(delta: DeltaEstimate, margin_pp: float) -> EquivalenceVerdict:
    """TOST equivalence verdict against a symmetric margin ``[-margin_pp, +margin_pp]``.

    - ``equivalent``  : the whole CI ``[ci_low, ci_high]`` lies inside ``[-margin, +margin]``.
    - ``inferior``    : ``ci_high < -margin`` (B is conclusively worse than A1 by > margin).
    - ``superior``    : ``ci_low > +margin`` (B is conclusively better than A1 by > margin).
    - ``inconclusive``: anything else (CI straddles a margin boundary).
    """

    lo = delta.ci_low
    hi = delta.ci_high
    verdict: Verdict
    if lo >= -margin_pp and hi <= margin_pp:
        verdict = "equivalent"
    elif hi < -margin_pp:
        verdict = "inferior"
    elif lo > margin_pp:
        verdict = "superior"
    else:
        verdict = "inconclusive"

    _log.info(
        "equivalence verdict",
        extra={
            "fields": {
                "test": "tost_equivalence",
                "margin_pp": margin_pp,
                "ci_low": lo,
                "ci_high": hi,
                "verdict": verdict,
            }
        },
    )
    return EquivalenceVerdict(delta=delta, margin=margin_pp, verdict=verdict)


def non_inferiority_verdict(delta: DeltaEstimate, margin_pp: float) -> EquivalenceVerdict:
    """One-sided non-inferiority verdict: is B no worse than A1 by more than ``margin_pp``?

    Returns ``equivalent`` (read here as "non-inferior": the CI rules out a true delta below
    ``-margin_pp``) when ``ci_low > -margin_pp``, otherwise ``inferior``. The ``EquivalenceVerdict``
    type is reused deliberately; only ``equivalent``/``inferior`` are produced by this one-sided
    test.
    """

    lo = delta.ci_low
    verdict: Verdict = "equivalent" if lo > -margin_pp else "inferior"

    _log.info(
        "non-inferiority verdict",
        extra={
            "fields": {
                "test": "non_inferiority",
                "margin_pp": margin_pp,
                "ci_low": lo,
                "verdict": verdict,
            }
        },
    )
    return EquivalenceVerdict(delta=delta, margin=margin_pp, verdict=verdict)
