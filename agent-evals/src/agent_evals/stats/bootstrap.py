"""Paired hierarchical bootstrap for the per-task accuracy delta (pp).

The design has two nested levels of sampling variability: which tasks we drew, and how each task's
finite K runs landed. A flat per-task-rate bootstrap would ignore the second level and produce a CI
that is too tight. So each resample is hierarchical: resample tasks with replacement, then for every
drawn task resample THAT task's runs with replacement before recomputing its rate.

All quantities are in PERCENTAGE POINTS (pp).
"""

from __future__ import annotations

import numpy as np

from ..logging import get_logger
from ..models import DeltaEstimate
from .pairing import PairedOutcomes

_log = get_logger("stats.bootstrap")

_METHOD = "paired_bootstrap"


def paired_bootstrap(
    paired: PairedOutcomes,
    *,
    alpha: float,
    n_resamples: int,
    seed: int,
) -> DeltaEstimate:
    """Hierarchical paired bootstrap CI for ``mean(delta_pp)``.

    point = ``mean(delta_pp)`` over the observed tasks. Each resample draws ``len(task_ids)`` task
    indices with replacement; for every drawn task it resamples that task's per-run resolved
    booleans (B and A1 independently) with replacement, recomputes each rate in pp, and takes the
    mean delta over the drawn tasks. The CI is the
    ``[100*alpha/2, 100*(1-alpha/2)]`` percentiles of the resample deltas. Fully deterministic for
    a fixed ``seed`` via ``np.random.default_rng``.

    Raises ``ValueError`` on empty input — there is no delta to estimate from zero tasks.
    """

    n_tasks = len(paired.task_ids)
    if n_tasks == 0:
        raise ValueError(
            "paired_bootstrap: no paired tasks; pair_arms produced an empty intersection"
        )

    rng = np.random.default_rng(seed)

    # Pre-pack each task's runs as int8 (0/1) arrays so resampling is pure-numpy and fast.
    b_runs = [np.asarray(r, dtype=np.int8) for r in paired.b_runs]
    a1_runs = [np.asarray(r, dtype=np.int8) for r in paired.a1_runs]

    point = float(np.mean(paired.delta_pp))

    resample_deltas = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        task_idx = rng.integers(0, n_tasks, size=n_tasks)
        delta_sum = 0.0
        for t in task_idx:
            br = b_runs[t]
            ar = a1_runs[t]
            b_draw = rng.integers(0, br.size, size=br.size)
            a_draw = rng.integers(0, ar.size, size=ar.size)
            b_pp = float(br[b_draw].mean()) * 100.0
            a1_pp = float(ar[a_draw].mean()) * 100.0
            delta_sum += b_pp - a1_pp
        resample_deltas[r] = delta_sum / n_tasks

    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(resample_deltas, lo_pct))
    ci_high = float(np.percentile(resample_deltas, hi_pct))

    _log.info(
        "paired bootstrap complete",
        extra={
            "fields": {
                "method": _METHOD,
                "n_tasks": n_tasks,
                "n_resamples": n_resamples,
                "alpha": alpha,
                "point": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        },
    )
    return DeltaEstimate(point=point, ci_low=ci_low, ci_high=ci_high, method=_METHOD)
