"""Pair two arms task-by-task for paired statistical comparison.

The headline comparison is paired: each task contributes one resolved-rate to arm B and one to
arm A1, and we analyze the per-task delta (B - A1). Pairing on the same tasks removes
between-task difficulty as a noise source, which the unpaired comparison cannot.

All rates are in PERCENTAGE POINTS (pp): a task's resolved rate is ``mean(resolved bools) * 100``.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from ..logging import get_logger
from ..models import ArmName, TaskResult

_log = get_logger("stats.pairing")


class PairedOutcomes(BaseModel):
    """Per-task paired outcomes for arms (B vs A1), aligned by ``task_ids`` index.

    ``b_rate_pp[i]``/``a1_rate_pp[i]`` are the resolved rates (pp) for ``task_ids[i]`` in each
    arm; ``delta_pp[i] == b_rate_pp[i] - a1_rate_pp[i]``. ``b_runs[i]``/``a1_runs[i]`` hold that
    task's raw per-run resolved booleans, kept so the bootstrap can resample run-to-run variance.
    """

    arm_b: ArmName
    arm_a1: ArmName
    task_ids: list[str] = Field(default_factory=list)
    b_rate_pp: list[float] = Field(default_factory=list)
    a1_rate_pp: list[float] = Field(default_factory=list)
    delta_pp: list[float] = Field(default_factory=list)
    b_runs: list[list[bool]] = Field(default_factory=list)
    a1_runs: list[list[bool]] = Field(default_factory=list)


def _group_runs(results: list[TaskResult], arm: ArmName) -> dict[str, list[bool]]:
    """Collect the per-run ``resolved`` booleans for one arm, keyed by task_id.

    An errored cell already carries ``resolved=False`` in the journal, so it is included here as
    an observed failure (a real outcome), not silently dropped.
    """

    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        if r.arm is arm:
            by_task[r.task_id].append(bool(r.resolved))
    return by_task


def _rate_pp(runs: list[bool]) -> float:
    """Resolved rate in pp = mean of resolved booleans * 100."""

    return sum(1 for x in runs if x) / len(runs) * 100.0


def pair_arms(results: list[TaskResult], arm_b: ArmName, arm_a1: ArmName) -> PairedOutcomes:
    """Pair ``arm_b`` against ``arm_a1`` on the intersection of their task_ids.

    Groups results by (arm, task_id); a task's rate is ``mean(resolved over its runs) * 100``.
    Only tasks present in BOTH arms are paired; tasks missing from either arm are logged and
    dropped (never fabricated). Output ``task_ids`` is sorted for determinism.
    """

    b_by_task = _group_runs(results, arm_b)
    a1_by_task = _group_runs(results, arm_a1)

    b_keys = set(b_by_task)
    a1_keys = set(a1_by_task)
    shared = sorted(b_keys & a1_keys)
    dropped = b_keys ^ a1_keys

    if dropped:
        _log.warning(
            "dropping tasks missing from one arm",
            extra={
                "fields": {
                    "arm_b": arm_b.value,
                    "arm_a1": arm_a1.value,
                    "dropped_count": len(dropped),
                    "dropped_task_ids": sorted(dropped),
                    "b_only": sorted(b_keys - a1_keys),
                    "a1_only": sorted(a1_keys - b_keys),
                }
            },
        )

    paired = PairedOutcomes(arm_b=arm_b, arm_a1=arm_a1)
    for task_id in shared:
        b_runs = b_by_task[task_id]
        a1_runs = a1_by_task[task_id]
        b_pp = _rate_pp(b_runs)
        a1_pp = _rate_pp(a1_runs)
        paired.task_ids.append(task_id)
        paired.b_rate_pp.append(b_pp)
        paired.a1_rate_pp.append(a1_pp)
        paired.delta_pp.append(b_pp - a1_pp)
        paired.b_runs.append(b_runs)
        paired.a1_runs.append(a1_runs)

    _log.info(
        "paired arms",
        extra={
            "fields": {
                "arm_b": arm_b.value,
                "arm_a1": arm_a1.value,
                "paired_tasks": len(paired.task_ids),
            }
        },
    )
    return paired
