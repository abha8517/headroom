"""Phase-1 scorecard: the paired-bootstrap + TOST equivalence verdict path."""

from __future__ import annotations

from agent_evals.config import StatsConfig
from agent_evals.models import ArmName, TaskResult
from agent_evals.report.scorecard import build_scorecard, render_scorecard


def _cells(arm: ArmName, pattern: dict[str, list[bool]]) -> list[TaskResult]:
    """Build cells for one arm: pattern maps task_id -> per-run resolved booleans."""

    out: list[TaskResult] = []
    for task_id, runs in pattern.items():
        for i, resolved in enumerate(runs):
            out.append(TaskResult(task_id=task_id, arm=arm, run_index=i, resolved=resolved))
    return out


def _stats() -> StatsConfig:
    # Small, fast, deterministic.
    return StatsConfig(k_runs=4, alpha=0.05, margin_lossy_pp=2.0, bootstrap_resamples=2000, seed=7)


def test_equivalent_when_arms_identical() -> None:
    # Constant-per-task outcomes (all-pass or all-fail) => zero within-task run variance, so an
    # identical B vs A1 yields every per-task delta == 0 => bootstrap CI [0,0] => EQUIVALENT.
    # (A *mixed* per-task pattern would instead leave a wide CI at this small n => inconclusive,
    # which is the real noise floor this framework is built to surface.)
    pattern = {
        "t0": [True, True, True, True],
        "t1": [True, True, True, True],
        "t2": [False, False, False, False],
        "t3": [True, True, True, True],
        "t4": [False, False, False, False],
    }
    results = _cells(ArmName.A1_PASSTHROUGH, pattern) + _cells(ArmName.B_HEADROOM, pattern)
    sc = build_scorecard(results, "exp-equiv", stats_config=_stats())
    assert sc.equivalence is not None
    assert sc.equivalence.verdict == "equivalent"
    assert sc.accuracy_delta_pp is not None and abs(sc.accuracy_delta_pp) < 1e-6
    assert sc.noise_floor_pp is not None and sc.noise_floor_pp > 0
    assert "EQUIVALENT" in render_scorecard(sc)


def test_inferior_when_b_much_worse() -> None:
    # B resolves far less often than A1 -> delta strongly negative -> INFERIOR.
    a1 = {f"t{i}": [True, True, True, True] for i in range(6)}
    b = {f"t{i}": [False, False, False, False] for i in range(6)}
    results = _cells(ArmName.A1_PASSTHROUGH, a1) + _cells(ArmName.B_HEADROOM, b)
    sc = build_scorecard(results, "exp-inf", stats_config=_stats())
    assert sc.equivalence is not None
    assert sc.equivalence.verdict == "inferior"
    assert sc.accuracy_delta_pp is not None and sc.accuracy_delta_pp < -50.0


def test_verdict_absent_without_stats_config() -> None:
    pattern = {f"t{i}": [True, False] for i in range(3)}
    results = _cells(ArmName.A1_PASSTHROUGH, pattern) + _cells(ArmName.B_HEADROOM, pattern)
    sc = build_scorecard(results, "exp-nostats")  # Phase-0 behaviour
    assert sc.equivalence is None
    assert "Phase 1" in render_scorecard(sc)


def test_verdict_absent_when_b_arm_missing() -> None:
    results = _cells(ArmName.A1_PASSTHROUGH, {"t0": [True, True]})
    sc = build_scorecard(results, "exp-missing", stats_config=_stats())
    assert sc.equivalence is None
    assert "unavailable" in sc.stats_note
