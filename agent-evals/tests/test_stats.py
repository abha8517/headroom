"""Deterministic, I/O-free tests for the agent-evals statistics layer."""

from __future__ import annotations

import pytest

from agent_evals.models import ArmName, DeltaEstimate, TaskResult
from agent_evals.stats.bootstrap import paired_bootstrap
from agent_evals.stats.pairing import pair_arms
from agent_evals.stats.power import is_underpowered, recommended_runs
from agent_evals.stats.tost import equivalence_verdict, non_inferiority_verdict
from agent_evals.stats.wilson import wilson_half_width_pp, wilson_interval


def _cells(task_id: str, arm: ArmName, resolved_pattern: list[bool]) -> list[TaskResult]:
    """Build K TaskResults for one (task, arm) from a list of per-run resolved booleans."""

    return [
        TaskResult(task_id=task_id, arm=arm, run_index=i, resolved=r)
        for i, r in enumerate(resolved_pattern)
    ]


# --------------------------------------------------------------------------- pairing


def test_pair_arms_rates_and_deltas() -> None:
    b = ArmName.B_HEADROOM
    a1 = ArmName.A1_PASSTHROUGH
    results: list[TaskResult] = []
    # t1: B 3/4 = 75pp, A1 2/4 = 50pp -> delta +25
    results += _cells("t1", b, [True, True, True, False])
    results += _cells("t1", a1, [True, True, False, False])
    # t2: B 0/4 = 0pp, A1 4/4 = 100pp -> delta -100
    results += _cells("t2", b, [False, False, False, False])
    results += _cells("t2", a1, [True, True, True, True])

    paired = pair_arms(results, b, a1)

    assert paired.task_ids == ["t1", "t2"]  # sorted, intersection
    assert paired.b_rate_pp == [75.0, 0.0]
    assert paired.a1_rate_pp == [50.0, 100.0]
    assert paired.delta_pp == [25.0, -100.0]
    assert paired.b_runs == [[True, True, True, False], [False, False, False, False]]
    assert paired.a1_runs == [[True, True, False, False], [True, True, True, True]]


def test_pair_arms_intersection_only_drops_unmatched() -> None:
    b = ArmName.B_HEADROOM
    a1 = ArmName.A1_PASSTHROUGH
    results: list[TaskResult] = []
    results += _cells("shared", b, [True, False])
    results += _cells("shared", a1, [True, True])
    results += _cells("b_only", b, [True, True])  # missing from a1 -> dropped
    results += _cells("a1_only", a1, [False, False])  # missing from b -> dropped

    paired = pair_arms(results, b, a1)

    assert paired.task_ids == ["shared"]
    assert paired.b_rate_pp == [50.0]
    assert paired.a1_rate_pp == [100.0]
    assert paired.delta_pp == [-50.0]


def test_pair_arms_errored_cell_counts_as_failure() -> None:
    # resolved=False (which an errored cell carries) is an observed failure, not dropped.
    b = ArmName.B_HEADROOM
    a1 = ArmName.A1_PASSTHROUGH
    results: list[TaskResult] = []
    results += [
        TaskResult(task_id="t", arm=b, run_index=0, resolved=True),
        TaskResult(task_id="t", arm=b, run_index=1, resolved=False, error="boom"),
    ]
    results += _cells("t", a1, [True, True])

    paired = pair_arms(results, b, a1)

    assert paired.task_ids == ["t"]
    assert paired.b_rate_pp == [50.0]  # the errored run counted as a failure
    assert paired.a1_rate_pp == [100.0]


# --------------------------------------------------------------------------- bootstrap


def _paired_from(b_runs: list[list[bool]], a1_runs: list[list[bool]]) -> list[TaskResult]:
    b = ArmName.B_HEADROOM
    a1 = ArmName.A1_PASSTHROUGH
    results: list[TaskResult] = []
    for i, runs in enumerate(b_runs):
        results += _cells(f"t{i}", b, runs)
    for i, runs in enumerate(a1_runs):
        results += _cells(f"t{i}", a1, runs)
    return results


def test_bootstrap_is_deterministic_for_same_seed() -> None:
    b_runs = [[True, True, False], [True, False, False], [True, True, True]]
    a1_runs = [[True, False, False], [False, False, False], [True, True, False]]
    paired = pair_arms(_paired_from(b_runs, a1_runs), ArmName.B_HEADROOM, ArmName.A1_PASSTHROUGH)

    d1 = paired_bootstrap(paired, alpha=0.05, n_resamples=2000, seed=777)
    d2 = paired_bootstrap(paired, alpha=0.05, n_resamples=2000, seed=777)

    assert d1 == d2
    assert d1.method == "paired_bootstrap"


def test_bootstrap_different_seed_differs() -> None:
    # Enough tasks/runs that the percentile grid is fine-grained, so two independent RNG
    # streams land on distinct CIs (a real property of the bootstrap at adequate resolution).
    b_runs = [
        [True, True, False, True, False, True],
        [True, False, False, True, True, False],
        [False, True, True, False, True, True],
        [True, True, True, False, False, False],
        [False, False, True, True, False, True],
    ]
    a1_runs = [
        [True, False, False, True, False, False],
        [False, True, False, False, True, False],
        [True, False, True, False, False, True],
        [False, False, True, True, False, False],
        [True, True, False, False, True, False],
    ]
    paired = pair_arms(_paired_from(b_runs, a1_runs), ArmName.B_HEADROOM, ArmName.A1_PASSTHROUGH)
    d1 = paired_bootstrap(paired, alpha=0.05, n_resamples=4000, seed=1)
    d2 = paired_bootstrap(paired, alpha=0.05, n_resamples=4000, seed=2)
    # Point is data-only (seed-independent); the CIs should not coincide exactly.
    assert d1.point == pytest.approx(d2.point)
    assert (d1.ci_low, d1.ci_high) != (d2.ci_low, d2.ci_high)


def test_bootstrap_point_equals_mean_delta() -> None:
    # B 100pp, A1 0pp on both tasks -> every per-task delta is +100, mean = 100.
    b_runs = [[True, True], [True, True]]
    a1_runs = [[False, False], [False, False]]
    paired = pair_arms(_paired_from(b_runs, a1_runs), ArmName.B_HEADROOM, ArmName.A1_PASSTHROUGH)
    d = paired_bootstrap(paired, alpha=0.05, n_resamples=1000, seed=5)
    assert d.point == pytest.approx(100.0)


def test_bootstrap_strong_positive_delta_ci_excludes_zero() -> None:
    # B fully resolves, A1 fully fails, across several tasks -> CI well above 0.
    b_runs = [[True, True, True]] * 5
    a1_runs = [[False, False, False]] * 5
    paired = pair_arms(_paired_from(b_runs, a1_runs), ArmName.B_HEADROOM, ArmName.A1_PASSTHROUGH)
    d = paired_bootstrap(paired, alpha=0.05, n_resamples=3000, seed=11)
    assert d.point == pytest.approx(100.0)
    assert d.ci_low > 0.0
    assert d.ci_high == pytest.approx(100.0)


def test_bootstrap_identical_arms_point_zero_ci_straddles() -> None:
    # Same per-task patterns in both arms with within-task variability so the CI has width.
    patterns = [
        [True, False, True, False],
        [True, True, False, False],
        [False, True, True, False],
    ]
    paired = pair_arms(_paired_from(patterns, patterns), ArmName.B_HEADROOM, ArmName.A1_PASSTHROUGH)
    d = paired_bootstrap(paired, alpha=0.05, n_resamples=4000, seed=23)
    assert d.point == pytest.approx(0.0)
    assert d.ci_low < 0.0 < d.ci_high


def test_bootstrap_empty_raises() -> None:
    paired = pair_arms([], ArmName.B_HEADROOM, ArmName.A1_PASSTHROUGH)
    with pytest.raises(ValueError, match="no paired tasks"):
        paired_bootstrap(paired, alpha=0.05, n_resamples=100, seed=0)


def test_bootstrap_alpha_widens_ci() -> None:
    b_runs = [[True, True, False], [True, False, False], [True, True, True]]
    a1_runs = [[True, False, False], [False, True, False], [True, True, False]]
    paired = pair_arms(_paired_from(b_runs, a1_runs), ArmName.B_HEADROOM, ArmName.A1_PASSTHROUGH)
    narrow = paired_bootstrap(paired, alpha=0.20, n_resamples=4000, seed=99)
    wide = paired_bootstrap(paired, alpha=0.01, n_resamples=4000, seed=99)
    assert (wide.ci_high - wide.ci_low) >= (narrow.ci_high - narrow.ci_low)


# --------------------------------------------------------------------------- tost


def _delta(low: float, high: float, point: float | None = None) -> DeltaEstimate:
    pt = point if point is not None else (low + high) / 2.0
    return DeltaEstimate(point=pt, ci_low=low, ci_high=high, method="test")


def test_equivalence_all_four_verdicts() -> None:
    margin = 2.0
    assert equivalence_verdict(_delta(-1.0, 1.0), margin).verdict == "equivalent"
    assert equivalence_verdict(_delta(-5.0, -3.0), margin).verdict == "inferior"
    assert equivalence_verdict(_delta(3.0, 5.0), margin).verdict == "superior"
    assert equivalence_verdict(_delta(-3.0, 1.0), margin).verdict == "inconclusive"
    # CI straddling the upper boundary is also inconclusive.
    assert equivalence_verdict(_delta(1.0, 3.0), margin).verdict == "inconclusive"


def test_equivalence_boundary_inclusive() -> None:
    margin = 2.0
    # CI exactly on the margins is still equivalent (subset uses <=/>=).
    assert equivalence_verdict(_delta(-2.0, 2.0), margin).verdict == "equivalent"


def test_equivalence_verdict_preserves_inputs() -> None:
    margin = 2.0
    d = _delta(-1.0, 1.0)
    v = equivalence_verdict(d, margin)
    assert v.delta == d
    assert v.margin == margin


def test_non_inferiority_verdicts() -> None:
    margin = 2.0
    # ci_low above -margin -> non-inferior ("equivalent")
    assert non_inferiority_verdict(_delta(-1.0, 5.0), margin).verdict == "equivalent"
    # ci_low below -margin -> inferior (regardless of how high ci_high is)
    assert non_inferiority_verdict(_delta(-3.0, 10.0), margin).verdict == "inferior"
    # exactly at -margin is NOT strictly greater -> inferior
    assert non_inferiority_verdict(_delta(-2.0, 5.0), margin).verdict == "inferior"


# --------------------------------------------------------------------------- wilson


def test_wilson_half_width_known_value() -> None:
    hw = wilson_half_width_pp(0.5, 100, confidence=0.95)
    assert hw == pytest.approx(9.6, abs=0.2)


def test_wilson_interval_within_unit() -> None:
    low, high = wilson_interval(50, 100, confidence=0.95)
    assert 0.0 <= low < high <= 1.0
    # Centered near 0.5.
    assert low == pytest.approx(0.404, abs=0.01)
    assert high == pytest.approx(0.596, abs=0.01)


def test_wilson_interval_extremes_clamped() -> None:
    low0, high0 = wilson_interval(0, 20, confidence=0.95)
    assert low0 == 0.0
    assert high0 > 0.0
    lown, highn = wilson_interval(20, 20, confidence=0.95)
    assert highn == 1.0
    assert lown < 1.0


def test_wilson_width_shrinks_with_n() -> None:
    w_small = wilson_half_width_pp(0.5, 50, confidence=0.95)
    w_large = wilson_half_width_pp(0.5, 500, confidence=0.95)
    assert w_large < w_small


def test_wilson_higher_confidence_widens() -> None:
    w90 = wilson_half_width_pp(0.5, 100, confidence=0.90)
    w99 = wilson_half_width_pp(0.5, 100, confidence=0.99)
    assert w99 > w90


def test_wilson_input_validation() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_half_width_pp(1.5, 10)


# --------------------------------------------------------------------------- power


def test_recommended_runs_decreases_with_more_tasks() -> None:
    k_few = recommended_runs(sigma_pp=20.0, margin_pp=5.0, n_tasks=50)
    k_many = recommended_runs(sigma_pp=20.0, margin_pp=5.0, n_tasks=200)
    assert k_many <= k_few
    assert k_many >= 1


def test_recommended_runs_decreases_with_larger_margin() -> None:
    k_tight = recommended_runs(sigma_pp=20.0, margin_pp=2.0, n_tasks=100)
    k_loose = recommended_runs(sigma_pp=20.0, margin_pp=8.0, n_tasks=100)
    assert k_loose <= k_tight


def test_recommended_runs_increases_with_sigma() -> None:
    k_low = recommended_runs(sigma_pp=10.0, margin_pp=5.0, n_tasks=100)
    k_high = recommended_runs(sigma_pp=40.0, margin_pp=5.0, n_tasks=100)
    assert k_high >= k_low


def test_recommended_runs_floor_is_one() -> None:
    # Huge task set + loose margin + tiny noise => analytic k < 1, floored at 1.
    k = recommended_runs(sigma_pp=1.0, margin_pp=50.0, n_tasks=1000)
    assert k == 1


def test_recommended_runs_validation() -> None:
    with pytest.raises(ValueError):
        recommended_runs(sigma_pp=0.0, margin_pp=5.0, n_tasks=10)
    with pytest.raises(ValueError):
        recommended_runs(sigma_pp=10.0, margin_pp=0.0, n_tasks=10)
    with pytest.raises(ValueError):
        recommended_runs(sigma_pp=10.0, margin_pp=5.0, n_tasks=0)


def test_is_underpowered_consistent_with_recommended() -> None:
    needed = recommended_runs(sigma_pp=30.0, margin_pp=4.0, n_tasks=80)
    assert is_underpowered(k_runs=needed - 1, sigma_pp=30.0, margin_pp=4.0, n_tasks=80)
    assert not is_underpowered(k_runs=needed, sigma_pp=30.0, margin_pp=4.0, n_tasks=80)
    assert not is_underpowered(k_runs=needed + 5, sigma_pp=30.0, margin_pp=4.0, n_tasks=80)
