"""Assemble and execute a full WITH-vs-WITHOUT experiment, then score it.

This is the Phase-1 entry the CLI ``run`` command drives. It builds the three arms (Direct /
Passthrough / Headroom), the Aider harness + Polyglot grader, runs the resumable orchestrator over
the selected exercises, and produces a scorecard with the paired-bootstrap + TOST verdict. All
time/identity is injected by the caller (the CLI) so the core stays reproducible.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .arms import HeadroomArm
from .benchmarks.aider_polyglot import AiderPolyglotGrader, load_polyglot_tasks
from .config import Settings
from .harnesses.aider import AiderHarness
from .logging import get_logger
from .manifest import build_manifest
from .metrics.savings import SavingsStore
from .models import ArmName, ArmSpec, ProxyMode, RunManifest
from .orchestrator import Journal, Orchestrator
from .protocols import Arm
from .report.scorecard import Scorecard, build_scorecard, render_scorecard

logger = get_logger("run")

# The canonical three-arm plan. Headline accuracy claim = B_HEADROOM vs A1_PASSTHROUGH.
_ARM_PLAN: list[tuple[ArmName, ProxyMode | None, str]] = [
    (ArmName.A0_DIRECT, None, "direct"),
    (ArmName.A1_PASSTHROUGH, ProxyMode.OFF, "passthrough"),
    (ArmName.B_HEADROOM, ProxyMode.TOKEN, "headroom"),
]


def build_arms(settings: Settings, run_dir: Path) -> list[Arm]:
    """Build the three HeadroomArms for the configured provider."""

    arms: list[Arm] = []
    for name, mode, label in _ARM_PLAN:
        spec = ArmSpec(name=name, provider=settings.provider, proxy_mode=mode, label=label)
        arms.append(HeadroomArm(spec, settings, run_dir))
    return arms


async def run_experiment(
    settings: Settings,
    *,
    now: datetime,
    run_dir: Path,
    headroom_repo_path: str,
    agent_evals_repo_path: str,
) -> tuple[RunManifest, Scorecard]:
    """Run the full experiment and return the pinned manifest + scorecard (also persisted)."""

    tasks = load_polyglot_tasks(settings.aider)
    if not tasks:
        raise RuntimeError(
            f"no polyglot tasks loaded from {settings.aider.exercises_dir} "
            f"(language={settings.aider.language!r})"
        )

    store = SavingsStore()
    harness = AiderHarness(settings, store=store)
    grader = AiderPolyglotGrader(settings.aider.language)
    arms = build_arms(settings, run_dir)
    journal = Journal(run_dir)
    orchestrator = Orchestrator(settings, arms, harness, grader, journal)

    logger.info(
        "experiment_start",
        extra={
            "fields": {
                "n_tasks": len(tasks),
                "arms": [a.spec.name.value for a in arms],
                "k_runs": settings.stats.k_runs,
                "provider": settings.provider.value,
                "model": settings.litellm_model_name(),
            }
        },
    )

    results = await orchestrator.run(tasks)

    manifest = build_manifest(
        settings,
        now=now,
        arms=[a.spec for a in arms],
        benchmark="aider_polyglot",
        benchmark_ref=grader.benchmark_ref,
        harness=harness.name,
        harness_version=harness.version,
        headroom_repo_path=headroom_repo_path,
        agent_evals_repo_path=agent_evals_repo_path,
    )
    scorecard = build_scorecard(results, manifest.experiment_id, stats_config=settings.stats)
    _persist(run_dir, manifest, scorecard)
    logger.info(
        "experiment_done",
        extra={
            "fields": {
                "experiment_id": manifest.experiment_id,
                "verdict": (scorecard.equivalence.verdict if scorecard.equivalence else None),
                "delta_pp": scorecard.accuracy_delta_pp,
            }
        },
    )
    return manifest, scorecard


def _persist(run_dir: Path, manifest: RunManifest, scorecard: Scorecard) -> None:
    """Write the manifest + scorecard (JSON + rendered text) into the run directory."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "scorecard.json").write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "scorecard.txt").write_text(render_scorecard(scorecard), encoding="utf-8")
