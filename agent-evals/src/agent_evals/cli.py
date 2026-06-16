"""agent-evals CLI.

``version`` / ``show-config`` are config introspection. ``run`` executes a full WITH-vs-WITHOUT
experiment (3 arms) over the Aider Polyglot subset and prints the scorecard with the TOST verdict.
Provider keys are loaded from a ``.env`` (never read from chat/source) for the live model calls.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import click

from . import __version__
from .config import Settings
from .logging import configure_logging, get_logger
from .models import Provider
from .report.scorecard import render_scorecard

logger = get_logger("cli")

# headroom repo root = .../headroom ; this file = .../headroom/agent-evals/src/agent_evals/cli.py
_AGENT_EVALS_ROOT = Path(__file__).resolve().parents[2]
_HEADROOM_ROOT = Path(__file__).resolve().parents[3]


@click.group()
def main() -> None:
    """End-to-end accuracy A/B for Headroom (coding-agent benchmarks WITH vs WITHOUT compression)."""


@main.command()
def version() -> None:
    """Print the agent-evals version."""

    click.echo(__version__)


@main.command(name="show-config")
def show_config() -> None:
    """Print the resolved settings (defaults + env) as JSON."""

    settings = Settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    click.echo(settings.model_dump_json(indent=2))


def _load_env(settings: Settings) -> str:
    """Load provider keys from a .env into os.environ for live runs. Returns the path used."""

    from dotenv import find_dotenv, load_dotenv

    if settings.dotenv_path is not None:
        path = str(settings.dotenv_path)
    else:
        path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)
    return path or "(none found)"


@main.command(name="run")
@click.option(
    "--provider", type=click.Choice([p.value for p in Provider]), help="Override provider."
)
@click.option("--model", "model_snapshot", help="Override model snapshot (e.g. claude-sonnet-4-6).")
@click.option("--language", help="Polyglot language subset (python/go/rust/java).")
@click.option("--subset-limit", type=int, help="Run only the first N exercises (sorted).")
@click.option(
    "--subset-names", help="Comma-separated exact exercise names (overrides --subset-limit)."
)
@click.option("--k-runs", type=int, help="Independent runs per (arm, task).")
@click.option("--concurrency", type=int, help="Concurrent rollouts per arm (use 1 for aider).")
@click.option("--exercises-dir", type=click.Path(), help="Path to the polyglot-benchmark checkout.")
@click.option("--run-dir", type=click.Path(), help="Where to write journal/manifest/scorecard.")
def run_cmd(
    provider: str | None,
    model_snapshot: str | None,
    language: str | None,
    subset_limit: int | None,
    subset_names: str | None,
    k_runs: int | None,
    concurrency: int | None,
    exercises_dir: str | None,
    run_dir: str | None,
) -> None:
    """Run the 3-arm WITH-vs-WITHOUT experiment and print the scorecard."""

    settings = Settings()
    if provider is not None:
        settings.provider = Provider(provider)
    if model_snapshot is not None:
        settings.model_snapshot = model_snapshot
    if language is not None:
        settings.aider.language = language
    if subset_limit is not None:
        settings.aider.subset_limit = subset_limit
    if subset_names:
        settings.aider.subset_names = [n.strip() for n in subset_names.split(",") if n.strip()]
    if k_runs is not None:
        settings.stats.k_runs = k_runs
    if concurrency is not None:
        settings.concurrency = concurrency
    if exercises_dir is not None:
        settings.aider.exercises_dir = Path(exercises_dir)

    configure_logging(settings.log_level, json_output=settings.log_json)
    env_path = _load_env(settings)
    logger.info("env_loaded", extra={"fields": {"dotenv": env_path}})

    now = datetime.now(timezone.utc)
    out_dir = Path(run_dir) if run_dir else (settings.run_dir / now.strftime("%Y%m%dT%H%M%SZ"))

    # Imported here so `version`/`show-config` never pull in aider/litellm.
    from .run import run_experiment

    _manifest, scorecard = asyncio.run(
        run_experiment(
            settings,
            now=now,
            run_dir=out_dir,
            headroom_repo_path=str(_HEADROOM_ROOT),
            agent_evals_repo_path=str(_AGENT_EVALS_ROOT),
        )
    )
    click.echo(render_scorecard(scorecard))
    click.echo(f"\nArtifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
