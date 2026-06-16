"""Aider Polyglot benchmark: task loader + grader.

The loader scans an Exercism-style checkout of ``github.com/Aider-AI/polyglot-benchmark`` and
turns each practice exercise into a :class:`~agent_evals.models.BenchTask`. The grader is a
pass-through of the pytest verdict the harness already computed during rollout: Aider + pytest IS
the official Polyglot grade, so it is produced inside
:class:`~agent_evals.harnesses.aider.AiderHarness` and merely surfaced here as a
:class:`~agent_evals.models.GradeResult`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_evals.config import AiderConfig
from agent_evals.logging import get_logger
from agent_evals.models import BenchTask, GradeResult

logger = get_logger("benchmarks.aider_polyglot")

# Exercism layout constants — bound to the polyglot-benchmark checkout structure.
_PRACTICE_SUBPATH = ("exercises", "practice")
_META_DIR = ".meta"
_CONFIG_FILE = "config.json"
_DOCS_DIR = ".docs"
_INSTRUCTIONS_FILE = "instructions.md"
_INSTRUCTIONS_APPEND_FILE = "instructions.append.md"

# Keys inside .meta/config.json's ``files`` block.
_FILES_KEY = "files"
_SOLUTION_KEY = "solution"
_TEST_KEY = "test"


def _practice_dir(cfg: AiderConfig) -> Path:
    """Resolve ``<exercises_dir>/<language>/exercises/practice`` (absolute)."""

    return (cfg.exercises_dir / cfg.language).joinpath(*_PRACTICE_SUBPATH).resolve()


def _read_config_files(exercise_dir: Path) -> dict[str, Any]:
    """Read the ``files`` block from an exercise's ``.meta/config.json``."""

    config_path = exercise_dir / _META_DIR / _CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(f"missing exercise config: {config_path}")
    data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"malformed exercise config (not an object): {config_path}")
    files = data.get(_FILES_KEY)
    if not isinstance(files, dict):
        raise ValueError(f"exercise config missing 'files' block: {config_path}")
    return files


def _str_list(value: Any) -> list[str]:
    """Coerce a config leaf into a list of strings (single string -> one-element list)."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _instruction_paths(exercise_dir: Path) -> list[str]:
    """Existing instruction docs (main + optional append), as exercise-relative paths."""

    out: list[str] = []
    for name in (_INSTRUCTIONS_FILE, _INSTRUCTIONS_APPEND_FILE):
        if (exercise_dir / _DOCS_DIR / name).exists():
            out.append(f"{_DOCS_DIR}/{name}")
    return out


def _select(exercise_dirs: list[Path], cfg: AiderConfig) -> list[Path]:
    """Apply subset selection: ``subset_names`` (exact) wins, else ``subset_limit``, else all."""

    by_name = {p.name: p for p in exercise_dirs}
    if cfg.subset_names:
        selected: list[Path] = []
        missing: list[str] = []
        for name in cfg.subset_names:
            if name in by_name:
                selected.append(by_name[name])
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"subset_names not found under {cfg.language!r}: {sorted(missing)}")
        return sorted(selected, key=lambda p: p.name)
    if cfg.subset_limit is not None:
        return exercise_dirs[: cfg.subset_limit]
    return exercise_dirs


def load_polyglot_tasks(cfg: AiderConfig) -> list[BenchTask]:
    """Scan the polyglot checkout and build one :class:`BenchTask` per practice exercise.

    Tasks are sorted by exercise name for determinism, then subset-selected. Each payload carries
    absolute exercise dir + config-relative solution/test/instruction paths so the harness can
    resolve them inside its isolated copy. Raises if the exercises directory is missing.
    """

    practice = _practice_dir(cfg)
    if not practice.is_dir():
        raise FileNotFoundError(
            f"polyglot exercises directory not found: {practice} "
            f"(check AiderConfig.exercises_dir={cfg.exercises_dir!r} and language={cfg.language!r})"
        )

    exercise_dirs = sorted((p for p in practice.iterdir() if p.is_dir()), key=lambda p: p.name)
    selected = _select(exercise_dirs, cfg)

    tasks: list[BenchTask] = []
    for exercise_dir in selected:
        files = _read_config_files(exercise_dir)
        solution_files = _str_list(files.get(_SOLUTION_KEY))
        test_files = _str_list(files.get(_TEST_KEY))
        if not solution_files:
            raise ValueError(f"exercise {exercise_dir.name!r} has no solution files in config")
        if not test_files:
            raise ValueError(f"exercise {exercise_dir.name!r} has no test files in config")
        tasks.append(
            BenchTask(
                task_id=exercise_dir.name,
                payload={
                    "exercise_dir": str(exercise_dir.resolve()),
                    "language": cfg.language,
                    "solution_files": solution_files,
                    "test_files": test_files,
                    "instructions_paths": _instruction_paths(exercise_dir),
                },
            )
        )

    logger.info(
        "loaded polyglot tasks",
        extra={
            "fields": {
                "language": cfg.language,
                "practice_dir": str(practice),
                "total_available": len(exercise_dirs),
                "selected": len(tasks),
            }
        },
    )
    return tasks


class AiderPolyglotGrader:
    """Pass-through grader for Aider Polyglot.

    The harness runs aider + pytest during rollout and writes the verdict into the prediction
    JSON (``{"resolved": bool, "tests_outcomes": [...], "exercise": str}``). This grader parses
    that JSON back into a :class:`GradeResult`. A missing / malformed / empty prediction grades as
    ``resolved=False`` with the parse failure recorded in ``detail`` — loud, never silent.
    """

    def __init__(self, language: str) -> None:
        self.name = "aider_polyglot"
        self.benchmark_ref = f"polyglot-benchmark@{language}"

    def grade(self, predictions: dict[str, str], tasks: list[BenchTask]) -> dict[str, GradeResult]:
        """Grade each task by parsing the harness-computed verdict in its prediction JSON."""

        results: dict[str, GradeResult] = {}
        for task in tasks:
            prediction = predictions.get(task.task_id, "")
            results[task.task_id] = self._grade_one(task.task_id, prediction)
        return results

    def _grade_one(self, task_id: str, prediction: str) -> GradeResult:
        if not prediction:
            logger.warning(
                "empty prediction; grading as unresolved",
                extra={"fields": {"task_id": task_id}},
            )
            return GradeResult(
                task_id=task_id,
                resolved=False,
                detail={"parse_error": "empty prediction"},
            )
        try:
            parsed: Any = json.loads(prediction)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "malformed prediction JSON; grading as unresolved",
                extra={"fields": {"task_id": task_id, "error": repr(exc)}},
            )
            return GradeResult(
                task_id=task_id,
                resolved=False,
                detail={"parse_error": repr(exc), "raw": prediction[:200]},
            )
        if not isinstance(parsed, dict):
            return GradeResult(
                task_id=task_id,
                resolved=False,
                detail={"parse_error": "prediction is not a JSON object"},
            )
        resolved = bool(parsed.get("resolved", False))
        return GradeResult(
            task_id=task_id,
            resolved=resolved,
            detail={
                "tests_outcomes": parsed.get("tests_outcomes", []),
                "exercise": parsed.get("exercise"),
            },
        )
