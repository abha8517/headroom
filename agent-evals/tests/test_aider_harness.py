"""Unit tests for the Aider Polyglot harness + benchmark.

No API keys, no network, no real model. The aider Coder and the pytest subprocess are both
monkeypatched; a tiny fake Exercism-style exercise tree is built under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent_evals.benchmarks.aider_polyglot import AiderPolyglotGrader, load_polyglot_tasks
from agent_evals.config import AiderConfig, Settings
from agent_evals.harnesses.aider import (
    AiderHarness,
    _active_task,
    _strip_provider_prefix,
    current_task_id,
    make_savings_logger,
)
from agent_evals.metrics.savings import SavingsStore
from agent_evals.models import BenchTask, Pricing

# --------------------------------------------------------------------------------------------
# Fixtures: a tiny fake polyglot checkout
# --------------------------------------------------------------------------------------------


def _make_exercise(
    practice: Path,
    name: str,
    *,
    solution_body: str = "def go():\n    return None\n",
) -> Path:
    """Create one Exercism-style exercise under ``practice/<name>``."""

    ex = practice / name
    meta = ex / ".meta"
    docs = ex / ".docs"
    meta.mkdir(parents=True)
    docs.mkdir(parents=True)
    solution_file = f"{name.replace('-', '_')}.py"
    test_file = f"{name.replace('-', '_')}_test.py"
    config = {
        "files": {
            "solution": [solution_file],
            "test": [test_file],
            "example": [".meta/example.py"],
        }
    }
    (meta / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (docs / "instructions.md").write_text(f"# {name}\nDo the thing.\n", encoding="utf-8")
    (docs / "instructions.append.md").write_text("Extra hint.\n", encoding="utf-8")
    (ex / solution_file).write_text(solution_body, encoding="utf-8")
    (ex / test_file).write_text(
        "from " + name.replace("-", "_") + " import go\n\n\ndef test_go():\n    assert go()\n",
        encoding="utf-8",
    )
    return ex


@pytest.fixture
def polyglot_root(tmp_path: Path) -> Path:
    """A fake ``<root>/python/exercises/practice`` tree with two exercises."""

    practice = tmp_path / "python" / "exercises" / "practice"
    practice.mkdir(parents=True)
    _make_exercise(practice, "alpha-task")
    _make_exercise(practice, "beta-task")
    return tmp_path


def _cfg(root: Path, **kwargs: Any) -> AiderConfig:
    return AiderConfig(exercises_dir=root, language="python", **kwargs)


# --------------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------------


def test_loader_builds_tasks(polyglot_root: Path) -> None:
    tasks = load_polyglot_tasks(_cfg(polyglot_root))
    assert [t.task_id for t in tasks] == ["alpha-task", "beta-task"]
    alpha = tasks[0]
    assert alpha.payload["language"] == "python"
    assert alpha.payload["solution_files"] == ["alpha_task.py"]
    assert alpha.payload["test_files"] == ["alpha_task_test.py"]
    assert alpha.payload["instructions_paths"] == [
        ".docs/instructions.md",
        ".docs/instructions.append.md",
    ]
    assert Path(alpha.payload["exercise_dir"]).is_dir()
    assert Path(alpha.payload["exercise_dir"]).name == "alpha-task"


def test_loader_subset_limit(polyglot_root: Path) -> None:
    tasks = load_polyglot_tasks(_cfg(polyglot_root, subset_limit=1))
    assert [t.task_id for t in tasks] == ["alpha-task"]


def test_loader_subset_names(polyglot_root: Path) -> None:
    tasks = load_polyglot_tasks(_cfg(polyglot_root, subset_names=["beta-task"]))
    assert [t.task_id for t in tasks] == ["beta-task"]


def test_loader_subset_names_unknown_raises(polyglot_root: Path) -> None:
    with pytest.raises(ValueError, match="subset_names not found"):
        load_polyglot_tasks(_cfg(polyglot_root, subset_names=["nope"]))


def test_loader_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_polyglot_tasks(_cfg(tmp_path / "does-not-exist"))


# --------------------------------------------------------------------------------------------
# Grader
# --------------------------------------------------------------------------------------------


def _grader() -> AiderPolyglotGrader:
    return AiderPolyglotGrader(language="python")


def test_grader_benchmark_ref() -> None:
    assert _grader().benchmark_ref == "polyglot-benchmark@python"
    assert _grader().name == "aider_polyglot"


def test_grader_resolved_true() -> None:
    pred = json.dumps({"resolved": True, "tests_outcomes": [False, True], "exercise": "alpha"})
    tasks = [BenchTask(task_id="alpha-task", payload={})]
    out = _grader().grade({"alpha-task": pred}, tasks)
    res = out["alpha-task"]
    assert res.resolved is True
    assert res.detail["tests_outcomes"] == [False, True]
    assert res.detail["exercise"] == "alpha"


def test_grader_resolved_false() -> None:
    pred = json.dumps({"resolved": False, "tests_outcomes": [False], "exercise": "alpha"})
    out = _grader().grade({"alpha-task": pred}, [BenchTask(task_id="alpha-task", payload={})])
    assert out["alpha-task"].resolved is False


def test_grader_malformed_json() -> None:
    out = _grader().grade(
        {"alpha-task": "{not json"}, [BenchTask(task_id="alpha-task", payload={})]
    )
    res = out["alpha-task"]
    assert res.resolved is False
    assert "parse_error" in res.detail


def test_grader_empty_prediction() -> None:
    out = _grader().grade({}, [BenchTask(task_id="alpha-task", payload={})])
    res = out["alpha-task"]
    assert res.resolved is False
    assert res.detail["parse_error"] == "empty prediction"


# --------------------------------------------------------------------------------------------
# Savings callback
# --------------------------------------------------------------------------------------------


class _FakeResponse:
    """Minimal litellm-response stand-in exposing ``_hidden_params``."""

    def __init__(self, headers: dict[str, str]) -> None:
        self._hidden_params = {"additional_headers": headers}


def _pricing() -> Pricing:
    return Pricing(input_usd_per_1m=2.0, output_usd_per_1m=10.0)


def test_strip_provider_prefix() -> None:
    raw = {
        "llm_provider-x-headroom-tokens-before": "1000",
        "x-other": "v",
    }
    out = _strip_provider_prefix(raw)
    assert out["x-headroom-tokens-before"] == "1000"
    assert out["x-other"] == "v"


def test_savings_logger_records_for_active_task() -> None:
    store = SavingsStore()
    logger = make_savings_logger(store, lambda: "t1", _pricing())
    resp = _FakeResponse(
        {
            "llm_provider-x-headroom-tokens-before": "1000",
            "llm_provider-x-headroom-tokens-after": "400",
        }
    )
    logger.log_success_event({}, resp, None, None)
    agg = store.aggregate("t1", _pricing())
    assert agg is not None
    assert agg.tokens_before == 1000
    assert agg.tokens_after == 400
    assert agg.tokens_saved == 600


def test_savings_logger_no_task_records_nothing() -> None:
    store = SavingsStore()
    logger = make_savings_logger(store, lambda: None, _pricing())
    resp = _FakeResponse(
        {
            "llm_provider-x-headroom-tokens-before": "1000",
            "llm_provider-x-headroom-tokens-after": "400",
        }
    )
    logger.log_success_event({}, resp, None, None)
    assert store.task_ids() == []


def test_savings_logger_no_headroom_headers_records_nothing() -> None:
    store = SavingsStore()
    logger = make_savings_logger(store, lambda: "t1", _pricing())
    logger.log_success_event({}, _FakeResponse({"x-unrelated": "1"}), None, None)
    assert store.task_ids() == []


def test_active_task_contextvar_resets() -> None:
    assert current_task_id.get() is None
    with _active_task("abc"):
        assert current_task_id.get() == "abc"
    assert current_task_id.get() is None


# --------------------------------------------------------------------------------------------
# run_task happy path + retry (Coder + pytest monkeypatched)
# --------------------------------------------------------------------------------------------


class _FakeCoder:
    """Fake aider Coder: on run(), writes a passing solution into the (copied) solution file."""

    def __init__(self, solution_path: Path) -> None:
        self._solution_path = solution_path
        self.runs = 0

    def run(self, with_message: str | None = None) -> None:
        self.runs += 1
        self._solution_path.write_text("def go():\n    return True\n", encoding="utf-8")


def _settings(root: Path, **aider_kwargs: Any) -> Settings:
    return Settings(aider=_cfg(root, **aider_kwargs))


def _patch_callback_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real litellm callback registration (litellm import) in unit tests."""

    monkeypatch.setattr(AiderHarness, "_callback_registered", True, raising=False)


@pytest.mark.asyncio
async def test_run_task_happy_path(
    polyglot_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_callback_off(monkeypatch)
    settings = _settings(polyglot_root, subset_names=["alpha-task"])
    store = SavingsStore()
    harness = AiderHarness(settings, store=store)

    tasks = load_polyglot_tasks(settings.aider)
    task = tasks[0]
    source_solution = Path(task.payload["exercise_dir"]) / "alpha_task.py"
    source_before = source_solution.read_text(encoding="utf-8")

    workdir = tmp_path / "wd"

    def _fake_build(self: AiderHarness, solution_paths: list[Path]) -> _FakeCoder:
        return _FakeCoder(solution_paths[0])

    def _fake_pytest(test_paths: list[Path], wd: Path, timeout_s: float) -> tuple[bool, str]:
        return True, "1 passed"

    monkeypatch.setattr(AiderHarness, "_build_coder", _fake_build, raising=True)
    monkeypatch.setattr(AiderHarness, "_run_pytest", staticmethod(_fake_pytest), raising=True)

    result = await harness.run_task(task, env={}, workdir=workdir, task_tag="alpha-task#0")

    assert result.error is None
    parsed = json.loads(result.prediction)
    assert parsed["resolved"] is True
    assert parsed["tests_outcomes"] == [True]
    assert parsed["exercise"] == "alpha-task"
    assert result.trajectory_path == workdir
    # The exercise was copied into the workdir; the source checkout is untouched.
    assert (workdir / "alpha-task" / "alpha_task.py").exists()
    assert source_solution.read_text(encoding="utf-8") == source_before


@pytest.mark.asyncio
async def test_run_task_retry_then_pass(
    polyglot_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_callback_off(monkeypatch)
    settings = _settings(polyglot_root, subset_names=["alpha-task"], tries=2)
    harness = AiderHarness(settings, store=SavingsStore())
    task = load_polyglot_tasks(settings.aider)[0]

    monkeypatch.setattr(
        AiderHarness,
        "_build_coder",
        lambda self, paths: _FakeCoder(paths[0]),
        raising=True,
    )

    outcomes = iter([(False, "1 failed"), (True, "1 passed")])

    def _fake_pytest(test_paths: list[Path], wd: Path, timeout_s: float) -> tuple[bool, str]:
        return next(outcomes)

    monkeypatch.setattr(AiderHarness, "_run_pytest", staticmethod(_fake_pytest), raising=True)

    result = await harness.run_task(task, env={}, workdir=tmp_path / "wd", task_tag="alpha-task#0")
    parsed = json.loads(result.prediction)
    assert parsed["resolved"] is True
    assert parsed["tests_outcomes"] == [False, True]


@pytest.mark.asyncio
async def test_run_task_all_fail(
    polyglot_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_callback_off(monkeypatch)
    settings = _settings(polyglot_root, subset_names=["alpha-task"], tries=2)
    harness = AiderHarness(settings, store=SavingsStore())
    task = load_polyglot_tasks(settings.aider)[0]

    monkeypatch.setattr(
        AiderHarness,
        "_build_coder",
        lambda self, paths: _FakeCoder(paths[0]),
        raising=True,
    )
    monkeypatch.setattr(
        AiderHarness,
        "_run_pytest",
        staticmethod(lambda test_paths, wd, timeout_s: (False, "boom")),
        raising=True,
    )

    result = await harness.run_task(task, env={}, workdir=tmp_path / "wd", task_tag="alpha-task#0")
    parsed = json.loads(result.prediction)
    assert parsed["resolved"] is False
    assert parsed["tests_outcomes"] == [False, False]
    assert result.error is None


@pytest.mark.asyncio
async def test_run_task_translates_base_url(
    polyglot_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_callback_off(monkeypatch)
    settings = _settings(polyglot_root, subset_names=["alpha-task"])
    harness = AiderHarness(settings, store=SavingsStore())
    task = load_polyglot_tasks(settings.aider)[0]

    monkeypatch.setattr(
        AiderHarness,
        "_build_coder",
        lambda self, paths: _FakeCoder(paths[0]),
        raising=True,
    )
    monkeypatch.setattr(
        AiderHarness,
        "_run_pytest",
        staticmethod(lambda test_paths, wd, timeout_s: (True, "ok")),
        raising=True,
    )
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)

    await harness.run_task(
        task,
        env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:18800"},
        workdir=tmp_path / "wd",
        task_tag="alpha-task#0",
    )
    assert os.environ.get("ANTHROPIC_API_BASE") == "http://127.0.0.1:18800"


@pytest.mark.asyncio
async def test_run_task_exception_is_loud(
    polyglot_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_callback_off(monkeypatch)
    settings = _settings(polyglot_root, subset_names=["alpha-task"])
    harness = AiderHarness(settings, store=SavingsStore())
    task = load_polyglot_tasks(settings.aider)[0]

    def _boom(self: AiderHarness, paths: list[Path]) -> Any:
        raise RuntimeError("coder build failed")

    monkeypatch.setattr(AiderHarness, "_build_coder", _boom, raising=True)

    result = await harness.run_task(task, env={}, workdir=tmp_path / "wd", task_tag="alpha-task#0")
    assert result.prediction == ""
    assert result.error is not None
    assert "coder build failed" in result.error


def test_harness_metadata(polyglot_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_callback_off(monkeypatch)
    harness = AiderHarness(_settings(polyglot_root), store=SavingsStore())
    assert harness.name == "aider"
    assert harness.version  # aider's installed version string
    from agent_evals.models import Provider

    assert harness.supported_providers == {Provider.ANTHROPIC, Provider.OPENAI}


# --------------------------------------------------------------------------------------------
# Optional live test (skipped without a real key) — minimal smoke against one exercise.
# --------------------------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires ANTHROPIC_API_KEY for a real model rollout",
)
@pytest.mark.asyncio
async def test_run_task_live(tmp_path: Path) -> None:  # pragma: no cover - opt-in only
    settings = Settings(
        aider=AiderConfig(
            exercises_dir=Path(".cache/polyglot-benchmark"),
            language="python",
            subset_limit=1,
            tries=1,
        )
    )
    harness = AiderHarness(settings, store=SavingsStore())
    tasks = load_polyglot_tasks(settings.aider)
    assert tasks, "no exercises found in the live checkout"
    result = await harness.run_task(
        tasks[0], env={}, workdir=tmp_path / "live", task_tag=tasks[0].task_id
    )
    assert result.error is None
    parsed = json.loads(result.prediction)
    assert "resolved" in parsed
