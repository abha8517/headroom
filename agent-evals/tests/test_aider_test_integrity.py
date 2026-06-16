"""Regression guard: the harness restores pristine test files before grading.

A model (via aider) can emit a diff that rewrites the exercise's test file with bogus
expectations — observed live on a retry. The grader must always run the REAL tests, so the
harness copies the pristine test file(s) from the source checkout over the working copy before
each pytest invocation. This test pins that behaviour without needing aider or a model.
"""

from __future__ import annotations

from pathlib import Path

from agent_evals.harnesses.aider import AiderHarness


def test_restore_tests_overwrites_tampered_copy(tmp_path: Path) -> None:
    src = tmp_path / "src" / "leap"
    dest = tmp_path / "dest" / "leap"
    (src).mkdir(parents=True)
    (dest).mkdir(parents=True)

    pristine = "def test_real():\n    assert leap(2000) is True\n"
    (src / "leap_test.py").write_text(pristine, encoding="utf-8")
    # The working copy has been tampered with (model rewrote the test to pass trivially).
    (dest / "leap_test.py").write_text("def test_fake():\n    assert True\n", encoding="utf-8")

    AiderHarness._restore_tests(src, dest, ["leap_test.py"])

    assert (dest / "leap_test.py").read_text(encoding="utf-8") == pristine


def test_restore_tests_handles_nested_and_missing(tmp_path: Path) -> None:
    src = tmp_path / "src" / "ex"
    dest = tmp_path / "dest" / "ex"
    (src / "tests").mkdir(parents=True)
    (dest).mkdir(parents=True)
    (src / "tests" / "t.py").write_text("X", encoding="utf-8")

    # Nested path is created on restore; a listed-but-absent source path is skipped (not an error).
    AiderHarness._restore_tests(src, dest, ["tests/t.py", "does_not_exist.py"])

    assert (dest / "tests" / "t.py").read_text(encoding="utf-8") == "X"
    assert not (dest / "does_not_exist.py").exists()
