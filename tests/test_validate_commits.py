from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import validate_commits


def test_commit_hashes_falls_back_to_target_revision_on_invalid_range(monkeypatch) -> None:
    def fake_run(command, check=True, capture_output=True, text=True):
        if command[:2] == ["git", "rev-list"]:
            if command[-1] == "oldsha..newsha":
                raise subprocess.CalledProcessError(
                    128,
                    command,
                    stderr="fatal: ambiguous argument 'oldsha..newsha'",
                )
            if command[-1] == "newsha":
                return subprocess.CompletedProcess(command, 0, stdout="newsha\n", stderr="")
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="newsha\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(validate_commits.subprocess, "run", fake_run)

    assert validate_commits.commit_hashes("oldsha..newsha") == ["newsha"]


def test_initial_commit_messages_are_skipped(monkeypatch) -> None:
    monkeypatch.setattr(validate_commits, "commit_hashes", lambda revision_range: ["56d11568514a"])
    monkeypatch.setattr(validate_commits, "commit_message", lambda commit: "Initial commit")
    monkeypatch.setattr(validate_commits.sys, "argv", ["validate_commits.py"])

    def fake_validate_message(message):
        return subprocess.CompletedProcess(["cz"], 1, stdout="", stderr="invalid")

    monkeypatch.setattr(validate_commits, "validate_message", fake_validate_message)

    assert validate_commits.main() == 0
