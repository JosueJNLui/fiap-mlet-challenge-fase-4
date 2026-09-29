#!/usr/bin/env python3
"""Valida mensagens de commit do Git com Commitizen (conventional commits)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def commit_hashes(revision_range: str | None) -> list[str]:
    command = ["git", "rev-list", "--reverse"]
    command.append(revision_range or "HEAD")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if not revision_range:
            raise

        fallback_revision = revision_range.split("..")[-1]
        fallback_command = ["git", "rev-list", "--reverse", fallback_revision]
        try:
            fallback_result = subprocess.run(
                fallback_command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            raise exc from None

        return [commit for commit in fallback_result.stdout.splitlines() if commit]

    return [commit for commit in result.stdout.splitlines() if commit]


def commit_message(commit: str) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%B", commit],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_message(message: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("cz") or "cz"
    return subprocess.run(
        [executable, "check", "--message", message],
        capture_output=True,
        text=True,
    )


def should_skip_message(message: str) -> bool:
    normalized = " ".join(message.split())
    return normalized.lower() == "initial commit"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate commit messages against Conventional Commits 1.0.0."
    )
    parser.add_argument(
        "--range",
        dest="revision_range",
        help="Git revision range to validate, for example origin/main..HEAD.",
    )
    parser.add_argument(
        "--message",
        help="Validate one explicit commit message instead of reading Git history.",
    )
    args = parser.parse_args()

    failures: list[tuple[str, str]] = []

    if args.message is not None:
        if should_skip_message(args.message):
            print("Skipping initial commit message.")
        else:
            result = validate_message(args.message)
            if result.returncode != 0:
                failures.append(("provided message", result.stderr or result.stdout))
    else:
        for commit in commit_hashes(args.revision_range):
            message = commit_message(commit)
            if should_skip_message(message):
                continue

            result = validate_message(message)
            if result.returncode != 0:
                subject = message.splitlines()[0] if message else "<empty>"
                failures.append((f"{commit[:12]} {subject}", result.stderr or result.stdout))

    if failures:
        print("Invalid conventional commit messages:", file=sys.stderr)
        for commit, error in failures:
            print(f"\n- {commit}", file=sys.stderr)
            print(error.strip(), file=sys.stderr)
        return 1

    print("All commit messages are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
