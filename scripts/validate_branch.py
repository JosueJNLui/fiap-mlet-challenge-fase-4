#!/usr/bin/env python3
"""Valida nomes de branch conforme a Conventional Branch 1.0.0."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

TRUNK_BRANCHES = {"main", "master", "develop"}
BRANCH_PATTERN = re.compile(
    r"^(feature|feat|bugfix|fix|hotfix|release|chore)/"
    r"[a-z0-9]+(?:\.[a-z0-9]+)*(?:-[a-z0-9]+(?:\.[a-z0-9]+)*)*$"
)


def current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_valid_branch(name: str) -> bool:
    return name in TRUNK_BRANCHES or BRANCH_PATTERN.fullmatch(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a branch name against Conventional Branch 1.0.0."
    )
    parser.add_argument(
        "branch",
        nargs="?",
        help="Branch name to validate. Defaults to the current Git branch.",
    )
    args = parser.parse_args()

    branch = args.branch or current_branch()
    if not branch:
        print("Could not determine branch name. Pass it explicitly.", file=sys.stderr)
        return 2

    if is_valid_branch(branch):
        print(f"Valid branch: {branch}")
        return 0

    print(
        f"Invalid branch: {branch}\n"
        "Expected main, master, develop, or <type>/<description> where type is one of "
        "feature, feat, bugfix, fix, hotfix, release, chore. Description must use "
        "lowercase letters, numbers, hyphens, and dots without leading, trailing, or "
        "consecutive separators.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
