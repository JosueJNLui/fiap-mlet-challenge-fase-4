#!/usr/bin/env python3
"""Valida tags do Git no formato semântico MAJOR.MINOR.PATCH."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

TAG_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def repository_tags() -> list[str]:
    result = subprocess.run(
        ["git", "tag", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [tag for tag in result.stdout.splitlines() if tag]


def is_valid_tag(tag: str) -> bool:
    return TAG_PATTERN.fullmatch(tag) is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate tags using semantic MAJOR.MINOR.PATCH format."
    )
    parser.add_argument(
        "tags",
        nargs="*",
        help="Tag names to validate. Defaults to all tags in the repository.",
    )
    args = parser.parse_args()

    tags = args.tags or repository_tags()
    invalid_tags = [tag for tag in tags if not is_valid_tag(tag)]

    if invalid_tags:
        print("Invalid semantic tags:", file=sys.stderr)
        for tag in invalid_tags:
            print(f"- {tag}", file=sys.stderr)
        print("Expected format: MAJOR.MINOR.PATCH, for example 1.2.3.", file=sys.stderr)
        return 1

    if tags:
        print(f"Valid semantic tags: {', '.join(tags)}")
    else:
        print("No tags to validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
