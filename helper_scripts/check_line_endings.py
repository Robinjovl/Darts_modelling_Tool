#!/usr/bin/env python3
"""
Check staged Python/C/C++/CUDA files for line-ending policy violations.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LF_ONLY_SUFFIXES = (
    ".py",
    ".pyi",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".tpp",
    ".cu",
    ".cuh",
)


def tracked_policy_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--", *[f"*{suffix}" for suffix in LF_ONLY_SUFFIXES]],
        text=True,
    )
    return [line for line in output.splitlines() if line]


def staged_blob(path: str) -> bytes | None:
    """
    Return staged file bytes, or None when the path is not in the index.
    """
    git_path = Path(path).as_posix()
    try:
        return subprocess.check_output(
            ["git", "show", f":{git_path}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None


def file_bytes(path: str) -> bytes | None:
    data = staged_blob(path)
    if data is not None:
        return data

    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def has_lone_cr(data: bytes) -> bool:
    for index, byte in enumerate(data):
        if byte != 13:
            continue
        if index + 1 >= len(data) or data[index + 1] != 10:
            return True
    return False


def check_lf_only(path: str, data: bytes) -> str | None:
    crlf_count = data.count(b"\r\n")
    if crlf_count or has_lone_cr(data):
        return (
            f"{path}: must be stored with LF line endings "
            f"(found {crlf_count} CRLF sequence(s))"
        )
    return None


def main(argv: list[str]) -> int:
    failures: list[str] = []
    checked = 0
    paths = argv or tracked_policy_files()

    for path in paths:
        if Path(path).suffix.lower() not in LF_ONLY_SUFFIXES:
            continue

        data = file_bytes(path)
        if data is None:
            continue

        checked += 1
        failure = check_lf_only(path, data)
        if failure:
            failures.append(failure)

    if failures:
        print("Line-ending policy violations:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            "Expected LF-only repository content for Python/C/C++/CUDA files.",
            file=sys.stderr,
        )
        return 1

    print(f"Line-ending policy check passed for {checked} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
