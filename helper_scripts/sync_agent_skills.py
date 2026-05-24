#!/usr/bin/env python3
"""Mirror project skills from .agents/skills to .claude/skills."""

import argparse
import hashlib
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

IGNORED_NAMES = {".DS_Store", "Thumbs.db"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
IGNORED_PARTS = {"__pycache__"}


def should_ignore(rel_path: Path) -> bool:
    if any(part in IGNORED_PARTS for part in rel_path.parts):
        return True
    if rel_path.name in IGNORED_NAMES:
        return True
    if rel_path.suffix in IGNORED_SUFFIXES:
        return True
    return False


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if should_ignore(rel):
            continue
        yield rel


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_trees(src: Path, dst: Path) -> tuple[list[str], list[str], list[str]]:
    src_files = {str(rel): file_sha256(src / rel) for rel in iter_files(src)}
    dst_files = {str(rel): file_sha256(dst / rel) for rel in iter_files(dst)}

    missing = sorted(set(src_files) - set(dst_files))
    extra = sorted(set(dst_files) - set(src_files))
    changed = sorted(
        rel
        for rel in set(src_files).intersection(dst_files)
        if src_files[rel] != dst_files[rel]
    )
    return missing, extra, changed


def sync(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror skills from .agents/skills to .claude/skills."
    )
    parser.add_argument(
        "--src", default=".agents/skills", help="Source skill directory"
    )
    parser.add_argument(
        "--dst", default=".claude/skills", help="Destination skill directory"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that destination matches source; do not modify files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.is_dir():
        print(f"Source directory does not exist: {src}", file=sys.stderr)
        return 1

    if args.check:
        if not dst.is_dir():
            print(f"Destination directory does not exist: {dst}", file=sys.stderr)
            return 1
        missing, extra, changed = compare_trees(src, dst)
        if missing or extra or changed:
            print("Skill trees are out of sync:", file=sys.stderr)
            for rel in missing:
                print(f"  missing in destination: {rel}", file=sys.stderr)
            for rel in extra:
                print(f"  extra in destination: {rel}", file=sys.stderr)
            for rel in changed:
                print(f"  content differs: {rel}", file=sys.stderr)
            return 1
        print(f"Skill trees are in sync: {src} == {dst}")
        return 0

    sync(src, dst)
    missing, extra, changed = compare_trees(src, dst)
    if missing or extra or changed:
        print("Sync finished but differences remain:", file=sys.stderr)
        for rel in missing:
            print(f"  missing in destination: {rel}", file=sys.stderr)
        for rel in extra:
            print(f"  extra in destination: {rel}", file=sys.stderr)
        for rel in changed:
            print(f"  content differs: {rel}", file=sys.stderr)
        return 1

    print(f"Synced skills: {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
