from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from .cache import CacheManager
from .config import SynsortConfig
from .sorter import SortResult, SynSorter

_EXCLUDED_DIRS = {".git", ".venv", "__pycache__"}


def _find_project_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for ancestor in [current, *current.parents]:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return current


def _iter_python_files(targets: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if target.is_file() and target.suffix == ".py":
            files.append(target)
            continue
        if target.is_dir():
            for candidate in target.rglob("*.py"):
                if any(part in _EXCLUDED_DIRS for part in candidate.parts):
                    continue
                files.append(candidate)
    return sorted(set(files))


def _print_result(result: SortResult) -> None:
    status = "UNCHANGED"
    if result.skipped:
        status = result.reason or "SKIPPED"
    elif result.changed:
        status = result.reason or "UPDATED"
    print(f"[{status}] {result.path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sort Python globals, functions, classes, and methods."
    )
    parser.add_argument(
        "paths", nargs="*", default=["."], help="Files or directories to lint"
    )
    parser.add_argument(
        "--check", action="store_true", help="Only check, do not rewrite files"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable the cache layer"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Drop the cache before running"
    )
    parser.add_argument(
        "--root", type=Path, help="Override the project root used for config and cache"
    )

    args = parser.parse_args(argv)

    targets = [Path(arg).resolve() for arg in args.paths]
    root = args.root.resolve() if args.root else _find_project_root(targets[0])

    config = SynsortConfig.load(root)
    cache_manager = None if args.no_cache else CacheManager(root / config.cache_file)

    if cache_manager and args.clear_cache:
        cache_manager.clear()
        cache_manager.flush()

    sorter = SynSorter(config, cache_manager)
    python_files = _iter_python_files(targets)
    if not python_files:
        print("No Python files found.")
        return 0

    changed = False
    failed = False
    for file_path in python_files:
        try:
            result = sorter.process_file(file_path, check=args.check)
            _print_result(result)
            changed = changed or result.changed
        except Exception as exc:  # pragma: no cover - defensive
            failed = True
            print(f"[ERROR] {file_path}: {exc}", file=sys.stderr)

    if cache_manager:
        cache_manager.flush()

    if failed:
        return 2
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
