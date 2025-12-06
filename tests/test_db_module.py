from __future__ import annotations

import ast
import importlib
import sys
from collections import Counter
from pathlib import Path

from synsort import SynsortConfig, SynSorter

DB_MODULE = "tests.data.db"
DB_PATH = Path(__file__).parent / "data" / "db.py"


def _function_counts(source: str) -> Counter[str]:
    tree = ast.parse(source)
    counts: Counter[str] = Counter()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] += 1
    return counts


def _reload_db_module():
    if DB_MODULE in sys.modules:
        del sys.modules[DB_MODULE]
    importlib.invalidate_caches()
    return importlib.import_module(DB_MODULE)


def test_db_module_remains_importable_after_sort(tmp_path: Path) -> None:
    """Ensure running synsort on db.py keeps it importable and restores it."""

    original_source = DB_PATH.read_text(encoding="utf-8")

    try:
        module_before = _reload_db_module()
        assert module_before is not None

        sorter = SynSorter(SynsortConfig.load(tmp_path))
        sorter.process_file(DB_PATH)

        module_after = _reload_db_module()
        assert module_after is not None
    finally:
        DB_PATH.write_text(original_source, encoding="utf-8")
        restored = _reload_db_module()
        assert restored is not None


def test_db_module_definitions_not_duplicated(tmp_path: Path) -> None:
    original_source = DB_PATH.read_text(encoding="utf-8")
    tmp_file = tmp_path / "db.py"
    tmp_file.write_text(original_source, encoding="utf-8")

    sorter = SynSorter(SynsortConfig.load(tmp_path))

    before = _function_counts(original_source)
    sorter.process_file(tmp_file)
    after = _function_counts(tmp_file.read_text(encoding="utf-8"))

    assert before == after
