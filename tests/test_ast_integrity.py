from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from textwrap import dedent

from synsort import SynsortConfig, SynSorter

SAMPLE_SOURCE = dedent(
    """
    import asyncio

    VALUE = 1

    def alpha(x: int) -> int:
        return beta(x) + VALUE

    def beta(x: int) -> int:
        return x * 2

    class Widget:
        FACTOR = 2

        def __init__(self, value: int) -> None:
            self.value = value

        @classmethod
        def build(cls, value: int) -> "Widget":
            return cls(value)

        async def scatter(self, queue: asyncio.Queue[int]) -> list[int]:
            return [item async for item in queue]

        def helper(self, term: int) -> int:
            return term * self.FACTOR

    class Gizmo:
        def process(self, value: int) -> int:
            def inner(payload: int) -> int:
                return payload + 1

            return inner(value) * beta(value)

    def gamma(flag: bool) -> int:
        if flag:
            return alpha(1)
        return beta(2)
    """
).lstrip()


def _collect_function_signatures(source: str) -> Counter[tuple[str, str]]:
    tree = ast.parse(source)
    signatures: Counter[tuple[str, str]] = Counter()

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
            self._record(node)

        def visit_AsyncFunctionDef(  # type: ignore[override]
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self._record(node)

        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            qual = ".".join([*self.stack, node.name]) if self.stack else node.name
            dump = ast.dump(node, include_attributes=False)
            signatures[(qual, dump)] += 1

    Collector().visit(tree)
    return signatures


def test_sorter_preserves_function_nodes(tmp_path: Path) -> None:
    path = tmp_path / "ast_integrity.py"
    path.write_text(SAMPLE_SOURCE, encoding="utf-8")

    sorter = SynSorter(SynsortConfig.load(tmp_path))

    before = _collect_function_signatures(SAMPLE_SOURCE)
    sorter.process_file(path)
    after = _collect_function_signatures(path.read_text(encoding="utf-8"))

    assert before == after
