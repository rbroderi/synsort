from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from synsort import SynsortConfig, SynSorter

COMPLEX_SOURCE = textwrap.dedent(
    '''
    """Module docstring that should stay intact after sorting."""

    from __future__ import annotations

    import asyncio
    from functools import lru_cache, wraps
    from pathlib import Path
    from typing import TYPE_CHECKING

    if __name__ == "__main__":
        raise SystemExit(main())

    CONFIG_PATH = "config.toml"
    __version__ = "0.0.1"
    AGENT_POOL: dict[str, int] = {"alpha": 1, "beta": 2}
    _PRIVATE_KEY = object()

    # module level comment tied to SETTINGS
    SETTINGS = {name: idx for idx, name in enumerate(("alpha", "beta", "gamma"))}

    def helper(value: int) -> int:
        return value * 2

    @wraps(helper)
    def decorated(value: int) -> int:
        return helper(value)

    async def bootstrap(config: dict[str, str]) -> None:
        await asyncio.sleep(len(config) / 10)

    def build_pipeline() -> list[int]:
        return [helper(idx) for idx in range(3)]

    if TYPE_CHECKING:
        from typing import TypedDict

        class WorkflowSpec(TypedDict):
            threshold: int

    PIPELINE = build_pipeline()

    class Workflow:
        """Docstring with literal braces {CONFIG_PATH!s}."""

        level = 1
        __slots__ = ("state", "history")

        def __new__(cls, value: int) -> "Workflow":
            instance = super().__new__(cls)
            instance.bootstrap = value
            return instance

        def __init__(self, value: int) -> None:
            self.state = {"value": value}
            self.history: list[int] = []

        @classmethod
        def create(cls, value: int) -> "Workflow":
            return cls(value)

        @lru_cache
        def public_cached(self, number: int) -> int:
            return self._private(number)

        def __repr__(self) -> str:
            return f"Workflow(state={self.state!r})"

        def public_api(self, payload: dict[str, int]) -> tuple[int, int]:
            total = sum(payload.values())
            return total, self._private(total)

        async def runner(self, queue: asyncio.Queue[int]) -> list[int]:
            results: list[int] = []
            async for chunk in queue:
                results.append(chunk)
            return results

        def _private(self, number: int) -> int:
            return number // 2

        async def __aenter__(self) -> "Workflow":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    def _main(argv: list[str] | None = None) -> int:
        payload = [helper(idx) for idx, _ in enumerate(argv or ("demo",))]
        return sum(payload)

    def main() -> int:
        return _main(["alpha", "beta", "gamma"])

    # public dispatch relies on structural pattern matching
    def public_dispatch(value: object) -> str:
        match value:
            case {"name": str(name), **rest}:
                return name
            case [first, *rest]:
                return f"{first}:{len(rest)}"
            case _:
                return repr(value)

    CONFIG_PATH = Path(CONFIG_PATH)
    '''
).lstrip()


EXTREME_SOURCE = textwrap.dedent(
    '''
    """Stress-case module exercising tricky syntax combinations."""

    from __future__ import annotations

    import asyncio
    import contextlib
    from dataclasses import dataclass
    from functools import singledispatch
    from typing import TYPE_CHECKING, Any, Callable, Protocol

    __all__ = ["Pipeline", "build_handlers", "route_payload"]

    PIPELINE_LABEL = "extreme"

    type HandlerFactory = Callable[[str], "BaseHandler"]
    type StepSpec = dict[str, Callable[[int], int]]

    if TYPE_CHECKING:
        from collections.abc import AsyncIterator

        class SupportsClose(Protocol):
            def close(self) -> None: ...

    @dataclass(slots=True)
    class Config:
        retries: int
        jitter: float

    CONFIG = Config(retries=3, jitter=0.5)

    class BaseHandler:
        def __init__(self, name: str) -> None:
            self.name = name

        def handle(self, payload: str) -> str:
            return payload.upper()

    class Pipeline:
        __slots__ = ("handlers",)

        def __init__(self, handlers: list[BaseHandler]) -> None:
            self.handlers = handlers

        async def drain(self, queue: asyncio.Queue[str]) -> list[str]:
            results: list[str] = []
            while chunk := await queue.get():
                results.append(chunk)
                if queue.empty():
                    break
            return results

        def _ensure(self, payload: str) -> str:
            return payload or self.handlers[0].name

    def build_handlers(factory: HandlerFactory) -> list[BaseHandler]:
        return [factory("alpha"), factory("beta")]

    def _make(name: str) -> BaseHandler:
        return BaseHandler(name)

    REGISTRY: dict[str, HandlerFactory] = {name: _make for name in ("alpha", "beta")}

    @contextlib.contextmanager
    def manage(resource: SupportsClose | None):
        try:
            yield resource
        finally:
            if resource is not None:
                resource.close()

    @singledispatch
    def route_payload(value: object) -> str:
        if isinstance(value, dict):
            match value:
                case {"kind": kind, **rest}:
                    return kind
        return str(value)

    @route_payload.register(int)
    def _(value: int) -> str:
        return f"int:{value}"

    async def stream(iterator: AsyncIterator[str]) -> list[str]:
        collected: list[str] = []
        async for item in iterator:
            collected.append(item)
        return collected

    def _main() -> int:
        return len(REGISTRY)

    def main() -> int:
        return _main()

    PIPELINE = Pipeline(build_handlers(_make))

    if __name__ == "__main__":
        raise SystemExit(main())
    '''
).lstrip()


def test_sorter_preserves_syntax_for_complex_module(tmp_path: Path) -> None:
    source = tmp_path / "complex_module.py"
    source.write_text(COMPLEX_SOURCE, encoding="utf-8")

    config = SynsortConfig.load(tmp_path)
    sorter = SynSorter(config)

    result = sorter.process_file(source)
    text = source.read_text(encoding="utf-8")
    compile(text, str(source), "exec")
    assert not result.skipped

    workflow_section = text.split("class Workflow:", 1)[1].split(
        "# public dispatch", 1
    )[0]
    assert workflow_section.index("__slots__") < workflow_section.index("level = 1")
    assert workflow_section.index("def __new__") < workflow_section.index(
        "def __init__"
    )

    guard_marker = 'if __name__ == "__main__":'
    assert text.count(guard_marker) == 1
    tail = text.rstrip()
    assert tail.endswith('if __name__ == "__main__":\n    raise SystemExit(main())')
    before_guard = tail[: tail.rfind(guard_marker)].rstrip()
    assert before_guard.endswith(
        'def main() -> int:\n    return _main(["alpha", "beta", "gamma"])'
    )
    assert text.index("def build_pipeline() -> list[int]:") < text.index(
        "PIPELINE = build_pipeline()"
    )
    assert text.index("PIPELINE = build_pipeline()") < text.index("if TYPE_CHECKING:")
    assert text.index("CONFIG_PATH = Path(CONFIG_PATH)") > text.index(
        "if TYPE_CHECKING:"
    )
    assert text.count(config.header_for("globals")) == 1
    assert text.count(config.header_for("public")) == 1
    assert text.count(sorter._method_header_for("public")) == 1
    assert text.count(sorter._method_header_for("dunder")) == 1
    assert text.count(sorter._method_header_for("private")) == 1


def test_multiple_main_guards_raise(tmp_path: Path) -> None:
    original = textwrap.dedent(
        """
        if __name__ == "__main__":
            print("one")

        VALUE = 1

        if __name__ == "__main__":
            print("two")
        """
    ).lstrip()

    source = tmp_path / "dupe.py"
    source.write_text(original, encoding="utf-8")

    sorter = SynSorter(SynsortConfig.load(tmp_path))

    with pytest.raises(ValueError, match="Multiple '__main__' guards"):
        sorter.process_file(source)

    assert source.read_text(encoding="utf-8") == original


def test_global_not_moved_when_dependency_defined_later(tmp_path: Path) -> None:
    source_text = textwrap.dedent(
        """
        from typing import TYPE_CHECKING

        ALPHA = 1

        if TYPE_CHECKING:
            from typing import Protocol

        CONSTANT = build_value()

        def build_value() -> int:
            return 41 + 1
        """
    ).lstrip()

    source = tmp_path / "delayed.py"
    source.write_text(source_text, encoding="utf-8")

    sorter = SynSorter(SynsortConfig.load(tmp_path))
    sorter.process_file(source)

    text = source.read_text(encoding="utf-8")
    assert text.index("CONSTANT = build_value()") > text.index("if TYPE_CHECKING:")


def test_sorter_handles_extreme_constructs(tmp_path: Path) -> None:
    source = tmp_path / "extreme_module.py"
    source.write_text(EXTREME_SOURCE, encoding="utf-8")

    sorter = SynSorter(SynsortConfig.load(tmp_path))
    result = sorter.process_file(source)
    text = source.read_text(encoding="utf-8")

    compile(text, str(source), "exec")
    assert not result.skipped

    assert text.count("__all__ = ") == 1
    assert "route_payload" in text.split("__all__ =", 1)[1]

    assert 'type HandlerFactory = Callable[[str], "BaseHandler"]' in text
    assert text.count("if TYPE_CHECKING:") == 1

    assert text.index("def _make(") < text.index("REGISTRY: dict[str, HandlerFactory]")

    decorator_idx = text.index("@contextlib.contextmanager")
    assert decorator_idx < text.index("def manage(")

    assert "@route_payload.register(int)" in text
    assert text.index("@route_payload.register(int)") < text.index("def _(value: int)")

    pipeline_idx = text.index("class Pipeline")
    assert pipeline_idx < text.index("PIPELINE = Pipeline")

    assert "match value:" in text
