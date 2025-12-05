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
