from __future__ import annotations

import ast
import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .cache import CacheManager
from .config import SynsortConfig

BlockKind = str
_CATEGORY_GLOBAL = "globals"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CodeBlock:
    name: str | None
    kind: BlockKind
    category: str
    leading_start: int
    end_line: int
    text: str
    usage: int
    header_indent: str
    dependencies: set[str] = field(default_factory=lambda: set())


@dataclass
class Segment:
    start_line: int
    end_line: int
    blocks: list[CodeBlock]
    indent: str


@dataclass
class SortResult:
    path: Path
    changed: bool
    skipped: bool
    reason: str | None = None


class SynSorter:
    def __init__(
        self, config: SynsortConfig, cache: CacheManager | None = None
    ) -> None:
        self.config = config
        self.cache = cache
        self._header_markers = {
            value.strip() for value in config.section_headers.values()
        }

    def process_file(self, path: Path, *, check: bool = False) -> SortResult:
        text = path.read_text(encoding="utf-8")
        original_digest = _sha256(text)

        if self.cache and self.cache.should_skip(
            path, original_digest, self.config.signature
        ):
            return SortResult(
                path=path, changed=False, skipped=True, reason="cache-hit"
            )

        new_text, changed = self._sort_text(text)
        if not changed:
            if self.cache:
                self.cache.update(path, original_digest, self.config.signature)
            return SortResult(path=path, changed=False, skipped=False, reason=None)

        if check:
            return SortResult(
                path=path, changed=True, skipped=False, reason="needs-formatting"
            )

        path.write_text(new_text, encoding="utf-8")
        if self.cache:
            self.cache.update(path, _sha256(new_text), self.config.signature)
        return SortResult(path=path, changed=True, skipped=False, reason="updated")

    def _sort_text(self, text: str) -> tuple[str, bool]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text, False

        lines = text.splitlines(keepends=True)
        usage = self._collect_usage(tree)
        blocks, guard_blocks = self._build_top_level_blocks(tree, lines, usage)
        main_blocks: list[CodeBlock] = []
        if guard_blocks:
            main_blocks = [
                block
                for block in blocks
                if block.kind == "function" and block.name in {"main", "_main"}
            ]
            if main_blocks:
                removal = {id(block) for block in main_blocks}
                blocks = [block for block in blocks if id(block) not in removal]
                main_blocks.sort(key=self._main_block_sort_key)

        if not blocks and not guard_blocks and not main_blocks:
            return text, False

        for special in [*guard_blocks, *main_blocks]:
            self._clear_range(lines, special.leading_start, special.end_line)

        segments = self._build_segments(blocks, lines, indent="")
        if segments:
            self._coalesce_module_sections(segments)

        rebuilt: list[str] = []
        module_seen: set[str] = set()
        if segments:
            cursor = 1
            for segment in segments:
                rebuilt.append("".join(lines[cursor - 1 : segment.start_line - 1]))
                seen = module_seen if not segment.indent else None
                rebuilt.append(self._render_segment(segment, seen_categories=seen))
                cursor = segment.end_line + 1
            rebuilt.append("".join(lines[cursor - 1 :]))
        else:
            rebuilt.append("".join(lines))

        new_text = "".join(rebuilt)
        if guard_blocks:
            new_text = self._append_guards(new_text, guard_blocks, main_blocks)
        return new_text, new_text != text

    def _build_top_level_blocks(
        self,
        tree: ast.Module,
        lines: list[str],
        usage: Counter[str],
    ) -> tuple[list[CodeBlock], list[CodeBlock]]:
        blocks: list[CodeBlock] = []
        guards: list[CodeBlock] = []
        for node in tree.body:
            if isinstance(node, ast.If) and self._is_main_guard(node):
                guard_block = self._build_guard_block(node, lines)
                if guard_block:
                    guards.append(guard_block)
                continue
            block = self._create_block(node, lines, usage)
            if block:
                blocks.append(block)
        guards.sort(key=lambda block: block.leading_start)
        if len(guards) > 1:
            locations = ", ".join(str(block.leading_start) for block in guards)
            raise ValueError(
                "Multiple '__main__' guards detected (lines "
                f"{locations}). synsort will not reorder this file."
            )
        return blocks, guards

    def _create_block(
        self,
        node: ast.AST,
        lines: list[str],
        usage: Counter[str],
    ) -> CodeBlock | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._build_function_block(node, lines, usage, is_method=False)
        if isinstance(node, ast.ClassDef):
            return self._build_class_block(node, lines, usage)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return self._build_global_block(node, lines, usage)
        return None

    def _build_global_block(
        self,
        node: ast.AST,
        lines: list[str],
        usage: Counter[str],
    ) -> CodeBlock | None:
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            return None
        leading_start = self._extend_with_comments(lines, start_line)
        text = self._slice_text(lines, leading_start, end_line)
        name = self._assignment_name(node)
        usage_score = usage.get(name or "", 0)
        dependencies = self._extract_global_dependencies(node)
        return CodeBlock(
            name=name,
            kind="global",
            category=_CATEGORY_GLOBAL,
            leading_start=leading_start,
            end_line=end_line,
            text=text,
            usage=usage_score,
            header_indent="",
            dependencies=dependencies,
        )

    def _build_guard_block(self, node: ast.If, lines: list[str]) -> CodeBlock | None:
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            return None
        leading_start = self._extend_with_comments(lines, start_line)
        text = self._slice_text(lines, leading_start, end_line)
        return CodeBlock(
            name="__main__",
            kind="guard",
            category="guard",
            leading_start=leading_start,
            end_line=end_line,
            text=text,
            usage=0,
            header_indent="",
        )

    def _build_function_block(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        usage: Counter[str],
        *,
        is_method: bool,
    ) -> CodeBlock:
        start_line = (
            min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
            if node.decorator_list
            else node.lineno
        )
        start_line = self._extend_with_comments(lines, start_line)
        end_line = node.end_lineno or node.lineno
        text = self._slice_text(lines, start_line, end_line)
        indent = self._leading_whitespace(lines[node.lineno - 1])
        category = self._categorize_name(node.name)
        usage_score = usage.get(node.name, 0)
        kind = "method" if is_method else "function"
        header_indent = indent if is_method else ""
        return CodeBlock(
            name=node.name,
            kind=kind,
            category=category,
            leading_start=start_line,
            end_line=end_line,
            text=text,
            usage=usage_score,
            header_indent=header_indent,
        )

    def _build_class_block(
        self,
        node: ast.ClassDef,
        lines: list[str],
        usage: Counter[str],
    ) -> CodeBlock | None:
        start_candidates = [node.lineno]
        if node.decorator_list:
            start_candidates.extend(
                decorator.lineno for decorator in node.decorator_list
            )
        start_line = min(start_candidates)
        start_line = self._extend_with_comments(lines, start_line)
        end_line = node.end_lineno or node.lineno
        raw_text = self._slice_text(lines, start_line, end_line)
        rewritten = self._rewrite_class_body(node, lines, usage, start_line, end_line)
        category = self._categorize_name(node.name)
        usage_score = usage.get(node.name, 0)
        return CodeBlock(
            name=node.name,
            kind="class",
            category=category,
            leading_start=start_line,
            end_line=end_line,
            text=rewritten or raw_text,
            usage=usage_score,
            header_indent="",
        )

    def _rewrite_class_body(
        self,
        node: ast.ClassDef,
        lines: list[str],
        usage: Counter[str],
        leading_start: int,
        end_line: int,
    ) -> str:
        method_blocks: list[CodeBlock] = []
        slots_block: CodeBlock | None = None
        docstring_range = self._docstring_range(node)
        first_body_line = node.body[0].lineno if node.body else node.lineno + 1

        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_blocks.append(
                    self._build_function_block(child, lines, usage, is_method=True)
                )
            elif slots_block is None and self._is_slots_assignment(child):
                slots_block = self._build_slots_block(child, lines)

        class_text = self._slice_text(lines, leading_start, end_line)
        if method_blocks:
            method_indent = method_blocks[0].header_indent
            segments = self._build_segments(method_blocks, lines, indent=method_indent)
            if segments:
                rebuilt: list[str] = []
                cursor = leading_start
                for segment in segments:
                    rebuilt.append("".join(lines[cursor - 1 : segment.start_line - 1]))
                    rebuilt.append(self._render_segment(segment, seen_categories=None))
                    cursor = segment.end_line + 1
                rebuilt.append("".join(lines[cursor - 1 : end_line]))
                class_text = "".join(rebuilt)

        return self._ensure_slots_first(
            class_text,
            slots_block,
            docstring_range,
            leading_start,
            first_body_line,
        )

    def _collect_usage(self, tree: ast.AST) -> Counter[str]:
        counter: Counter[str] = Counter()

        class UsageVisitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name) -> None:  # type: ignore[override]
                if isinstance(node.ctx, ast.Load):
                    counter[node.id] += 1
                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute) -> None:  # type: ignore[override]
                counter[node.attr] += 1
                self.generic_visit(node)

        UsageVisitor().visit(tree)
        return counter

    def _build_segments(
        self,
        blocks: list[CodeBlock],
        lines: list[str],
        *,
        indent: str,
    ) -> list[Segment]:
        if not blocks:
            return []

        ordered = sorted(blocks, key=lambda block: block.leading_start)
        segments: list[Segment] = []
        current: list[CodeBlock] = []
        current_start = 0
        current_end = 0
        previous_end = 0

        for block in ordered:
            if not current:
                current = [block]
                current_start = block.leading_start
                current_end = block.end_line
            else:
                gap_start = previous_end + 1
                gap_end = block.leading_start - 1
                if self._only_comments_or_whitespace(lines, gap_start, gap_end):
                    current.append(block)
                    current_end = block.end_line
                else:
                    adjusted_start = self._extend_segment_start(current_start, lines)
                    segments.append(
                        Segment(adjusted_start, current_end, current, indent)
                    )
                    current = [block]
                    current_start = block.leading_start
                    current_end = block.end_line
            previous_end = block.end_line

        if current:
            adjusted_start = self._extend_segment_start(current_start, lines)
            segments.append(Segment(adjusted_start, current_end, current, indent))

        return segments

    def _coalesce_module_sections(self, segments: list[Segment]) -> None:
        module_segments = [segment for segment in segments if not segment.indent]
        if not module_segments:
            return

        categories = ["globals", "dunder", "public", "private"]
        first_segment = module_segments[0]
        captured_by_category: dict[str, list[CodeBlock]] = {
            category: [] for category in categories
        }
        seen_by_category: dict[str, set[str]] = {
            category: set() for category in categories
        }
        managed_names: set[str] = {
            block.name
            for segment in module_segments
            for block in segment.blocks
            if block.name
        }
        defined_names: set[str] = set()

        for segment in module_segments:
            remaining: list[CodeBlock] = []
            for block in segment.blocks:
                block_name = block.name
                category = block.category

                if (
                    category in categories
                    and block_name
                    and block_name not in seen_by_category[category]
                ):
                    can_capture = True
                    if block.kind == "global":
                        deps = block.dependencies.intersection(managed_names)
                        if not deps.issubset(defined_names):
                            can_capture = False

                    if can_capture:
                        captured_by_category[category].append(block)
                        seen_by_category[category].add(block_name)
                        defined_names.add(block_name)
                        continue

                if category in categories and block_name:
                    seen_by_category[category].add(block_name)

                remaining.append(block)
                if block_name:
                    defined_names.add(block_name)

            segment.blocks = remaining

        insert_sequence: list[CodeBlock] = []
        for category in categories:
            insert_sequence.extend(captured_by_category[category])

        if not insert_sequence:
            return

        first_segment.blocks = insert_sequence + first_segment.blocks

    def _extend_segment_start(self, start_line: int, lines: list[str]) -> int:
        index = start_line - 2
        while index >= 0:
            stripped = lines[index].strip()
            if not stripped:
                start_line -= 1
                index -= 1
                continue
            if stripped in self._header_markers:
                start_line -= 1
                index -= 1
                continue
            break
        return start_line

    def _render_segment(
        self, segment: Segment, *, seen_categories: set[str] | None = None
    ) -> str:
        if not segment.blocks:
            return ""

        is_method_segment = bool(segment.blocks and segment.blocks[0].kind == "method")

        def sort_blocks(blocks: list[CodeBlock]) -> list[CodeBlock]:
            return sorted(
                blocks, key=lambda b: self._block_sort_key(b, is_method_segment)
            )

        blocks_by_category: dict[str, list[CodeBlock]] = {}
        for block in segment.blocks:
            blocks_by_category.setdefault(block.category, []).append(block)

        ordered_blocks: list[CodeBlock] = []
        order_seen: set[str] = set()
        for category in self.config.order:
            category_blocks = blocks_by_category.get(category)
            if category_blocks:
                ordered_blocks.extend(sort_blocks(category_blocks))
                order_seen.add(category)

        extra_categories = [
            category
            for category in blocks_by_category.keys()
            if category not in order_seen
        ]
        for category in sorted(extra_categories):
            ordered_blocks.extend(sort_blocks(blocks_by_category[category]))

        resolved_blocks = self._apply_dependency_ordering(ordered_blocks)

        parts: list[str] = []
        last_category: str | None = None
        for block in resolved_blocks:
            category = block.category
            if category != last_category:
                emit_header = True
                if seen_categories is not None:
                    if category in seen_categories:
                        emit_header = False
                    else:
                        seen_categories.add(category)
                if emit_header:
                    if parts and not parts[-1].endswith("\n\n"):
                        parts.append("\n")
                    header_text = self.config.header_for(category)
                    if is_method_segment:
                        header_text = self._method_header_for(category)
                    parts.append(f"{segment.indent}{header_text}\n")
                elif parts and not parts[-1].endswith("\n\n"):
                    parts.append("\n")
                last_category = category
            block_text = block.text.rstrip()
            parts.append(f"{block_text}\n\n")

        rendered = "".join(parts).rstrip()
        return f"{rendered}\n\n" if rendered else ""

    def _method_header_for(self, category: str) -> str:
        base = self.config.header_for(category)
        replacements = {
            "Members": "Methods",
            "members": "methods",
            "Functions": "Methods",
            "functions": "methods",
        }
        for needle, replacement in replacements.items():
            if needle in base:
                head, tail = base.rsplit(needle, 1)
                return f"{head}{replacement}{tail}"
        return f"{base} (methods)"

    def _extend_with_comments(self, lines: list[str], start_line: int) -> int:
        index = start_line - 2
        boundary = start_line
        while index >= 0:
            stripped = lines[index].strip()
            if not stripped:
                index -= 1
                boundary -= 1
                continue
            if stripped.startswith("#") and stripped not in self._header_markers:
                index -= 1
                boundary -= 1
                continue
            break
        return boundary

    def _only_comments_or_whitespace(
        self, lines: list[str], start_line: int, end_line: int
    ) -> bool:
        if start_line > end_line:
            return True
        for idx in range(max(start_line - 1, 0), min(end_line, len(lines))):
            stripped = lines[idx].strip()
            if stripped and not stripped.startswith("#"):
                return False
        return True

    def _slice_text(self, lines: list[str], start_line: int, end_line: int) -> str:
        return "".join(lines[start_line - 1 : end_line])

    def _leading_whitespace(self, line: str) -> str:
        return line[: len(line) - len(line.lstrip(" \t"))]

    def _clear_range(self, lines: list[str], start_line: int, end_line: int) -> None:
        for idx in range(start_line - 1, end_line):
            if 0 <= idx < len(lines):
                lines[idx] = ""

    def _append_guards(
        self,
        base_text: str,
        guard_blocks: list[CodeBlock],
        main_blocks: list[CodeBlock],
    ) -> str:
        main_chunks = [
            block.text.rstrip() for block in main_blocks if block.text.strip()
        ]
        guard_chunks = [
            block.text.rstrip() for block in guard_blocks if block.text.strip()
        ]
        if not guard_chunks and not main_chunks:
            return base_text

        trailer_parts: list[str] = []
        if main_chunks:
            trailer_parts.append("\n\n".join(main_chunks))
        if guard_chunks:
            trailer_parts.append("\n\n".join(guard_chunks))

        trailer = "\n\n".join(trailer_parts)
        body = base_text.rstrip()
        if body:
            return f"{body}\n\n{trailer}\n"
        return f"{trailer}\n"

    def _block_sort_key(
        self, block: CodeBlock, method_segment: bool
    ) -> tuple[int, int, str, int]:
        priority = 100
        if method_segment and block.kind == "method":
            priority = self._method_priority(block.name)
        return (priority, -block.usage, block.name or "", block.leading_start)

    def _apply_dependency_ordering(self, blocks: list[CodeBlock]) -> list[CodeBlock]:
        if not blocks:
            return blocks

        name_to_block: dict[str, CodeBlock] = {}
        order_index: dict[int, int] = {}
        block_ids = {id(block) for block in blocks}
        for idx, block in enumerate(blocks):
            order_index[id(block)] = idx
            if block.name and block.name not in name_to_block:
                name_to_block[block.name] = block

        graph: dict[int, list[CodeBlock]] = {id(block): [] for block in blocks}
        indegree: dict[int, int] = {id(block): 0 for block in blocks}

        for block in blocks:
            if not block.dependencies:
                continue
            for dep_name in block.dependencies:
                dep_block = name_to_block.get(dep_name)
                if (
                    dep_block
                    and id(dep_block) in block_ids
                    and dep_block is not block
                    and dep_block.kind in {"function", "class"}
                ):
                    graph[id(dep_block)].append(block)
                    indegree[id(block)] += 1

        heap: list[tuple[int, CodeBlock]] = []
        for block in blocks:
            if indegree[id(block)] == 0:
                heapq.heappush(heap, (order_index[id(block)], block))

        resolved: list[CodeBlock] = []
        while heap:
            _, block = heapq.heappop(heap)
            resolved.append(block)
            for neighbor in graph[id(block)]:
                indegree[id(neighbor)] -= 1
                if indegree[id(neighbor)] == 0:
                    heapq.heappush(heap, (order_index[id(neighbor)], neighbor))

        if len(resolved) != len(blocks):
            return blocks
        return resolved

    def _main_block_sort_key(self, block: CodeBlock) -> tuple[int, int]:
        order = 0 if block.name == "_main" else 1
        return (order, block.leading_start)

    def _method_priority(self, name: str | None) -> int:
        if name == "__new__":
            return 0
        if name == "__init__":
            return 1
        return 10

    def _is_slots_assignment(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__slots__":
                    return True
        elif isinstance(node, ast.AnnAssign):
            return isinstance(node.target, ast.Name) and node.target.id == "__slots__"
        return False

    def _build_slots_block(self, node: ast.AST, lines: list[str]) -> CodeBlock | None:
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            return None
        if start_line < 1 or start_line - 1 >= len(lines):
            return None
        leading_start = self._extend_with_comments(lines, start_line)
        text = self._slice_text(lines, leading_start, end_line)
        indent = self._leading_whitespace(lines[start_line - 1])
        return CodeBlock(
            name="__slots__",
            kind="field",
            category="field",
            leading_start=leading_start,
            end_line=end_line,
            text=text,
            usage=0,
            header_indent=indent,
        )

    def _docstring_range(self, node: ast.ClassDef) -> tuple[int, int] | None:
        if not node.body:
            return None
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return first.lineno, getattr(first, "end_lineno", first.lineno)
        return None

    def _ensure_slots_first(
        self,
        class_text: str,
        slots_block: CodeBlock | None,
        docstring_range: tuple[int, int] | None,
        class_start_line: int,
        first_body_line: int,
    ) -> str:
        if not slots_block:
            return class_text
        lines = class_text.splitlines(keepends=True)
        slots_start = slots_block.leading_start - class_start_line
        slots_end = slots_block.end_line - class_start_line + 1
        if slots_start < 0 or slots_end > len(lines):
            return class_text
        snippet = lines[slots_start:slots_end]
        if not snippet:
            return class_text
        del lines[slots_start:slots_end]

        if docstring_range:
            insert_idx = docstring_range[1] - class_start_line + 1
        else:
            insert_idx = max(1, first_body_line - class_start_line)
        insert_idx = max(1, min(insert_idx, len(lines)))
        lines[insert_idx:insert_idx] = snippet
        return "".join(lines)

    def _categorize_name(self, name: str) -> str:
        if name.startswith("__") and name.endswith("__"):
            return "dunder"
        if name.startswith("_"):
            return "private"
        return "public"

    def _assignment_name(self, node: ast.AST) -> str | None:
        target: str | None = None
        if isinstance(node, ast.Assign):
            for possible in node.targets:
                if isinstance(possible, ast.Name):
                    target = possible.id
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target = node.target.id
        return target

    def _extract_global_dependencies(self, node: ast.AST) -> set[str]:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if value is None:
            return set()
        return self._collect_name_dependencies(value)

    def _collect_name_dependencies(self, node: ast.AST) -> set[str]:
        names: set[str] = set()

        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, name_node: ast.Name) -> None:  # type: ignore[override]
                if isinstance(name_node.ctx, ast.Load):
                    names.add(name_node.id)
                self.generic_visit(name_node)

        NameCollector().visit(node)
        return names

    def _is_main_guard(self, node: ast.If) -> bool:
        if not isinstance(node.test, ast.Compare):
            return False
        comparator = node.test.comparators[0] if node.test.comparators else None
        if comparator is None or len(node.test.comparators) != 1:
            return False
        if not node.test.ops or not isinstance(node.test.ops[0], ast.Eq):
            return False

        def _is_name(expr: ast.AST, value: str) -> bool:
            return isinstance(expr, ast.Name) and expr.id == value

        def _is_main_literal(expr: ast.AST) -> bool:
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                return expr.value == "__main__"
            return False

        left = node.test.left
        right = comparator
        return (_is_name(left, "__name__") and _is_main_literal(right)) or (
            _is_name(right, "__name__") and _is_main_literal(left)
        )
