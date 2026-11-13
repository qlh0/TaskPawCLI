"""chunker.py — split source files into search-friendly code chunks.

Python files are chunked by AST node (function / class / method) so a search hit
covers a whole definition instead of a single line. Other languages fall back to
blank-line blocks with a hard line cap.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(slots=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    kind: str
    name: str
    content: str


def chunk_code(path: str, text: str, *, max_lines: int = 120) -> list[CodeChunk]:
    if path.endswith(".py"):
        try:
            return _chunk_python(path, text, max_lines)
        except SyntaxError:
            pass
    return _chunk_blocks(path, text, max_lines)


def _chunk_python(path: str, text: str, max_lines: int) -> list[CodeChunk]:
    tree = ast.parse(text)
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    nodes = [node for node in tree.body if _is_def(node)]
    first_line = min((_node_start(node) for node in nodes), default=0)
    if first_line > 1 and any(line.strip() for line in lines[: first_line - 1]):
        chunks.extend(_make(path, lines, 1, first_line - 1, "module", "", max_lines))

    for node in nodes:
        if isinstance(node, ast.ClassDef):
            methods = [item for item in node.body if _is_def(item)]
            start = _node_start(node)
            end = (methods[0].lineno - 1) if methods else (node.end_lineno or start)
            if end >= start:
                chunks.extend(_make(path, lines, start, end, "class", node.name, max_lines))
            for method in methods:
                chunks.extend(
                    _node_chunks(path, lines, method, prefix=f"{node.name}.", max_lines=max_lines)
                )
        else:
            chunks.extend(_node_chunks(path, lines, node, prefix="", max_lines=max_lines))

    chunks.sort(key=lambda chunk: (chunk.start_line, chunk.end_line))
    return chunks


def _is_def(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", []) or []
    return min([node.lineno] + [item.lineno for item in decorators])


def _node_chunks(
    path: str,
    lines: list[str],
    node: ast.AST,
    *,
    prefix: str,
    max_lines: int,
) -> list[CodeChunk]:
    start = _node_start(node)
    end = getattr(node, "end_lineno", None) or start
    kind = "method" if prefix else "function"
    name = f"{prefix}{getattr(node, 'name', '')}"
    return _make(path, lines, start, end, kind, name, max_lines)


def _make(
    path: str,
    lines: list[str],
    start: int,
    end: int,
    kind: str,
    name: str,
    max_lines: int,
) -> list[CodeChunk]:
    start = max(1, start)
    end = min(len(lines), end)
    if end < start:
        return []
    span = end - start + 1
    if span <= max_lines:
        content = "\n".join(lines[start - 1 : end]).strip("\n")
        if not content.strip():
            return []
        return [CodeChunk(path, start, end, kind, name, content)]

    chunks: list[CodeChunk] = []
    cursor = start
    part = 1
    while cursor <= end:
        part_end = min(end, cursor + max_lines - 1)
        content = "\n".join(lines[cursor - 1 : part_end]).strip("\n")
        if content.strip():
            label = f"{name} (part {part})" if name else f"part {part}"
            chunks.append(CodeChunk(path, cursor, part_end, kind, label, content))
        part += 1
        cursor = part_end + 1
    return chunks


def _chunk_blocks(path: str, text: str, max_lines: int) -> list[CodeChunk]:
    lines = text.splitlines()
    blocks: list[tuple[int, int]] = []
    start = None
    for index, line in enumerate(lines, start=1):
        if line.strip():
            start = index if start is None else start
        elif start is not None:
            blocks.append((start, index - 1))
            start = None
    if start is not None:
        blocks.append((start, len(lines)))

    merged: list[tuple[int, int]] = []
    for block_start, block_end in blocks:
        if merged and block_end - merged[-1][0] + 1 <= max(20, max_lines // 3):
            merged[-1] = (merged[-1][0], block_end)
        else:
            merged.append((block_start, block_end))

    chunks: list[CodeChunk] = []
    for block_start, block_end in merged:
        chunks.extend(_make(path, lines, block_start, block_end, "block", "", max_lines))
    return chunks
