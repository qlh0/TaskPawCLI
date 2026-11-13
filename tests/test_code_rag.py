from __future__ import annotations

import asyncio

from taskpaw.config import load_config
from taskpaw.memory import HashEmbedder
from taskpaw.rag import CodeIndex, chunk_code
from taskpaw.tools.base import ToolContext
from taskpaw.tools.builtins import _search_code

PY_SOURCE = '''"""demo module."""

import os

CONSTANT = 1


class Greeter:
    """Say hello."""

    greeting = "hi"

    def greet(self, name):
        return f"{self.greeting} {name}"


def helper(value):
    return value * 2
'''


def _index(tmp_path, *, dim: int = 32, **overrides) -> CodeIndex:
    return CodeIndex(
        tmp_path,
        embedder=HashEmbedder(dim),
        min_similarity=0.0,
        **overrides,
    )


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_python_files_are_chunked_by_definition():
    chunks = chunk_code("demo.py", PY_SOURCE)

    by_name = {chunk.name: chunk for chunk in chunks if chunk.name}
    assert "Greeter" in by_name
    assert "Greeter.greet" in by_name
    assert "helper" in by_name
    assert by_name["Greeter.greet"].kind == "method"
    assert by_name["helper"].kind == "function"
    greet = by_name["Greeter.greet"]
    lines = PY_SOURCE.splitlines()
    assert "def greet" in lines[greet.start_line - 1]
    assert "return f" in lines[greet.end_line - 1]


def test_non_python_files_fall_back_to_blocks():
    text = "line one\nline two\n\nline three\nline four\n"
    chunks = chunk_code("notes.txt", text)

    assert chunks
    assert all(chunk.kind == "block" for chunk in chunks)
    assert chunks[0].start_line == 1


def test_long_blocks_are_split():
    text = "\n".join(f"line {index}" for index in range(300))
    chunks = chunk_code("long.txt", text, max_lines=100)

    assert len(chunks) >= 3
    assert all(chunk.end_line - chunk.start_line + 1 <= 100 for chunk in chunks)


# ---------------------------------------------------------------------------
# Indexing and retrieval
# ---------------------------------------------------------------------------


def test_rebuild_and_keyword_search(tmp_path):
    _write(tmp_path / "demo.py", PY_SOURCE)
    index = _index(tmp_path)

    count = index.rebuild()

    assert count >= 3
    results = index.search("greet", limit=5)
    assert results
    assert results[0].path == "demo.py"
    assert results[0].line <= results[0].end_line
    assert "Greeter.greet" in {item.name for item in results}


def test_search_survives_without_fts(tmp_path):
    _write(tmp_path / "demo.py", PY_SOURCE)
    index = _index(tmp_path)
    index.rebuild()
    index.fts_enabled = False

    results = index.search("greet", limit=5)

    assert results
    assert results[0].path == "demo.py"


def test_incremental_update_skips_unchanged_files(tmp_path):
    demo = tmp_path / "demo.py"
    _write(demo, PY_SOURCE)
    index = _index(tmp_path)
    index.rebuild()

    assert index.update() == 0

    _write(demo, PY_SOURCE + "\n\ndef extra():\n    return 42\n")
    assert index.update() > 0
    assert "extra" in {item.name for item in index.search("extra", limit=5)}


def test_update_removes_deleted_files(tmp_path):
    demo = tmp_path / "demo.py"
    _write(demo, PY_SOURCE)
    index = _index(tmp_path)
    index.rebuild()
    demo.unlink()

    assert index.update() == 0
    assert index.search("greet", limit=5) == []


def test_vector_candidates_join_the_fused_results(tmp_path):
    _write(tmp_path / "demo.py", PY_SOURCE)
    _write(tmp_path / "other.py", "def unrelated_topic():\n    return 1\n")
    index = _index(tmp_path)
    index.rebuild()

    keyword = index.search("helper", limit=5)
    assert keyword

    def fake_vector(conn, query):
        rows = conn.execute("select id from code_chunks_v2 where path = 'other.py'").fetchall()
        return index._load_results(conn, [int(row[0]) for row in rows])

    original = index._vector_candidates
    index._vector_candidates = fake_vector  # type: ignore[method-assign]
    try:
        fused = index.search("helper", limit=6)
    finally:
        index._vector_candidates = original  # type: ignore[method-assign]

    assert "other.py" in {item.path for item in fused}


def test_changed_embedding_model_requires_rebuild(tmp_path):
    _write(tmp_path / "demo.py", PY_SOURCE)
    _index(tmp_path, dim=32).rebuild()

    changed = CodeIndex(tmp_path, embedder=HashEmbedder(64), min_similarity=0.0)
    results = changed.search("greet", limit=5)

    assert results
    assert "rebuild" in changed.vector_skip_reason


# ---------------------------------------------------------------------------
# Tool integration
# ---------------------------------------------------------------------------


def test_search_code_tool_returns_line_ranges(tmp_path, monkeypatch):
    _write(tmp_path / "demo.py", PY_SOURCE)
    CodeIndex(tmp_path).rebuild()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    context = ToolContext(cwd=str(tmp_path), config=config)

    result = asyncio.run(_search_code({"query": "greet"}, context))

    assert result.is_error is False
    assert "demo.py:" in result.content
    assert "Greeter.greet" in result.content
