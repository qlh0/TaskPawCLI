from __future__ import annotations

import asyncio

import pytest

from taskpaw.config import load_config
from taskpaw.memory import MemoryManager, MemoryNotFoundError, MemoryScopeError
from taskpaw.prompt import PromptAssembler
from taskpaw.tools import ToolRegistry, get_builtin_tools
from taskpaw.tools.base import ToolContext
from taskpaw.tools.builtins import _update_memory


def _manager(tmp_path, scope="project-a") -> MemoryManager:
    return MemoryManager(tmp_path / "memory.db", scope=scope)


def _context(tmp_path, monkeypatch) -> ToolContext:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.memory.long_term_db_path = str(tmp_path / "memory.db")
    return ToolContext(cwd=str(tmp_path), config=config)


# ---------------------------------------------------------------------------
# Manager level
# ---------------------------------------------------------------------------


def test_update_keeps_id_and_created_at(tmp_path):
    manager = _manager(tmp_path)
    memory_id = manager.save("用户喜欢用 pytest", kind="preference", importance=0.6)
    before = manager.get(memory_id)

    updated, merged = manager.update(memory_id, "用户改用 uv run pytest", importance=0.8)

    assert merged == []
    assert updated.id == memory_id
    assert updated.content == "用户改用 uv run pytest"
    assert updated.created_at == before.created_at
    assert updated.updated_at >= before.updated_at
    assert updated.content_hash != before.content_hash
    assert updated.importance == 0.8
    assert manager.get(memory_id).content == "用户改用 uv run pytest"


def test_update_missing_memory_raises(tmp_path):
    manager = _manager(tmp_path)

    with pytest.raises(MemoryNotFoundError):
        manager.update(999, "anything")


def test_update_rejects_another_scope(tmp_path):
    other = MemoryManager(tmp_path / "memory.db", scope="project-b")
    memory_id = other.save("另一个项目的偏好")

    current = MemoryManager(tmp_path / "memory.db", scope="project-a")
    with pytest.raises(MemoryScopeError):
        current.update(memory_id, "不该被改")

    assert other.get(memory_id).content == "另一个项目的偏好"


def test_update_validates_content(tmp_path):
    manager = _manager(tmp_path)
    memory_id = manager.save("原始内容")

    with pytest.raises(ValueError):
        manager.update(memory_id, "   ")
    with pytest.raises(ValueError):
        manager.update(memory_id, "x" * (manager.max_content_length + 1))
    with pytest.raises(ValueError):
        manager.update(memory_id, "新内容", importance=1.5)


def test_update_with_identical_content_is_a_noop(tmp_path):
    manager = _manager(tmp_path)
    memory_id = manager.save("完全相同的内容")
    before = manager.get(memory_id)

    updated, merged = manager.update(memory_id, "完全相同的内容")

    assert merged == []
    assert updated.updated_at == before.updated_at


def test_hash_conflict_merges_into_the_updated_row(tmp_path):
    manager = _manager(tmp_path)
    first = manager.save("旧内容", importance=0.4, confidence=0.5)
    second = manager.save("目标内容", importance=0.9, confidence=0.8)

    updated, merged = manager.update(first, "目标内容", importance=0.6)

    assert merged == [second]
    assert updated.id == first
    assert updated.content == "目标内容"
    assert updated.importance == 0.9  # max of both
    assert updated.confidence == 0.8  # max of both
    assert manager.get(second) is None
    rows = manager.list(limit=10)
    assert [row.id for row in rows] == [first]


# ---------------------------------------------------------------------------
# Tool level
# ---------------------------------------------------------------------------


def test_update_memory_tool_is_registered_without_forced_approval():
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    tool = registry.get("update_memory")

    assert tool is not None
    assert tool.requires_approval is False
    assert tool.is_read_only is False
    assert tool.parameters["required"] == ["memory_id", "content"]


def test_update_memory_handler_rewrites_the_entry(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    manager = MemoryManager(str(tmp_path / "memory.db"), scope=str(tmp_path))
    memory_id = manager.save("旧偏好")

    result = asyncio.run(_update_memory({"memory_id": memory_id, "content": "新偏好"}, context))

    assert result.is_error is False
    assert f"#{memory_id}" in result.content
    assert manager.get(memory_id).content == "新偏好"


def test_update_memory_handler_reports_missing_entry(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)

    result = asyncio.run(_update_memory({"memory_id": 12345, "content": "新内容"}, context))

    assert result.is_error is True
    assert "does not exist" in result.content


def test_update_memory_handler_rejects_foreign_scope(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    other = MemoryManager(str(tmp_path / "memory.db"), scope=str(tmp_path / "elsewhere"))
    memory_id = other.save("别处的内容")

    result = asyncio.run(_update_memory({"memory_id": memory_id, "content": "越权修改"}, context))

    assert result.is_error is True
    assert "different project scope" in result.content


# ---------------------------------------------------------------------------
# Prompt guidance
# ---------------------------------------------------------------------------


def test_prompt_guides_correction_to_update_memory(tmp_path):
    config = load_config(project_root=tmp_path)
    prompt = PromptAssembler(
        config=config,
        cwd=str(tmp_path),
        tool_names=["search_memory", "update_memory", "save_memory"],
        model="fake",
        provider="fake",
    ).build_static()

    assert "update_memory" in prompt
    assert "search_memory first" in prompt
