from __future__ import annotations

import json

import pytest

import taskpaw.memory.vector_store as vector_store
from taskpaw.config import load_config
from taskpaw.memory import HashEmbedder, MemoryManager, create_embedder
from taskpaw.memory.vector_store import MemoryVectorIndex


def _manager(tmp_path, *, dim: int = 32, scope: str = "project") -> MemoryManager:
    return MemoryManager(
        tmp_path / "memory.db",
        scope,
        embedder=HashEmbedder(dim),
        embedding_min_similarity=0.0,
    )


def _vector_count(manager: MemoryManager) -> int:
    with manager._connect() as conn:
        index = MemoryVectorIndex(
            conn,
            dim=manager.embedder.dim,
            model=manager.embedder.model_name,
        )
        return index.count()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


def test_hash_embedder_is_deterministic_and_normalised():
    embedder = HashEmbedder(16)
    first = embedder.embed(["pytest runner"])[0]
    second = embedder.embed(["pytest runner"])[0]
    other = embedder.embed(["docker deploy"])[0]

    assert first == second
    assert first != other
    assert len(first) == 16
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Storage and sync
# ---------------------------------------------------------------------------


def test_save_stores_a_vector(tmp_path):
    manager = _manager(tmp_path)
    manager.save("用户偏好使用 pytest")

    assert _vector_count(manager) == 1


def test_update_refreshes_the_vector(tmp_path):
    manager = _manager(tmp_path)
    memory_id = manager.save("旧内容 alpha")
    manager.update(memory_id, "新内容 beta")

    assert _vector_count(manager) == 1
    hits = manager.recall("beta", limit=5)
    assert [entry.id for entry in hits] == [memory_id]


def test_delete_removes_the_vector(tmp_path):
    manager = _manager(tmp_path)
    memory_id = manager.save("会被删除的记忆")
    manager.delete(memory_id)

    assert _vector_count(manager) == 0
    assert manager.recall("会被删除的记忆", limit=5) == []


def test_clear_removes_all_vectors(tmp_path):
    manager = _manager(tmp_path)
    manager.save("第一条")
    manager.save("第二条")
    manager.clear()

    assert _vector_count(manager) == 0


def test_reindex_backfills_existing_memories(tmp_path):
    plain = MemoryManager(tmp_path / "memory.db", "project")
    plain.save("pytest 测试偏好")
    plain.save("docker 部署约束")

    manager = MemoryManager(
        tmp_path / "memory.db",
        "project",
        embedder=HashEmbedder(32),
        embedding_min_similarity=0.0,
    )
    assert manager.reindex() == 2
    assert _vector_count(manager) == 2


def test_recall_fuses_keyword_and_vector_candidates(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    keyword_id = manager.save("alpha 相关记忆")
    vector_only = manager.save("beta 完全不同的记忆")

    monkeypatch.setattr(
        manager,
        "_vector_candidates",
        lambda query, kinds: [manager.get(vector_only)],
    )

    hits = manager.recall("alpha", limit=5)

    assert {entry.id for entry in hits} == {keyword_id, vector_only}


def test_fuse_orders_by_rrf_score_and_dedupes(tmp_path):
    manager = _manager(tmp_path)
    first = manager.save("one")
    second = manager.save("two")
    a = manager.get(first)
    b = manager.get(second)

    fused = manager._fuse([a], [b, a], 5)

    assert [entry.id for entry in fused] == [first, second]


def test_model_change_disables_vector_recall(tmp_path):
    manager = _manager(tmp_path, dim=32)
    manager.save("模型变更测试")

    changed = MemoryManager(
        tmp_path / "memory.db",
        "project",
        embedder=HashEmbedder(64),
        embedding_min_similarity=0.0,
    )
    hits = changed.recall("模型变更测试", limit=5)

    assert [entry.id for entry in hits] == [1]
    assert "reindex" in changed.vector_skip_reason


def test_blob_fallback_still_searches(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "load_sqlite_vec", lambda conn: False)
    manager = _manager(tmp_path)
    memory_id = manager.save("fallback searchable")

    with manager._connect() as conn:
        index = MemoryVectorIndex(conn, dim=32, model="hash")
        assert index.kind == "blob"
        hits = index.search(
            HashEmbedder(32).embed(["fallback searchable"])[0],
            limit=3,
            min_similarity=0.0,
        )
    assert hits[0][0] == memory_id


def test_stored_vectors_are_normalised_before_search(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "v.db")
    index = MemoryVectorIndex(conn, dim=4, model="hash")
    index.upsert(1, [10.0, 0.0, 0.0, 0.0])

    hits = index.search([1.0, 0.0, 0.0, 0.0], limit=3, min_similarity=0.0)

    assert hits and hits[0][0] == 1
    assert hits[0][1] > 0.99
    conn.close()


def test_sqlite_vec_roundtrip(tmp_path):
    manager = _manager(tmp_path)
    memory_id = manager.save("vector roundtrip")

    with manager._connect() as conn:
        index = MemoryVectorIndex(conn, dim=32, model="hash")
        if index.kind != "vec0":
            pytest.skip("sqlite-vec fell back to the blob index")
        hits = index.search(
            HashEmbedder(32).embed(["vector roundtrip"])[0],
            limit=3,
            min_similarity=0.0,
        )
    assert hits[0][0] == memory_id


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_embeddings_config_section_enables_hash_backend(tmp_path):
    config_dir = tmp_path / ".taskpaw"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"embeddings": {"enabled": True, "backend": "hash", "dim": 16}}),
        encoding="utf-8",
    )

    config = load_config(project_root=tmp_path)
    embedder = create_embedder(config)

    assert config.embeddings.enabled is True
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dim == 16
