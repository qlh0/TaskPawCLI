from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import unicodedata
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taskpaw.memory.embedding import Embedder
from taskpaw.memory.vector_store import MemoryVectorIndex, reset_index

DEFAULT_MAX_ENTRIES_PER_SCOPE = 1_000
DEFAULT_MAX_CONTENT_LENGTH = 8_000
_MAX_METADATA_LENGTH = 128
_ENTRY_COLUMNS = """
    id, scope, content, created_at, kind, source, importance, confidence,
    updated_at, expires_at, access_count, content_hash
"""
_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


class MemoryNotFoundError(LookupError):
    """Raised when the requested memory id does not exist."""


class MemoryScopeError(PermissionError):
    """Raised when the memory belongs to another project scope."""


@dataclass(slots=True)
class MemoryEntry:
    id: int
    scope: str
    content: str
    created_at: str
    kind: str = "fact"
    source: str = "manual"
    importance: float = 0.5
    confidence: float = 1.0
    updated_at: str = ""
    expires_at: str | None = None
    access_count: int = 0
    content_hash: str = ""


class MemoryManager:
    def __init__(
        self,
        db_path: str | Path,
        scope: str,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES_PER_SCOPE,
        max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
        embedder: Embedder | None = None,
        embedding_min_similarity: float = 0.25,
        candidate_depth: int = 30,
        rrf_k: int = 60,
    ):
        self.db_path = Path(db_path).expanduser()
        self.scope = str(scope)
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_content_length = _positive_int(max_content_length, "max_content_length")
        self.embedder = embedder
        self.embedding_min_similarity = float(embedding_min_similarity)
        self.candidate_depth = max(1, int(candidate_depth))
        self.rrf_k = max(1, int(rrf_k))
        self.vector_skip_reason = ""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save(
        self,
        content: str,
        *,
        kind: str = "fact",
        source: str = "manual",
        importance: float = 0.5,
        confidence: float = 1.0,
        expires_at: str | datetime | None = None,
    ) -> int:
        cleaned = _clean_content(content)
        if len(cleaned) > self.max_content_length:
            raise ValueError(f"memory content exceeds {self.max_content_length} characters")
        normalized_kind = _validate_metadata(kind, "kind")
        normalized_source = _validate_metadata(source, "source")
        normalized_importance = _validate_score(importance, "importance")
        normalized_confidence = _validate_score(confidence, "confidence")
        normalized_expiry = _normalize_optional_timestamp(expires_at)
        now = _now_iso()
        content_hash = _hash_content(cleaned)
        vector = self._embed_one(cleaned)

        with self._connect() as conn:
            conn.execute("begin immediate")
            self._purge_expired(conn)
            row = conn.execute(
                f"""
                select {_ENTRY_COLUMNS}
                from memories
                where scope = ? and content_hash = ?
                """,
                (self.scope, content_hash),
            ).fetchone()
            if row:
                existing = _entry_from_row(row)
                memory_id = existing.id
                conn.execute(
                    """
                    update memories
                    set content = ?, kind = ?, source = ?, importance = ?, confidence = ?,
                        updated_at = ?, expires_at = ?
                    where id = ? and scope = ?
                    """,
                    (
                        cleaned,
                        normalized_kind,
                        normalized_source,
                        max(existing.importance, normalized_importance),
                        max(existing.confidence, normalized_confidence),
                        now,
                        normalized_expiry if normalized_expiry is not None else existing.expires_at,
                        memory_id,
                        self.scope,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    insert into memories(
                        scope, content, created_at, kind, source, importance, confidence,
                        updated_at, expires_at, access_count, content_hash
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        self.scope,
                        cleaned,
                        now,
                        normalized_kind,
                        normalized_source,
                        normalized_importance,
                        normalized_confidence,
                        now,
                        normalized_expiry,
                        content_hash,
                    ),
                )
                memory_id = int(cursor.lastrowid)
            self._enforce_quota(conn, protected_id=memory_id)
            self._store_vector(conn, memory_id, vector)
            return memory_id

    def update(
        self,
        memory_id: int,
        content: str,
        *,
        kind: str | None = None,
        source: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        expires_at: str | datetime | None = None,
    ) -> tuple[MemoryEntry, list[int]]:
        """Rewrite an existing memory in place.

        The id is preserved. When the new content hashes to another memory in the
        same scope, that duplicate is merged into this row (max importance and
        confidence, earliest created_at, summed access count) and removed.

        Returns the updated entry together with the merged memory ids.
        """

        normalized_id = _positive_int(memory_id, "memory_id")
        cleaned = _clean_content(content)
        if len(cleaned) > self.max_content_length:
            raise ValueError(f"memory content exceeds {self.max_content_length} characters")
        normalized_kind = _validate_metadata(kind, "kind") if kind is not None else None
        normalized_source = _validate_metadata(source, "source") if source is not None else None
        normalized_importance = (
            _validate_score(importance, "importance") if importance is not None else None
        )
        normalized_confidence = (
            _validate_score(confidence, "confidence") if confidence is not None else None
        )
        expiry_given = expires_at is not None
        normalized_expiry = _normalize_optional_timestamp(expires_at) if expiry_given else None

        content_hash = _hash_content(cleaned)
        now = _now_iso()

        with self._connect() as conn:
            conn.execute("begin immediate")
            self._purge_expired(conn)
            row = conn.execute(
                f"select {_ENTRY_COLUMNS} from memories where id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(f"memory {normalized_id} does not exist")
            existing = _entry_from_row(row)
            if existing.scope != self.scope:
                raise MemoryScopeError(
                    f"memory {normalized_id} belongs to a different project scope"
                )

            unchanged = (
                content_hash == existing.content_hash
                and normalized_kind is None
                and normalized_source is None
                and normalized_importance is None
                and normalized_confidence is None
                and not expiry_given
            )
            if unchanged:
                return existing, []

            final_kind = normalized_kind or existing.kind
            final_source = normalized_source or existing.source
            final_importance = (
                normalized_importance if normalized_importance is not None else existing.importance
            )
            final_confidence = (
                normalized_confidence if normalized_confidence is not None else existing.confidence
            )
            final_expiry = normalized_expiry if expiry_given else existing.expires_at
            created_at = existing.created_at
            access_count = existing.access_count
            merged: list[int] = []
            vector: list[float] | None = None
            if content_hash != existing.content_hash:
                vector = self._embed_one(cleaned)

            if content_hash != existing.content_hash:
                duplicate = conn.execute(
                    f"""
                    select {_ENTRY_COLUMNS} from memories
                    where scope = ? and content_hash = ? and id != ?
                    """,
                    (self.scope, content_hash, normalized_id),
                ).fetchone()
                if duplicate is not None:
                    other = _entry_from_row(duplicate)
                    final_importance = max(final_importance, other.importance)
                    final_confidence = max(final_confidence, other.confidence)
                    created_at = min(created_at, other.created_at)
                    access_count += other.access_count
                    final_expiry = _merge_expiry(final_expiry, other.expires_at)
                    conn.execute("delete from memories where id = ?", (other.id,))
                    merged.append(other.id)

            conn.execute(
                """
                update memories
                set content = ?, created_at = ?, kind = ?, source = ?, importance = ?,
                    confidence = ?, updated_at = ?, expires_at = ?, access_count = ?,
                    content_hash = ?
                where id = ? and scope = ?
                """,
                (
                    cleaned,
                    created_at,
                    final_kind,
                    final_source,
                    final_importance,
                    final_confidence,
                    now,
                    final_expiry,
                    access_count,
                    content_hash,
                    normalized_id,
                    self.scope,
                ),
            )
            self._enforce_quota(conn, protected_id=normalized_id)
            if vector is not None:
                self._store_vector(conn, normalized_id, vector)
            if merged:
                self._delete_vectors(conn, merged)
            updated_row = conn.execute(
                f"select {_ENTRY_COLUMNS} from memories where id = ?",
                (normalized_id,),
            ).fetchone()
        return _entry_from_row(updated_row), merged

    def get(self, memory_id: int) -> MemoryEntry | None:
        """Return one memory from the current scope, or ``None``."""

        normalized_id = _positive_int(memory_id, "memory_id")
        with self._connect() as conn:
            row = conn.execute(
                f"select {_ENTRY_COLUMNS} from memories where id = ? and scope = ?",
                (normalized_id, self.scope),
            ).fetchone()
        return _entry_from_row(row) if row is not None else None

    def list(self, limit: int = 20, *, kind: str | None = None) -> list[MemoryEntry]:
        normalized_limit = _normalize_limit(limit)
        if normalized_limit == 0:
            return []
        params: list[Any] = [self.scope, _now_iso()]
        kind_clause = ""
        if kind is not None:
            kind_clause = " and kind = ?"
            params.append(_validate_metadata(kind, "kind"))
        params.append(normalized_limit)
        with self._connect() as conn:
            self._purge_expired(conn)
            rows = conn.execute(
                f"""
                select {_ENTRY_COLUMNS}
                from memories
                where scope = ?
                  and (expires_at is null or expires_at > ?)
                  {kind_clause}
                order by updated_at desc, id desc
                limit ?
                """,
                params,
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search without mutating access counters; kept compatible with the original API."""

        return self.recall(query, limit=limit, mark_access=False)

    def recall(
        self,
        query: str,
        limit: int = 10,
        *,
        kinds: Iterable[str] | None = None,
        min_score: float = 0.0,
        mark_access: bool = True,
    ) -> list[MemoryEntry]:
        normalized_limit = _normalize_limit(limit)
        if normalized_limit == 0:
            return []
        cleaned_query = " ".join(str(query or "").split())
        if not cleaned_query:
            return self.list(normalized_limit)

        normalized_kinds = None
        if kinds is not None:
            normalized_kinds = {_validate_metadata(kind, "kind") for kind in kinds}
            if not normalized_kinds:
                return []

        keyword_entries = self._keyword_candidates(
            cleaned_query,
            normalized_kinds,
            min_score=min_score,
            depth=max(normalized_limit, self.candidate_depth),
        )
        vector_entries: list[MemoryEntry] = []
        if self.embedder is not None:
            try:
                vector_entries = self._vector_candidates(cleaned_query, normalized_kinds)
            except Exception:  # noqa: BLE001 - vectors are best effort
                vector_entries = []

        selected = self._fuse(keyword_entries, vector_entries, normalized_limit)
        if mark_access and selected:
            self.mark_access(entry.id for entry in selected)
            for entry in selected:
                entry.access_count += 1
        return selected

    def _keyword_candidates(
        self,
        query: str,
        kinds: set[str] | None,
        *,
        min_score: float,
        depth: int,
    ) -> list[MemoryEntry]:
        params: list[Any] = [self.scope, _now_iso()]
        kinds_clause = ""
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            kinds_clause = f" and kind in ({placeholders})"
            params.extend(sorted(kinds))

        with self._connect() as conn:
            self._purge_expired(conn)
            rows = conn.execute(
                f"""
                select {_ENTRY_COLUMNS}
                from memories
                where scope = ?
                  and (expires_at is null or expires_at > ?)
                  {kinds_clause}
                """,
                params,
            ).fetchall()

        now = datetime.now(UTC)
        ranked: list[tuple[float, MemoryEntry]] = []
        for row in rows:
            entry = _entry_from_row(row)
            score = _relevance_score(query, entry, now)
            if score > 0 and score >= float(min_score):
                ranked.append((score, entry))
        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].importance,
                item[1].confidence,
                item[1].updated_at,
                item[1].id,
            ),
            reverse=True,
        )
        return [entry for _score, entry in ranked[:depth]]

    def _vector_candidates(
        self,
        query: str,
        kinds: set[str] | None,
    ) -> list[MemoryEntry]:
        if self.embedder is None:
            return []
        vectors = self.embedder.embed([query])
        if not vectors:
            return []
        with self._connect() as conn:
            self._purge_expired(conn)
            index = self._vector_index(conn)
            if index is None:
                return []
            hits = index.search(
                vectors[0],
                limit=self.candidate_depth,
                min_similarity=self.embedding_min_similarity,
            )
            if not hits:
                return []
            ids = [memory_id for memory_id, _score in hits]
            placeholders = ", ".join("?" for _ in ids)
            rows = conn.execute(
                f"""
                select {_ENTRY_COLUMNS} from memories
                where scope = ?
                  and id in ({placeholders})
                  and (expires_at is null or expires_at > ?)
                """,
                [self.scope, *ids, _now_iso()],
            ).fetchall()

        by_id = {entry.id: entry for entry in (_entry_from_row(row) for row in rows)}
        entries = [by_id[memory_id] for memory_id in ids if memory_id in by_id]
        if kinds:
            entries = [entry for entry in entries if entry.kind in kinds]
        return entries

    def _fuse(
        self,
        keyword: list[MemoryEntry],
        vector: list[MemoryEntry],
        limit: int,
    ) -> list[MemoryEntry]:
        """Reciprocal Rank Fusion over the keyword and vector candidate lists."""

        if not vector:
            return keyword[:limit]
        by_id = {entry.id: entry for entry in keyword}
        by_id.update({entry.id: entry for entry in vector})
        scores: dict[int, float] = {}
        for rank, entry in enumerate(keyword, start=1):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, entry in enumerate(vector, start=1):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (self.rrf_k + rank)
        ordered = sorted(
            scores.items(),
            key=lambda item: (item[1], by_id[item[0]].importance, by_id[item[0]].updated_at),
            reverse=True,
        )
        return [by_id[memory_id] for memory_id, _score in ordered[:limit]]

    # ------------------------------------------------------------------
    # Vector index maintenance
    # ------------------------------------------------------------------

    def _embed_one(self, content: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            vectors = self.embedder.embed([content])
        except Exception:  # noqa: BLE001 - degrade to keyword-only recall
            self.vector_skip_reason = "embedding failed; using keyword recall only"
            return None
        return vectors[0] if vectors else None

    def _vector_index(self, conn: sqlite3.Connection) -> MemoryVectorIndex | None:
        if self.embedder is None:
            return None
        index = MemoryVectorIndex(conn, dim=self.embedder.dim, model=self.embedder.model_name)
        stored = index.meta()
        if stored is None:
            index.write_meta()
        elif stored != (index.model, index.dim):
            self.vector_skip_reason = (
                f"embedding model changed from {stored[0]} to {index.model}; "
                "run `taskpaw memory reindex`"
            )
            return None
        return index

    def _store_vector(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        vector: list[float] | None,
    ) -> None:
        if vector is None:
            return
        index = self._vector_index(conn)
        if index is None:
            return
        try:
            index.upsert(memory_id, vector)
            index.prune_orphans()
        except sqlite3.Error:
            self.vector_skip_reason = "failed to store memory vector"

    def _delete_vectors(self, conn: sqlite3.Connection, ids: Iterable[int]) -> None:
        if self.embedder is None:
            return
        index = self._vector_index(conn)
        if index is None:
            return
        with suppress(sqlite3.Error):
            index.delete(ids)

    def reindex(self) -> int:
        """Rebuild every vector in this scope with the current embedding model."""

        if self.embedder is None:
            raise ValueError("embeddings are not enabled")
        with self._connect() as conn:
            self._purge_expired(conn)
            reset_index(conn)
            index = MemoryVectorIndex(
                conn,
                dim=self.embedder.dim,
                model=self.embedder.model_name,
            )
            index.write_meta()
            rows = conn.execute(
                "select id, content from memories where scope = ? "
                "and (expires_at is null or expires_at > ?)",
                (self.scope, _now_iso()),
            ).fetchall()
            if not rows:
                return 0
            vectors = self.embedder.embed([str(row["content"]) for row in rows])
            for row, vector in zip(rows, vectors, strict=False):
                index.upsert(int(row["id"]), vector)
        return len(rows)

    def mark_access(self, memory_ids: int | Iterable[int]) -> int:
        ids = [memory_ids] if isinstance(memory_ids, int) else list(memory_ids)
        normalized_ids = list(dict.fromkeys(int(memory_id) for memory_id in ids))
        if not normalized_ids:
            return 0

        changed = 0
        with self._connect() as conn:
            for start in range(0, len(normalized_ids), 500):
                chunk = normalized_ids[start : start + 500]
                placeholders = ", ".join("?" for _ in chunk)
                cursor = conn.execute(
                    f"""
                    update memories
                    set access_count = access_count + 1
                    where scope = ? and id in ({placeholders})
                    """,
                    [self.scope, *chunk],
                )
                changed += int(cursor.rowcount)
        return changed

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            self._purge_expired(conn)
            row = conn.execute(
                """
                select count(*) as total,
                       coalesce(sum(access_count), 0) as total_access_count,
                       coalesce(avg(importance), 0) as average_importance,
                       coalesce(avg(confidence), 0) as average_confidence,
                       min(created_at) as oldest_created_at,
                       max(updated_at) as newest_updated_at
                from memories
                where scope = ?
                """,
                (self.scope,),
            ).fetchone()
            kind_rows = conn.execute(
                """
                select kind, count(*) as count
                from memories
                where scope = ?
                group by kind
                order by kind
                """,
                (self.scope,),
            ).fetchall()
        return {
            "scope": self.scope,
            "total": int(row["total"]),
            "by_kind": {str(item["kind"]): int(item["count"]) for item in kind_rows},
            "total_access_count": int(row["total_access_count"]),
            "average_importance": float(row["average_importance"]),
            "average_confidence": float(row["average_confidence"]),
            "oldest_created_at": row["oldest_created_at"],
            "newest_updated_at": row["newest_updated_at"],
            "max_entries": self.max_entries,
        }

    def delete(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "delete from memories where scope = ? and id = ?",
                (self.scope, int(memory_id)),
            )
            if not cursor.rowcount:
                return False
            self._delete_vectors(conn, [int(memory_id)])
            return True

    def clear(self) -> int:
        with self._connect() as conn:
            ids = [
                int(row[0])
                for row in conn.execute(
                    "select id from memories where scope = ?", (self.scope,)
                ).fetchall()
            ]
            cursor = conn.execute("delete from memories where scope = ?", (self.scope,))
            self._delete_vectors(conn, ids)
            return int(cursor.rowcount)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists memories (
                    id integer primary key autoincrement,
                    scope text not null,
                    content text not null,
                    created_at text not null,
                    kind text not null default 'fact',
                    source text not null default 'manual',
                    importance real not null default 0.5,
                    confidence real not null default 1.0,
                    updated_at text not null default '',
                    expires_at text,
                    access_count integer not null default 0,
                    content_hash text not null default ''
                )
                """
            )
            columns = {
                str(row["name"]) for row in conn.execute("pragma table_info(memories)").fetchall()
            }
            migrations = {
                "kind": "text not null default 'fact'",
                "source": "text not null default 'legacy'",
                "importance": "real not null default 0.5",
                "confidence": "real not null default 1.0",
                "updated_at": "text not null default ''",
                "expires_at": "text",
                "access_count": "integer not null default 0",
                "content_hash": "text not null default ''",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    conn.execute(f"alter table memories add column {name} {definition}")

            conn.execute("drop index if exists idx_memories_scope_hash")
            self._normalize_and_deduplicate(conn)
            conn.execute("create index if not exists idx_memories_scope on memories(scope, id)")
            conn.execute(
                """
                create unique index if not exists idx_memories_scope_hash
                on memories(scope, content_hash)
                """
            )
            conn.execute(
                """
                create index if not exists idx_memories_scope_updated
                on memories(scope, updated_at desc, id desc)
                """
            )
            conn.execute(
                """
                create index if not exists idx_memories_scope_kind
                on memories(scope, kind)
                """
            )
            self._enforce_quota(conn)

    def _normalize_and_deduplicate(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(f"select {_ENTRY_COLUMNS} from memories order by id").fetchall()
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        now = _now_iso()
        for row in rows:
            memory_id = int(row["id"])
            content = str(row["content"] or "").strip()
            if not content:
                conn.execute("delete from memories where id = ?", (memory_id,))
                continue
            scope = str(row["scope"] or "")
            created_at = _coerce_timestamp(row["created_at"], now)
            updated_at = _coerce_timestamp(row["updated_at"], created_at)
            state: dict[str, Any] = {
                "id": memory_id,
                "scope": scope,
                "content": content,
                "created_at": created_at,
                "kind": _coerce_metadata(row["kind"], "fact"),
                "source": _coerce_metadata(row["source"], "legacy"),
                "importance": _coerce_score(row["importance"], 0.5),
                "confidence": _coerce_score(row["confidence"], 1.0),
                "updated_at": updated_at,
                "expires_at": _coerce_optional_timestamp(row["expires_at"]),
                "access_count": max(int(row["access_count"] or 0), 0),
                "content_hash": _hash_content(content),
            }
            key = (scope, state["content_hash"])
            keeper = seen.get(key)
            if keeper is None:
                self._write_migrated_row(conn, state)
                seen[key] = state
                continue

            if state["updated_at"] >= keeper["updated_at"]:
                keeper["content"] = state["content"]
                keeper["kind"] = state["kind"]
                keeper["source"] = state["source"]
                keeper["updated_at"] = state["updated_at"]
            keeper["created_at"] = min(keeper["created_at"], state["created_at"])
            keeper["importance"] = max(keeper["importance"], state["importance"])
            keeper["confidence"] = max(keeper["confidence"], state["confidence"])
            keeper["expires_at"] = _merge_expiry(keeper["expires_at"], state["expires_at"])
            keeper["access_count"] += state["access_count"]
            self._write_migrated_row(conn, keeper)
            conn.execute("delete from memories where id = ?", (memory_id,))

    def _write_migrated_row(self, conn: sqlite3.Connection, state: dict[str, Any]) -> None:
        conn.execute(
            """
            update memories
            set scope = ?, content = ?, created_at = ?, kind = ?, source = ?,
                importance = ?, confidence = ?, updated_at = ?, expires_at = ?,
                access_count = ?, content_hash = ?
            where id = ?
            """,
            (
                state["scope"],
                state["content"],
                state["created_at"],
                state["kind"],
                state["source"],
                state["importance"],
                state["confidence"],
                state["updated_at"],
                state["expires_at"],
                state["access_count"],
                state["content_hash"],
                state["id"],
            ),
        )

    def _purge_expired(self, conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            """
            delete from memories
            where scope = ? and expires_at is not null and expires_at <= ?
            """,
            (self.scope, _now_iso()),
        )
        return int(cursor.rowcount)

    def _enforce_quota(
        self,
        conn: sqlite3.Connection,
        *,
        protected_id: int | None = None,
    ) -> int:
        self._purge_expired(conn)
        total = int(
            conn.execute("select count(*) from memories where scope = ?", (self.scope,)).fetchone()[
                0
            ]
        )
        excess = total - self.max_entries
        if excess <= 0:
            return 0
        protected_clause = ""
        params: list[Any] = [self.scope]
        if protected_id is not None:
            protected_clause = " and id != ?"
            params.append(protected_id)
        params.append(excess)
        cursor = conn.execute(
            f"""
            delete from memories
            where id in (
                select id from memories
                where scope = ? {protected_clause}
                order by importance asc, confidence asc, access_count asc,
                         updated_at asc, id asc
                limit ?
            )
            """,
            params,
        )
        return int(cursor.rowcount)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 30000")
        return conn


def _entry_from_row(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        id=int(row["id"]),
        scope=str(row["scope"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
        kind=str(row["kind"]),
        source=str(row["source"]),
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        updated_at=str(row["updated_at"]),
        expires_at=str(row["expires_at"]) if row["expires_at"] is not None else None,
        access_count=int(row["access_count"]),
        content_hash=str(row["content_hash"]),
    )


def _clean_content(content: str) -> str:
    if not isinstance(content, str):
        raise TypeError("memory content must be a string")
    cleaned = content.strip()
    if not cleaned:
        raise ValueError("memory content must not be empty")
    return cleaned


def _hash_content(content: str) -> str:
    normalized = _normalize_text(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _validate_metadata(value: str, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"memory {field} must not be empty")
    if len(normalized) > _MAX_METADATA_LENGTH:
        raise ValueError(f"memory {field} exceeds {_MAX_METADATA_LENGTH} characters")
    return normalized


def _coerce_metadata(value: Any, fallback: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > _MAX_METADATA_LENGTH:
        return fallback
    return normalized


def _validate_score(value: float, field: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"memory {field} must be a number between 0 and 1") from exc
    if not 0 <= normalized <= 1:
        raise ValueError(f"memory {field} must be between 0 and 1")
    return normalized


def _coerce_score(value: Any, fallback: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(max(normalized, 0.0), 1.0)


def _positive_int(value: int, field: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return normalized


def _normalize_limit(value: int) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_optional_timestamp(value: str | datetime | None) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ValueError("expires_at must be a valid ISO-8601 timestamp")
    return parsed.isoformat()


def _coerce_optional_timestamp(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    parsed = _parse_timestamp(str(value))
    return parsed.isoformat() if parsed else None


def _coerce_timestamp(value: Any, fallback: str) -> str:
    parsed = _parse_timestamp(str(value or ""))
    if parsed:
        return parsed.isoformat()
    fallback_parsed = _parse_timestamp(fallback)
    return fallback_parsed.isoformat() if fallback_parsed else _now_iso()


def _parse_timestamp(value: str | datetime) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _merge_expiry(left: str | None, right: str | None) -> str | None:
    if left is None or right is None:
        return None
    return max(left, right)


def _lexical_features(value: str) -> set[str]:
    normalized = _normalize_text(value)
    features = set(_WORD_RE.findall(normalized))
    for sequence in _CJK_RE.findall(normalized):
        characters = list(sequence)
        features.update(characters)
        features.update(
            "".join(characters[index : index + 2]) for index in range(len(characters) - 1)
        )
        if len(sequence) > 1:
            features.add(sequence)
    return features


def _relevance_score(query: str, entry: MemoryEntry, now: datetime) -> float:
    normalized_query = _normalize_text(query)
    normalized_content = _normalize_text(entry.content)
    query_features = _lexical_features(normalized_query)
    content_features = _lexical_features(normalized_content)
    if not query_features:
        return 0.0
    overlap = query_features & content_features
    substring_match = normalized_query in normalized_content
    if not overlap and not substring_match:
        return 0.0
    coverage = len(overlap) / len(query_features)
    lexical = min(1.0, 0.8 * coverage + (0.2 if substring_match else 0.0))
    updated_at = _parse_timestamp(entry.updated_at) or _parse_timestamp(entry.created_at) or now
    age_days = max((now - updated_at).total_seconds(), 0.0) / 86_400
    recency = 1.0 / (1.0 + age_days / 30.0)
    access = min(math.log1p(max(entry.access_count, 0)) / math.log(11), 1.0)
    return (
        0.72 * lexical
        + 0.12 * entry.importance
        + 0.08 * entry.confidence
        + 0.06 * recency
        + 0.02 * access
    )
