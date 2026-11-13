"""vector_store.py — sqlite-vec backed memory vectors with a Python fallback.

Vectors are always normalised before being stored, so the L2 distance returned by
sqlite-vec can be converted back to cosine similarity. When the extension is not
available the same interface falls back to a blob table plus a Python cosine scan,
which is fine for the per-project memory cap (1k entries by default).
"""

from __future__ import annotations

import math
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from contextlib import suppress

META_TABLE = "memory_vector_meta"
VEC_TABLE = "vec_memories"
BLOB_TABLE = "memory_vectors_blob"


def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Try to load the sqlite-vec extension into *conn*."""

    try:
        import sqlite_vec
    except Exception:  # noqa: BLE001 - optional dependency
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except Exception:  # noqa: BLE001 - unsupported build/platform
        return False
    finally:
        with suppress(Exception):
            conn.enable_load_extension(False)


class MemoryVectorIndex:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        dim: int,
        model: str,
    ) -> None:
        self.conn = conn
        self.dim = max(1, int(dim))
        self.model = str(model)
        self.kind = "vec0" if load_sqlite_vec(conn) else "blob"
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema and metadata
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self.conn.execute(
            f"create table if not exists {META_TABLE}(key text primary key, value text not null)"
        )
        if self.kind == "vec0":
            self.conn.execute(
                f"create virtual table if not exists {VEC_TABLE} "
                f"using vec0(memory_id integer primary key, embedding float[{self.dim}])"
            )
        else:
            self.conn.execute(
                f"create table if not exists {BLOB_TABLE}("
                "memory_id integer primary key, dim integer not null, vector blob not null)"
            )

    def meta(self) -> tuple[str, int] | None:
        row = self.conn.execute(
            f"select key, value from {META_TABLE} where key in ('model', 'dim')"
        ).fetchall()
        values = {str(key): str(value) for key, value in row}
        if "model" not in values or "dim" not in values:
            return None
        try:
            return values["model"], int(values["dim"])
        except ValueError:
            return None

    def write_meta(self) -> None:
        self.conn.executemany(
            f"insert or replace into {META_TABLE}(key, value) values (?, ?)",
            (("model", self.model), ("dim", str(self.dim))),
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(self, memory_id: int, vector: Sequence[float]) -> None:
        normalized = _normalize(vector)
        if not normalized:
            return
        packed = _pack(normalized, self.dim)
        if self.kind == "vec0":
            self.conn.execute(f"delete from {VEC_TABLE} where memory_id = ?", (int(memory_id),))
            self.conn.execute(
                f"insert into {VEC_TABLE}(memory_id, embedding) values (?, ?)",
                (int(memory_id), packed),
            )
        else:
            self.conn.execute(
                f"insert or replace into {BLOB_TABLE}(memory_id, dim, vector) values (?, ?, ?)",
                (int(memory_id), self.dim, packed),
            )

    def delete(self, memory_ids: Iterable[int]) -> int:
        ids = [int(memory_id) for memory_id in memory_ids]
        if not ids:
            return 0
        table = VEC_TABLE if self.kind == "vec0" else BLOB_TABLE
        removed = 0
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            cursor = self.conn.execute(
                f"delete from {table} where memory_id in ({placeholders})", chunk
            )
            removed += int(cursor.rowcount)
        return removed

    def clear(self) -> None:
        for table in (VEC_TABLE, BLOB_TABLE):
            try:
                self.conn.execute(f"delete from {table}")
            except sqlite3.Error:
                continue

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        min_similarity: float = 0.0,
    ) -> list[tuple[int, float]]:
        query = _normalize(vector)
        if not query:
            return []
        hits: list[tuple[int, float]] = []
        if self.kind == "vec0":
            packed = _pack(query, self.dim)
            rows = self.conn.execute(
                f"select memory_id, distance from {VEC_TABLE} "
                f"where embedding match ? and k = {max(1, int(limit))}",
                (packed,),
            ).fetchall()
            for memory_id, distance in rows:
                similarity = 1.0 - (float(distance) ** 2) / 2.0
                hits.append((int(memory_id), max(-1.0, min(1.0, similarity))))
        else:
            rows = self.conn.execute(f"select memory_id, vector from {BLOB_TABLE}").fetchall()
            for memory_id, blob in rows:
                similarity = _cosine(query, _unpack(blob, self.dim))
                hits.append((int(memory_id), similarity))

        filtered = [item for item in hits if item[1] >= min_similarity]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered[: max(1, int(limit))]

    def count(self) -> int:
        table = VEC_TABLE if self.kind == "vec0" else BLOB_TABLE
        try:
            row = self.conn.execute(f"select count(*) from {table}").fetchone()
        except sqlite3.Error:
            return 0
        return int(row[0]) if row else 0

    def prune_orphans(self) -> None:
        """Drop vectors whose memory row no longer exists (quota eviction)."""

        table = VEC_TABLE if self.kind == "vec0" else BLOB_TABLE
        try:
            self.conn.execute(
                f"delete from {table} where memory_id not in (select id from memories)"
            )
        except sqlite3.Error:
            return


def reset_index(conn: sqlite3.Connection) -> None:
    """Drop all vector tables so the next index can be rebuilt from scratch."""

    for statement in (
        f"drop table if exists {VEC_TABLE}",
        f"drop table if exists {BLOB_TABLE}",
        f"drop table if exists {META_TABLE}",
    ):
        try:
            conn.execute(statement)
        except sqlite3.Error:
            continue


def _pack(vector: Sequence[float], dim: int) -> bytes:
    values = list(vector)[:dim]
    if len(values) < dim:
        values.extend([0.0] * (dim - len(values)))
    return struct.pack(f"<{dim}f", *values)


def _unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return []
    return [value / norm for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=False)))
