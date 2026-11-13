"""vector_store.py — sqlite-vec backed vector tables with a Python fallback.

Vectors are always normalised before being stored, so the L2 distance returned by
sqlite-vec can be converted back to cosine similarity. When the extension is not
available the same interface falls back to a blob table plus a Python cosine scan.

The class is generic over table/column names: long-term memory uses the default
``memory_*`` tables while the code index passes its own ``code_*`` tables.
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


class VectorIndex:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        dim: int,
        model: str,
        vec_table: str = VEC_TABLE,
        blob_table: str = BLOB_TABLE,
        meta_table: str = META_TABLE,
        id_column: str = "memory_id",
        orphan_parent: str | None = None,
    ) -> None:
        self.conn = conn
        self.dim = max(1, int(dim))
        self.model = str(model)
        self.vec_table = vec_table
        self.blob_table = blob_table
        self.meta_table = meta_table
        self.id_column = id_column
        self.orphan_parent = orphan_parent
        self.kind = "vec0" if load_sqlite_vec(conn) else "blob"
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema and metadata
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self.conn.execute(
            f"create table if not exists {self.meta_table}"
            "(key text primary key, value text not null)"
        )
        if self.kind == "vec0":
            self.conn.execute(
                f"create virtual table if not exists {self.vec_table} "
                f"using vec0({self.id_column} integer primary key, embedding float[{self.dim}])"
            )
        else:
            self.conn.execute(
                f"create table if not exists {self.blob_table}("
                f"{self.id_column} integer primary key, dim integer not null, "
                "vector blob not null)"
            )

    def meta(self) -> tuple[str, int] | None:
        row = self.conn.execute(
            f"select key, value from {self.meta_table} where key in ('model', 'dim')"
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
            f"insert or replace into {self.meta_table}(key, value) values (?, ?)",
            (("model", self.model), ("dim", str(self.dim))),
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(self, vector_id: int, vector: Sequence[float]) -> None:
        normalized = _normalize(vector)
        if not normalized:
            return
        packed = _pack(normalized, self.dim)
        if self.kind == "vec0":
            self.conn.execute(
                f"delete from {self.vec_table} where {self.id_column} = ?", (int(vector_id),)
            )
            self.conn.execute(
                f"insert into {self.vec_table}({self.id_column}, embedding) values (?, ?)",
                (int(vector_id), packed),
            )
        else:
            self.conn.execute(
                f"insert or replace into {self.blob_table}"
                f"({self.id_column}, dim, vector) values (?, ?, ?)",
                (int(vector_id), self.dim, packed),
            )

    def delete(self, vector_ids: Iterable[int]) -> int:
        ids = [int(vector_id) for vector_id in vector_ids]
        if not ids:
            return 0
        table = self.vec_table if self.kind == "vec0" else self.blob_table
        removed = 0
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            cursor = self.conn.execute(
                f"delete from {table} where {self.id_column} in ({placeholders})", chunk
            )
            removed += int(cursor.rowcount)
        return removed

    def clear(self) -> None:
        for table in (self.vec_table, self.blob_table):
            with suppress(sqlite3.Error):
                self.conn.execute(f"delete from {table}")

    def prune_orphans(self) -> None:
        """Drop vectors whose parent row no longer exists (quota/rebuild cleanup)."""

        if not self.orphan_parent:
            return
        table = self.vec_table if self.kind == "vec0" else self.blob_table
        with suppress(sqlite3.Error):
            self.conn.execute(
                f"delete from {table} where {self.id_column} not in "
                f"(select id from {self.orphan_parent})"
            )

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
                f"select {self.id_column}, distance from {self.vec_table} "
                f"where embedding match ? and k = {max(1, int(limit))}",
                (packed,),
            ).fetchall()
            for vector_id, distance in rows:
                similarity = 1.0 - (float(distance) ** 2) / 2.0
                hits.append((int(vector_id), max(-1.0, min(1.0, similarity))))
        else:
            rows = self.conn.execute(
                f"select {self.id_column}, vector from {self.blob_table}"
            ).fetchall()
            for vector_id, blob in rows:
                similarity = _cosine(query, _unpack(blob, self.dim))
                hits.append((int(vector_id), similarity))

        filtered = [item for item in hits if item[1] >= min_similarity]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered[: max(1, int(limit))]

    def count(self) -> int:
        table = self.vec_table if self.kind == "vec0" else self.blob_table
        try:
            row = self.conn.execute(f"select count(*) from {table}").fetchone()
        except sqlite3.Error:
            return 0
        return int(row[0]) if row else 0


class MemoryVectorIndex(VectorIndex):
    """Long-term memory index (kept as a named subclass for compatibility)."""

    def __init__(self, conn: sqlite3.Connection, *, dim: int, model: str) -> None:
        super().__init__(
            conn,
            dim=dim,
            model=model,
            vec_table=VEC_TABLE,
            blob_table=BLOB_TABLE,
            meta_table=META_TABLE,
            id_column="memory_id",
            orphan_parent="memories",
        )


def reset_index(
    conn: sqlite3.Connection,
    *,
    vec_table: str = VEC_TABLE,
    blob_table: str = BLOB_TABLE,
    meta_table: str = META_TABLE,
) -> None:
    """Drop vector tables so the next index can be rebuilt from scratch."""

    for statement in (
        f"drop table if exists {vec_table}",
        f"drop table if exists {blob_table}",
        f"drop table if exists {meta_table}",
    ):
        with suppress(sqlite3.Error):
            conn.execute(statement)


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
