"""code_index.py — chunk-level code index with keyword + vector retrieval.

Python files are chunked by AST node; other text files fall back to line blocks.
Keyword search uses SQLite FTS5 when available (LIKE otherwise), and optional
embeddings add a vector list; the two are fused with RRF.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from taskpaw.memory.embedding import Embedder
from taskpaw.memory.vector_store import VectorIndex, reset_index
from taskpaw.rag.chunker import chunk_code

TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".sh",
}

SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", "target", "__pycache__"}
MAX_FILE_BYTES = 512 * 1024
INDEX_VERSION = "2"

_TERM_RE = re.compile(r"[0-9A-Za-z_]+")
_VEC_TABLE = "vec_code_chunks"
_BLOB_TABLE = "code_vectors_blob"
_META_TABLE = "code_vector_meta"


@dataclass(slots=True)
class CodeSearchResult:
    path: str
    line: int
    snippet: str
    end_line: int = 0
    name: str = ""
    kind: str = ""


class CodeIndex:
    def __init__(
        self,
        root: str | Path,
        db_path: str | Path | None = None,
        *,
        embedder: Embedder | None = None,
        min_similarity: float = 0.25,
        candidate_depth: int = 30,
        rrf_k: int = 60,
        max_file_bytes: int = MAX_FILE_BYTES,
    ):
        self.root = Path(root).resolve()
        self.db_path = (
            Path(db_path).expanduser() if db_path else self.root / ".taskpaw" / "code_index.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.min_similarity = float(min_similarity)
        self.candidate_depth = max(1, int(candidate_depth))
        self.rrf_k = max(1, int(rrf_k))
        self.max_file_bytes = max(1024, int(max_file_bytes))
        self.vector_skip_reason = ""
        self.needs_full_rebuild = False
        self.fts_enabled = False
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def rebuild(self, path: str | Path | None = None) -> int:
        base = self._resolve(path or self.root)
        files = [base] if base.is_file() else list(self._iter_files(base))
        with self._connect() as conn:
            if base == self.root:
                self._clear_all(conn)
            else:
                for file_path in files:
                    self._delete_file(conn, str(file_path.relative_to(self.root)))
            count = 0
            for file_path in files:
                count += self._index_file(conn, file_path, force=True)
            self._prune_orphans(conn)
            self._write_meta(conn)
        self.needs_full_rebuild = False
        return count

    def update(self, path: str | Path | None = None) -> int:
        """Incremental refresh: reindex changed files and drop deleted ones."""

        if self.needs_full_rebuild:
            return self.rebuild(path)
        base = self._resolve(path or self.root)
        files = [base] if base.is_file() else list(self._iter_files(base))
        seen: set[str] = set()
        count = 0
        with self._connect() as conn:
            for file_path in files:
                seen.add(str(file_path.relative_to(self.root)))
                count += self._index_file(conn, file_path)
            if base == self.root:
                for row in conn.execute(
                    "select path from code_files where root = ?", (str(self.root),)
                ).fetchall():
                    if str(row[0]) not in seen:
                        self._delete_file(conn, str(row[0]))
            self._prune_orphans(conn)
        return count

    def _index_file(self, conn: sqlite3.Connection, file_path: Path, *, force: bool = False) -> int:
        try:
            raw = file_path.read_bytes()
        except OSError:
            return 0
        if len(raw) > self.max_file_bytes:
            return 0
        rel = str(file_path.relative_to(self.root))
        digest = hashlib.sha256(raw).hexdigest()
        row = conn.execute(
            "select content_hash from code_files where root = ? and path = ?",
            (str(self.root), rel),
        ).fetchone()
        if row and str(row[0]) == digest and not force:
            return 0

        self._delete_file(conn, rel)
        text = raw.decode("utf-8", errors="ignore")
        chunks = chunk_code(rel, text)
        if not chunks:
            self._record_file(conn, rel, digest, file_path)
            return 0

        vectors = self._embed([chunk.content for chunk in chunks])
        index = self._vector_index(conn) if self.embedder is not None else None
        for offset, chunk in enumerate(chunks):
            cursor = conn.execute(
                """
                insert into code_chunks_v2
                    (root, path, start_line, end_line, kind, name, content)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(self.root),
                    chunk.path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.kind,
                    chunk.name,
                    chunk.content,
                ),
            )
            chunk_id = int(cursor.lastrowid or 0)
            if self.fts_enabled:
                conn.execute(
                    "insert into code_fts(chunk_id, content) values (?, ?)",
                    (chunk_id, chunk.content),
                )
            if index is not None and vectors is not None and offset < len(vectors):
                try:
                    index.upsert(chunk_id, vectors[offset])
                except sqlite3.Error:
                    self.vector_skip_reason = "failed to store code vector"
        self._record_file(conn, rel, digest, file_path)
        return len(chunks)

    def _record_file(
        self,
        conn: sqlite3.Connection,
        rel: str,
        digest: str,
        file_path: Path,
    ) -> None:
        stat = file_path.stat()
        conn.execute(
            """
            insert or replace into code_files
                (root, path, content_hash, mtime, size, indexed_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                str(self.root),
                rel,
                digest,
                float(stat.st_mtime),
                int(stat.st_size),
                datetime.now(UTC).isoformat(),
            ),
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        if self.embedder is None:
            return None
        try:
            return self.embedder.embed(texts)
        except Exception:  # noqa: BLE001 - vectors are best effort
            self.vector_skip_reason = "embedding failed; code search falls back to keywords"
            return None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[CodeSearchResult]:
        cleaned = " ".join(str(query or "").split())
        if not cleaned:
            return []
        max_results = max(1, min(int(limit), 50))
        depth = max(max_results, self.candidate_depth)
        with self._connect() as conn:
            keyword = self._keyword_candidates(conn, cleaned, depth)
            vector = self._vector_candidates(conn, cleaned)
        return self._fuse(keyword, vector, max_results)

    def _keyword_candidates(
        self,
        conn: sqlite3.Connection,
        query: str,
        depth: int,
    ) -> list[tuple[int, CodeSearchResult]]:
        terms = [term for term in _TERM_RE.findall(query) if len(term) >= 2]
        if self.fts_enabled and terms:
            match = " ".join(f'"{term}"' for term in terms)
            rows = conn.execute(
                """
                select f.chunk_id
                from code_fts f
                join code_chunks_v2 c on c.id = f.chunk_id
                where code_fts match ? and c.root = ?
                order by bm25(code_fts)
                limit ?
                """,
                (match, str(self.root), depth),
            ).fetchall()
            ids = [int(row[0]) for row in rows]
            if ids:
                return self._load_results(conn, ids)

        patterns = [f"%{term.lower()}%" for term in (terms or [query.lower()])]
        clauses = " and ".join("lower(content) like ?" for _ in patterns)
        rows = conn.execute(
            f"""
            select id from code_chunks_v2
            where root = ? and {clauses}
            order by path, start_line
            limit ?
            """,
            [str(self.root), *patterns, depth],
        ).fetchall()
        return self._load_results(conn, [int(row[0]) for row in rows])

    def _vector_candidates(
        self,
        conn: sqlite3.Connection,
        query: str,
    ) -> list[tuple[int, CodeSearchResult]]:
        if self.embedder is None:
            return []
        index = self._vector_index(conn)
        if index is None:
            return []
        vectors = self._embed([query])
        if not vectors:
            return []
        hits = index.search(
            vectors[0],
            limit=self.candidate_depth,
            min_similarity=self.min_similarity,
        )
        ids = [chunk_id for chunk_id, _score in hits]
        return self._load_results(conn, ids)

    def _load_results(
        self,
        conn: sqlite3.Connection,
        ids: list[int],
    ) -> list[tuple[int, CodeSearchResult]]:
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            select id, path, start_line, end_line, kind, name, content
            from code_chunks_v2
            where root = ? and id in ({placeholders})
            """,
            [str(self.root), *ids],
        ).fetchall()
        by_id = {
            int(row[0]): CodeSearchResult(
                path=str(row[1]),
                line=int(row[2]),
                snippet=str(row[6])[:2000],
                end_line=int(row[3]),
                name=str(row[5] or ""),
                kind=str(row[4] or ""),
            )
            for row in rows
        }
        return [(chunk_id, by_id[chunk_id]) for chunk_id in ids if chunk_id in by_id]

    def _fuse(
        self,
        keyword: list[tuple[int, CodeSearchResult]],
        vector: list[tuple[int, CodeSearchResult]],
        limit: int,
    ) -> list[CodeSearchResult]:
        if not vector:
            return [result for _chunk_id, result in keyword[:limit]]
        scores: dict[int, float] = {}
        results: dict[int, CodeSearchResult] = {}
        for rank, (chunk_id, result) in enumerate(keyword, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            results[chunk_id] = result
        for rank, (chunk_id, result) in enumerate(vector, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            results.setdefault(chunk_id, result)
        ordered = sorted(
            scores.items(),
            key=lambda item: (item[1], -results[item[0]].line),
            reverse=True,
        )
        return [results[chunk_id] for chunk_id, _score in ordered[:limit]]

    # ------------------------------------------------------------------
    # Schema and helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("drop table if exists code_chunks")
            conn.execute(
                """
                create table if not exists code_files (
                    root text not null,
                    path text not null,
                    content_hash text not null,
                    mtime real not null,
                    size integer not null,
                    indexed_at text not null,
                    primary key (root, path)
                )
                """
            )
            conn.execute(
                """
                create table if not exists code_chunks_v2 (
                    id integer primary key autoincrement,
                    root text not null,
                    path text not null,
                    start_line integer not null,
                    end_line integer not null,
                    kind text not null,
                    name text not null,
                    content text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_code_chunks_v2_root_path "
                "on code_chunks_v2(root, path)"
            )
            conn.execute(
                "create table if not exists code_meta (key text primary key, value text not null)"
            )
            try:
                conn.execute(
                    "create virtual table if not exists code_fts "
                    "using fts5(chunk_id UNINDEXED, content, tokenize='unicode61')"
                )
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False

            version_row = conn.execute(
                "select value from code_meta where key = 'index_version'"
            ).fetchone()
            self.needs_full_rebuild = version_row is None or str(version_row[0]) != INDEX_VERSION

    def _write_meta(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "insert or replace into code_meta(key, value) values ('index_version', ?)",
            (INDEX_VERSION,),
        )

    def _delete_file(self, conn: sqlite3.Connection, rel: str) -> None:
        rows = conn.execute(
            "select id from code_chunks_v2 where root = ? and path = ?",
            (str(self.root), rel),
        ).fetchall()
        ids = [int(row[0]) for row in rows]
        if ids:
            placeholders = ", ".join("?" for _ in ids)
            conn.execute(f"delete from code_chunks_v2 where id in ({placeholders})", ids)
            if self.fts_enabled:
                conn.execute(f"delete from code_fts where chunk_id in ({placeholders})", ids)
            index = self._vector_index(conn)
            if index is not None:
                index.delete(ids)
        conn.execute("delete from code_files where root = ? and path = ?", (str(self.root), rel))

    def _clear_all(self, conn: sqlite3.Connection) -> None:
        conn.execute("delete from code_chunks_v2 where root = ?", (str(self.root),))
        conn.execute("delete from code_files where root = ?", (str(self.root),))
        if self.fts_enabled:
            conn.execute("delete from code_fts")
        self._reset_vectors(conn)

    def _prune_orphans(self, conn: sqlite3.Connection) -> None:
        index = self._vector_index(conn)
        if index is not None:
            index.prune_orphans()

    def _vector_index(self, conn: sqlite3.Connection) -> VectorIndex | None:
        if self.embedder is None:
            return None
        index = VectorIndex(
            conn,
            dim=self.embedder.dim,
            model=self.embedder.model_name,
            vec_table=_VEC_TABLE,
            blob_table=_BLOB_TABLE,
            meta_table=_META_TABLE,
            id_column="chunk_id",
            orphan_parent="code_chunks_v2",
        )
        stored = index.meta()
        if stored is None:
            index.write_meta()
        elif stored != (index.model, index.dim):
            self.vector_skip_reason = (
                f"embedding model changed from {stored[0]} to {index.model}; rebuild the code index"
            )
            return None
        return index

    def _reset_vectors(self, conn: sqlite3.Connection) -> None:
        reset_index(
            conn,
            vec_table=_VEC_TABLE,
            blob_table=_BLOB_TABLE,
            meta_table=_META_TABLE,
        )

    def _iter_files(self, base: Path):
        for path in base.rglob("*"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        resolved.relative_to(self.root)
        return resolved

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)
