"""The per-project embedding store: chunks and their vectors.

Vectors go in a sqlite-vec `vec0` virtual table when the sqlite-vec extension
is available, so KNN search runs in C over the whole table instead of
unpacking every row into a python float32 array and scoring it by hand. Falls
back to a plain BLOB column and brute-force python scoring when it is not -
same fallback philosophy as `tools/vectors.py`'s numpy/stdlib split.

An existing store keeps whatever schema it was created with: mode is decided
once, at creation, from whether sqlite-vec loaded at the time, and is never
switched under an existing store.db. Re-run `llmcli ingest --force` after
installing sqlite-vec to move a project onto the faster backend.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import List, Sequence

from llmcli.tools.vectors import dot, normalize, pack, unpack

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id       INTEGER PRIMARY KEY,
    path     TEXT UNIQUE NOT NULL,
    sha256   TEXT NOT NULL,
    chunks   INTEGER NOT NULL,
    added_at REAL NOT NULL
);
"""

LEGACY_CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id     INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord    INTEGER NOT NULL,
    text   TEXT NOT NULL,
    vec    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
"""

VEC_CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id     INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord    INTEGER NOT NULL,
    text   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Best-effort extension load - absent package, a build with extension
    loading disabled, and a sandboxed sqlite3 all fail differently, so this
    is a capability probe rather than something with one expected exception."""
    if sqlite_vec is None:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(DOCUMENTS_SCHEMA)

        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if existing:
            cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks)")}
            self.vec_mode = "vec" not in cols
            if self.vec_mode and not _load_sqlite_vec(self.conn):
                raise RuntimeError(
                    f"{path} was built with sqlite-vec but the extension isn't available "
                    f"now - install it with `pip install 'llmcli[corpus]'`"
                )
        else:
            self.vec_mode = _load_sqlite_vec(self.conn)

        self.conn.executescript(VEC_CHUNKS_SCHEMA if self.vec_mode else LEGACY_CHUNKS_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def digest(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def is_current(self, path: Path, sha: str) -> bool:
        row = self.conn.execute(
            "SELECT sha256 FROM documents WHERE path = ?", (str(path.resolve()),)
        ).fetchone()
        return bool(row and row["sha256"] == sha)

    def _vec_table_exists(self) -> bool:
        return self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone() is not None

    def _ensure_vec_table(self, dim: int) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
        if row is None:
            self.conn.execute(f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{dim}])")
            self.conn.execute("INSERT INTO meta (key, value) VALUES ('dim', ?)", (str(dim),))
            return
        stored = int(row["value"])
        if stored != dim:
            raise RuntimeError(
                f"this store holds {stored}-dimensional vectors but the current embedding "
                f"model produced {dim} - switching embedding models needs a fresh store "
                f"(delete store.db and re-run `llmcli ingest`)"
            )

    def _delete_vectors_for(self, doc_ids: Sequence[int]) -> None:
        if not self.vec_mode or not doc_ids or not self._vec_table_exists():
            return
        qmarks = ",".join("?" * len(doc_ids))
        self.conn.execute(
            f"DELETE FROM vec_chunks WHERE rowid IN "
            f"(SELECT id FROM chunks WHERE doc_id IN ({qmarks}))",
            list(doc_ids),
        )

    def replace_document(
        self, path: Path, sha: str, chunks: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        key = str(path.resolve())
        cur = self.conn.cursor()
        old = cur.execute("SELECT id FROM documents WHERE path = ?", (key,)).fetchone()
        if old:
            self._delete_vectors_for([old["id"]])
        cur.execute("DELETE FROM documents WHERE path = ?", (key,))
        cur.execute(
            "INSERT INTO documents (path, sha256, chunks, added_at) VALUES (?, ?, ?, ?)",
            (key, sha, len(chunks), time.time()),
        )
        doc_id = cur.lastrowid

        if self.vec_mode:
            if vectors:
                self._ensure_vec_table(len(vectors[0]))
            for i, (text, vec) in enumerate(zip(chunks, vectors)):
                cur.execute(
                    "INSERT INTO chunks (doc_id, ord, text) VALUES (?, ?, ?)", (doc_id, i, text)
                )
                cur.execute(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(normalize(vec))),
                )
        else:
            cur.executemany(
                "INSERT INTO chunks (doc_id, ord, text, vec) VALUES (?, ?, ?, ?)",
                [
                    (doc_id, i, text, pack(normalize(vec)))
                    for i, (text, vec) in enumerate(zip(chunks, vectors))
                ],
            )
        self.conn.commit()

    def documents(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT path, chunks, added_at FROM documents ORDER BY path"
        ).fetchall()

    def forget(self, pattern: str) -> int:
        doc_ids = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM documents WHERE path LIKE ?", (f"%{pattern}%",)
            )
        ]
        self._delete_vectors_for(doc_ids)
        cur = self.conn.execute("DELETE FROM documents WHERE path LIKE ?", (f"%{pattern}%",))
        self.conn.commit()
        return cur.rowcount

    def search(self, query_vec: Sequence[float], k: int = 4, min_score: float = 0.0):
        qv = normalize(query_vec)
        if self.vec_mode:
            if not self._vec_table_exists():
                return []
            rows = self.conn.execute(
                """
                SELECT c.text, d.path, v.distance
                FROM vec_chunks v
                JOIN chunks c ON c.id = v.rowid
                JOIN documents d ON d.id = c.doc_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (sqlite_vec.serialize_float32(qv), k),
            ).fetchall()
            scored = []
            for row in rows:
                # both sides are unit vectors, so L2 distance and cosine
                # similarity are a direct trade: score = 1 - distance^2 / 2
                score = 1.0 - (row["distance"] ** 2) / 2.0
                if score >= min_score:
                    scored.append((score, row["path"], row["text"]))
            return scored

        rows = self.conn.execute(
            "SELECT c.text, c.vec, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id"
        ).fetchall()
        scored = []
        for row in rows:
            score = dot(qv, unpack(row["vec"]))
            if score >= min_score:
                scored.append((score, row["path"], row["text"]))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:k]
