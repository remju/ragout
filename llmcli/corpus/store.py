"""The per-project embedding store: chunks and their vectors in sqlite."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import List, Sequence

from llmcli.tools.vectors import dot, normalize, pack, unpack

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id       INTEGER PRIMARY KEY,
    path     TEXT UNIQUE NOT NULL,
    sha256   TEXT NOT NULL,
    chunks   INTEGER NOT NULL,
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id     INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord    INTEGER NOT NULL,
    text   TEXT NOT NULL,
    vec    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
"""


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

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

    def replace_document(
        self, path: Path, sha: str, chunks: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        key = str(path.resolve())
        cur = self.conn.cursor()
        cur.execute("DELETE FROM documents WHERE path = ?", (key,))
        cur.execute(
            "INSERT INTO documents (path, sha256, chunks, added_at) VALUES (?, ?, ?, ?)",
            (key, sha, len(chunks), time.time()),
        )
        doc_id = cur.lastrowid
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
        cur = self.conn.execute("DELETE FROM documents WHERE path LIKE ?", (f"%{pattern}%",))
        self.conn.commit()
        return cur.rowcount

    def search(self, query_vec: Sequence[float], k: int = 4, min_score: float = 0.0):
        qv = normalize(query_vec)
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
