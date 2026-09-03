"""Content-hash embedding cache, shared by anything that embeds in bulk.

Keyed by (sha256 of the text, model), so re-running after small edits only
re-embeds what changed and switching embedding models never mixes vector
spaces.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from llmcli.tools.vectors import pack, unpack

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    hash   TEXT NOT NULL,
    model  TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (hash, model)
);
"""


class EmbeddingCache:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(CACHE_SCHEMA)

    def get_many(self, hashes: List[str], model: str) -> Dict[str, Sequence[float]]:
        out: Dict[str, Sequence[float]] = {}
        for i in range(0, len(hashes), 500):
            window = hashes[i : i + 500]
            qmarks = ",".join("?" * len(window))
            rows = self.conn.execute(
                f"SELECT hash, vector FROM embeddings WHERE model = ? AND hash IN ({qmarks})",
                [model, *window],
            )
            for h, blob in rows:
                out[h] = unpack(blob)
        return out

    def put_many(self, items: List[Tuple[str, str, Sequence[float]]]) -> None:
        if not items:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (hash, model, vector) VALUES (?, ?, ?)",
            [(h, model, pack(vec)) for h, model, vec in items],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
