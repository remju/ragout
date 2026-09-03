"""Bulk embedding with caching and a progress bar.

The callers differ in what they are embedding - document chunks, code units -
but not in how it gets embedded, so that part lives here.
"""

from __future__ import annotations

import hashlib
import sys
from typing import List, Optional, Sequence

from llmcli.tools.cache import EmbeddingCache
from llmcli.tools.client import LLMClient
from llmcli.tools.progress import Progress
from llmcli.tools.vectors import normalize


def embed_texts(
    texts: Sequence[str],
    client: LLMClient,
    cache: Optional[EmbeddingCache] = None,
    progress: bool = True,
    label: str = "embedding",
) -> List[Sequence[float]]:
    """Return one L2-normalised vector per text, reusing anything cached."""
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    model = client.embed_model
    cached = cache.get_many(hashes, model) if cache else {}

    missing = [i for i, h in enumerate(hashes) if h not in cached]
    if cached and missing:
        print(f"{len(texts) - len(missing)} cached, {len(missing)} to embed", file=sys.stderr)

    if missing:
        def persist(offset: int, vecs: List[List[float]]) -> None:
            """Commit one batch immediately, so Ctrl-C keeps prior batches."""
            items = []
            for k, vec in enumerate(vecs):
                idx = missing[offset + k]
                norm_vec = normalize(vec)
                cached[hashes[idx]] = norm_vec
                items.append((hashes[idx], model, norm_vec))
            if cache:
                cache.put_many(items)  # put_many commits
            prog.advance(len(vecs))

        with Progress(label, len(missing), progress) as prog:
            vectors = client.embed([texts[i] for i in missing], on_batch=persist)

        # Fall back for clients that ignore on_batch (e.g. test doubles).
        leftover = []
        for idx, vec in zip(missing, vectors):
            if hashes[idx] not in cached:
                norm_vec = normalize(vec)
                cached[hashes[idx]] = norm_vec
                leftover.append((hashes[idx], model, norm_vec))
        if leftover and cache:
            cache.put_many(leftover)

    return [cached[h] for h in hashes]
