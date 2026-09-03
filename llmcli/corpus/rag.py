"""Turning retrieved chunks into a grounded prompt."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from llmcli.corpus.store import Store
from llmcli.tools.client import LLMClient

RAG_SYSTEM = (
    "You answer questions using the provided context excerpts. "
    "Cite the source filenames you used. If the context does not contain the "
    "answer, say so plainly instead of guessing."
)


def build_context(hits: Sequence[Tuple[float, str, str]], budget: int = 6000) -> str:
    blocks, used = [], 0
    for score, path, text in hits:
        block = f"[source: {Path(path).name} | relevance {score:.2f}]\n{text}"
        if used + len(block) > budget:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def retrieve(client: LLMClient, store: Store, question: str, k: int, min_score: float):
    vec = client.embed([question])[0]
    return store.search(vec, k=k, min_score=min_score)


def rag_messages(
    question: str,
    context: str,
    history: Optional[List[dict]] = None,
    system: str = RAG_SYSTEM,
) -> List[dict]:
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history)
    user = (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context
        else f"(no matching context found in the local store)\n\nQuestion: {question}"
    )
    msgs.append({"role": "user", "content": user})
    return msgs
