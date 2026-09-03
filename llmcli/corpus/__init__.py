"""Documents: read them, chunk them, embed them, retrieve them.

Owns everything about a body of documents - the file formats it can read, how
text is split, the per-project sqlite store, and turning retrieved chunks into
a grounded prompt. It asks llmcli.tools to do the embedding.
"""

from llmcli.corpus.documents import TEXT_SUFFIXES, chunk_text, iter_files, read_document
from llmcli.corpus.rag import RAG_SYSTEM, build_context, rag_messages, retrieve
from llmcli.corpus.store import Store

__all__ = [
    "TEXT_SUFFIXES", "read_document", "iter_files", "chunk_text",
    "Store", "RAG_SYSTEM", "build_context", "retrieve", "rag_messages",
]
