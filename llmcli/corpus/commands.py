"""Commands over the document corpus: ingest, docs, forget, ask."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from llmcli.corpus.documents import chunk_text, iter_files, read_document
from llmcli.corpus.rag import build_context, rag_messages, retrieve
from llmcli.corpus.store import Store
from llmcli.tools.args import Parents
from llmcli.tools.client import LLMClient
from llmcli.tools.workspace import Config


def cmd_ingest(args, cfg: Config) -> None:
    cfg.require()
    client, store = LLMClient(cfg), Store(cfg.db)
    files = iter_files(args.paths, recursive=not args.no_recursive)
    if not files:
        sys.exit("nothing to ingest")

    total = 0
    for path in files:
        try:
            sha = store.digest(path)
            if store.is_current(path, sha) and not args.force:
                print(f"= {path} (unchanged)")
                continue
            text = read_document(path)
            chunks = chunk_text(text, size=args.chunk_size, overlap=args.overlap)
            if not chunks:
                print(f"! {path} (empty)")
                continue
            vectors = client.embed(chunks)
            store.replace_document(path, sha, chunks, vectors)
            total += len(chunks)
            print(f"+ {path} ({len(chunks)} chunks)")
        except (RuntimeError, OSError) as exc:
            print(f"! {path}: {exc}", file=sys.stderr)
    store.close()
    print(f"\nindexed {total} new chunks")


def cmd_docs(args, cfg: Config) -> None:
    cfg.require()
    store = Store(cfg.db)
    rows = store.documents()
    if not rows:
        print("store is empty - try `llmcli ingest ./notes`")
    for row in rows:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["added_at"]))
        print(f"{row['chunks']:>5} chunks  {stamp}  {row['path']}")
    store.close()


def cmd_forget(args, cfg: Config) -> None:
    cfg.require()
    store = Store(cfg.db)
    n = store.forget(args.pattern)
    store.conn.execute("VACUUM")
    store.close()
    print(f"removed {n} document(s)")


def cmd_ask(args, cfg: Config) -> None:
    cfg.require()
    client, store = LLMClient(cfg), Store(cfg.db)
    hits = retrieve(client, store, args.question, args.top_k, args.min_score)
    context = build_context(hits)
    if args.show_context:
        print(context or "(no context)", "\n" + "=" * 60)
    client.chat(
        rag_messages(args.question, context),
        stream=not args.no_stream,
        temperature=args.temperature,
    )
    if hits and not args.quiet:
        names = sorted({Path(p).name for _, p, _ in hits})
        print("\nsources: " + ", ".join(names))
    store.close()


def register(sub, parents: Parents) -> None:
    p = sub.add_parser("ask", parents=[parents.common, parents.gen, parents.ret],
                       help="one-shot question answered from your documents")
    p.add_argument("question")
    p.add_argument("--show-context", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true", help="hide the sources line")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("ingest", parents=[parents.common],
                       help="add files or folders to the store")
    p.add_argument("paths", nargs="+")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    p.add_argument("--no-recursive", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("docs", parents=[parents.common], help="list ingested documents")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("forget", parents=[parents.common],
                       help="delete documents matching a substring")
    p.add_argument("pattern")
    p.set_defaults(func=cmd_forget)
