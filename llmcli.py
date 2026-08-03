#!/usr/bin/env python3
"""
llmcli - a tiny chat + RAG client for any OpenAI-compatible LLM API.

Settings and embeddings live in .llmcli/ folders. Starting at the cwd (or
--workspace/-w DIR), llmcli walks up the tree: settings are inherited from
every ancestor with the nearest winning, while the embedding store is always
the nearest .llmcli/. So a parent directory can hold the url, api key and
models once, and each project subdirectory keeps its own documents.

    ~/work/.llmcli/           url + api key + models      (shared)
    ~/work/handbook/.llmcli/  store.db                    (handbook's docs)
    ~/work/legal/.llmcli/     store.db + model override   (legal's docs)

A lone directory with its own .llmcli/ works exactly the same way, with
config and embeddings side by side in one place.

Commands
    init      create a workspace here; unset values inherit from parents
    config    show the effective settings and where each came from
    model     show or change the model for this directory
    chat      interactive chat REPL (optionally grounded on your documents)
    ask       one-shot question answered from your ingested documents
    ingest    chunk + embed files or directories into the nearest store
    docs      list what has been ingested
    forget    remove documents from the store

Commands other than init exit with an error if no workspace is found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install requests")

try:
    import numpy as np
except ImportError:  # numpy is optional; we fall back to pure python
    np = None


WORKSPACE_DIRNAME = ".llmcli"

DEFAULT_SYSTEM = "You are a concise, accurate assistant."
RAG_SYSTEM = (
    "You answer questions using the provided context excerpts. "
    "Cite the source filenames you used. If the context does not contain the "
    "answer, say so plainly instead of guessing."
)


# --------------------------------------------------------------------------
# workspace + config
# --------------------------------------------------------------------------


class Workspace:
    """Locates the .llmcli/ folders that apply to a directory.

    Resolution starts at --workspace/-w DIR, else $LLMCLI_WORKSPACE, else the
    cwd, and then walks up to the filesystem root collecting every .llmcli/
    it finds. Settings are inherited down the chain with the nearest file
    winning, while the embedding store is always the *nearest* .llmcli/ - so a
    parent directory can hold the url, key and models once, and each project
    subdirectory keeps its own documents:

        ~/work/.llmcli/          url + api key + models   (shared)
        ~/work/handbook/.llmcli/ store.db                 (handbook's docs)
        ~/work/legal/.llmcli/    store.db + model override (legal's docs)

    A directory with its own .llmcli/ and no ancestors behaves exactly as
    before: config and embeddings side by side in one place.
    """

    def __init__(self, start: Path, chain: List[Path]):
        self.start = start
        self.chain = chain  # existing .llmcli dirs, nearest first

    @property
    def local(self) -> Path:
        """Where `init` would create a workspace."""
        return self.start / WORKSPACE_DIRNAME

    @property
    def active(self) -> Optional[Path]:
        """Nearest existing .llmcli - owns the embedding store."""
        return self.chain[0] if self.chain else None

    @property
    def db_path(self) -> Path:
        return (self.active or self.local) / "store.db"

    def exists(self) -> bool:
        return bool(self.chain)

    @classmethod
    def resolve(cls, args: argparse.Namespace) -> "Workspace":
        raw = (
            getattr(args, "workspace", None)
            or os.environ.get("LLMCLI_WORKSPACE")
            or Path.cwd()
        )
        start = Path(raw).expanduser().resolve()
        chain = [
            d / WORKSPACE_DIRNAME
            for d in (start, *start.parents)
            if (d / WORKSPACE_DIRNAME / "config.json").exists()
        ]
        if getattr(args, "no_inherit", False):
            chain = chain[:1]
        return cls(start, chain)


@dataclass
class Config:
    url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    ws: Optional[Workspace] = None

    FIELDS = ("url", "api_key", "model", "embed_model")

    def __post_init__(self):
        self.sources: dict = {}  # field -> where the effective value came from

    @property
    def db(self) -> str:
        return str(self.ws.db_path)

    @classmethod
    def load(cls, args: argparse.Namespace) -> "Config":
        ws = Workspace.resolve(args)
        cfg = cls(ws=ws)
        for f in cls.FIELDS:
            cfg.sources[f] = "default" if getattr(cfg, f) else None

        # farthest ancestor first, so nearer files override
        for folder in reversed(ws.chain):
            path = folder / "config.json"
            try:
                data = json.loads(path.read_text() or "{}")
            except (OSError, json.JSONDecodeError):
                print(f"warning: could not parse {path}", file=sys.stderr)
                continue
            for f in cls.FIELDS:
                if data.get(f):
                    setattr(cfg, f, data[f])
                    cfg.sources[f] = str(path)

        for f, var in zip(cls.FIELDS, ("LLM_API_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_EMBED_MODEL")):
            if os.environ.get(var):
                setattr(cfg, f, os.environ[var])
                cfg.sources[f] = f"${var}"

        for f in cls.FIELDS:
            val = getattr(args, f, None)
            if val:
                setattr(cfg, f, val)
                cfg.sources[f] = "--flag"
        return cfg

    # -- writing ----------------------------------------------------------

    def _write(self, folder: Path, updates: dict) -> Path:
        """Merge `updates` into folder/config.json, leaving other keys alone."""
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.json"
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                pass
        data.update({k: v for k, v in updates.items() if v})
        path.write_text(json.dumps(data, indent=2))
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path

    def save_local(self, updates: dict) -> Path:
        """Write to the workspace being created/used here."""
        return self._write(self.ws.local, updates)

    def save_active(self, updates: dict) -> Path:
        """Write to the nearest existing workspace."""
        return self._write(self.ws.active or self.ws.local, updates)

    def require(self) -> None:
        """Every command except init needs a workspace here or above."""
        if not self.ws.exists():
            sys.exit(
                f"no llmcli workspace at or above {self.ws.start}\n"
                f"  expected {self.ws.local / 'config.json'} (or one in a parent directory)\n"
                f"  run `llmcli init --url ... --model ...` here, "
                f"or point at one with --workspace /path/to/dir"
            )
        if not self.url:
            sys.exit(
                f"no API url for {self.ws.start} - run `llmcli init --url ...` "
                f"here or in a parent directory"
            )


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------


class LLMClient:
    """Minimal client for OpenAI-compatible /chat/completions and /embeddings."""

    def __init__(self, cfg: Config, timeout: int = 120):
        self.cfg = cfg
        self.timeout = timeout
        self.session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
            headers["x-api-key"] = cfg.api_key  # some gateways want this instead
        self.session.headers.update(headers)

    def _endpoint(self, path: str) -> str:
        base = self.cfg.url.rstrip("/")
        # allow the user to paste a full endpoint url
        for suffix in ("/chat/completions", "/embeddings", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}/{path.lstrip('/')}"

    def _post(self, path: str, payload: dict, stream: bool = False):
        try:
            resp = self.session.post(
                self._endpoint(path), json=payload, timeout=self.timeout, stream=stream
            )
        except requests.RequestException as exc:
            raise SystemExit(f"request failed: {exc}")
        if resp.status_code >= 400:
            body = resp.text[:500]
            raise SystemExit(f"API error {resp.status_code}: {body}")
        return resp

    def chat(self, messages: List[dict], stream: bool = True, **kw) -> str:
        payload = {"model": self.cfg.model, "messages": messages, "stream": stream}
        payload.update({k: v for k, v in kw.items() if v is not None})

        if not stream:
            data = self._post("chat/completions", payload).json()
            text = data["choices"][0]["message"]["content"]
            print(text)
            return text

        resp = self._post("chat/completions", payload, stream=True)
        parts: List[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        sys.stdout.write("\n")
        return "".join(parts)

    def embed(self, texts: Sequence[str], batch: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            window = list(texts[i : i + batch])
            payload = {"model": self.cfg.embed_model, "input": window}
            data = self._post("embeddings", payload).json()
            rows = sorted(data["data"], key=lambda r: r.get("index", 0))
            out.extend(r["embedding"] for r in rows)
        return out


# --------------------------------------------------------------------------
# document loading + chunking
# --------------------------------------------------------------------------

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".html", ".htm", ".xml",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h",
    ".cpp", ".hpp", ".rb", ".sh", ".sql",
}


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("PDF support needs: pip install pypdf")
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            raise RuntimeError("DOCX support needs: pip install python-docx")
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if suffix in TEXT_SUFFIXES or suffix == "":
        return path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError(f"unsupported file type: {suffix}")


def iter_files(paths: Iterable[str], recursive: bool = True) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            globber = p.rglob("*") if recursive else p.glob("*")
            for child in sorted(globber):
                if child.is_file() and not child.name.startswith("."):
                    if child.suffix.lower() in TEXT_SUFFIXES | {".pdf", ".docx"}:
                        found.append(child)
        elif p.is_file():
            found.append(p)
        else:
            print(f"skipping (not found): {p}", file=sys.stderr)
    return found


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    """Split on paragraph boundaries, packing up to `size` characters per chunk."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        while len(para) > size:  # hard-split monster paragraphs
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(para[:size])
            para = para[size - overlap :]
        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= size:
            buf += "\n\n" + para
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n\n" + para).strip() if tail else para
    if buf:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------
# vector helpers (numpy if present, stdlib otherwise)
# --------------------------------------------------------------------------


def normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack(blob: bytes) -> Sequence[float]:
    if np is not None:
        return np.frombuffer(blob, dtype="<f4")
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if np is not None:
        return float(np.dot(np.asarray(a, dtype="f4"), np.asarray(b, dtype="f4")))
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# rag glue
# --------------------------------------------------------------------------


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


def rag_messages(question: str, context: str, history: Optional[List[dict]] = None) -> List[dict]:
    msgs = [{"role": "system", "content": RAG_SYSTEM}]
    if history:
        msgs.extend(history)
    user = (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context
        else f"(no matching context found in the local store)\n\nQuestion: {question}"
    )
    msgs.append({"role": "user", "content": user})
    return msgs


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _origin(cfg: Config, field: str) -> str:
    """Human label for where a setting came from."""
    src = cfg.sources.get(field)
    if not src or src in ("default", "--flag"):
        return {"default": "(default)", "--flag": "(this command)"}.get(src, "")
    if src.startswith("$"):
        return f"({src})"
    folder = Path(src).parent
    if cfg.ws.active and folder == cfg.ws.active:
        return "(here)"
    return f"(inherited from {folder.parent})"


def cmd_init(args, cfg: Config) -> None:
    if cfg.ws.local.joinpath("config.json").exists() and not args.force:
        sys.exit(f"workspace already exists at {cfg.ws.local} (use --force to overwrite)")

    # write only what was passed, so inherited settings stay live and keep
    # tracking the parent if it changes later
    updates = {f: getattr(args, f, None) for f in Config.FIELDS}
    if not any(updates.values()) and not cfg.ws.exists():
        sys.exit("init needs at least --url (e.g. --url https://api.openai.com/v1)")
    if not cfg.url:
        sys.exit("no --url given and none inherited from a parent directory")

    path = cfg.save_local(updates)
    inherited = [f for f in Config.FIELDS if not updates.get(f)]
    cfg = Config.load(args)  # reload so provenance reflects the new file

    print(f"initialised workspace {cfg.ws.start}")
    print(f"  config      {path}")
    print(f"  database    {cfg.ws.db_path}")
    for field, shown in (
        ("url", cfg.url),
        ("api_key", "set" if cfg.api_key else "(unset)"),
        ("model", cfg.model),
        ("embed_model", cfg.embed_model),
    ):
        print(f"  {field:<11} {shown} {_origin(cfg, field)}")
    if inherited and len(cfg.ws.chain) > 1:
        print("\nsettings not written here stay linked to the parent workspace")


def cmd_config(args, cfg: Config) -> None:
    cfg.require()
    print(f"workspace   {cfg.ws.start}")
    print(f"database    {cfg.ws.db_path}")
    print(f"url         {cfg.url} {_origin(cfg, 'url')}")
    print(f"api_key     {'set' if cfg.api_key else '(unset)'} {_origin(cfg, 'api_key')}")
    print(f"model       {cfg.model} {_origin(cfg, 'model')}")
    print(f"embed_model {cfg.embed_model} {_origin(cfg, 'embed_model')}")
    print("\nconfig files, nearest first:")
    for folder in cfg.ws.chain:
        print(f"  {folder / 'config.json'}")


def cmd_model(args, cfg: Config) -> None:
    cfg.require()
    if not args.name:
        print(f"chat model   {cfg.model} {_origin(cfg, 'model')}")
        print(f"embed model  {cfg.embed_model} {_origin(cfg, 'embed_model')}")
        return

    field = "embed_model" if args.embed else "model"
    label = "embed model" if args.embed else "chat model"
    # default target is the config actually in use; --here pins it to this dir
    path = cfg.save_local({field: args.name}) if args.here else cfg.save_active({field: args.name})
    print(f"{label} -> {args.name}")
    print(f"written to {path}")
    if args.embed:
        print("note: existing embeddings were built with the old model - "
              "re-run `llmcli ingest --force` to rebuild them")


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


HELP = """
  /rag on|off     toggle document retrieval for following turns
  /sources        show the chunks retrieved for the last question
  /system TEXT    replace the system prompt
  /reset          clear the conversation history
  /save FILE      write the transcript to FILE
  /help  /exit
"""


def cmd_chat(args, cfg: Config) -> None:
    cfg.require()
    client = LLMClient(cfg)
    store = Store(cfg.db)
    use_rag = args.rag
    system = args.system or (RAG_SYSTEM if use_rag else DEFAULT_SYSTEM)
    history: List[dict] = []
    last_hits: List[Tuple[float, str, str]] = []

    print(f"model {cfg.model}  rag {'on' if use_rag else 'off'}  (/help for commands)")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd, rest = cmd.lower(), rest.strip()
            if cmd in ("/exit", "/quit", "/q"):
                break
            if cmd == "/help":
                print(HELP)
            elif cmd == "/rag":
                use_rag = rest != "off"
                print(f"rag {'on' if use_rag else 'off'}")
            elif cmd == "/sources":
                for score, path, text in last_hits:
                    print(f"\n[{score:.3f}] {path}\n{text[:400]}...")
                if not last_hits:
                    print("(nothing retrieved yet)")
            elif cmd == "/system":
                system = rest or system
                print("system prompt updated")
            elif cmd == "/reset":
                history.clear()
                print("history cleared")
            elif cmd == "/save":
                target = Path(rest or "transcript.md").expanduser()
                target.write_text(
                    "\n\n".join(f"**{m['role']}**: {m['content']}" for m in history)
                )
                print(f"wrote {target}")
            else:
                print("unknown command - /help")
            continue

        if use_rag:
            last_hits = retrieve(client, store, line, args.top_k, args.min_score)
            context = build_context(last_hits)
            turn = (
                f"Context:\n{context}\n\nQuestion: {line}" if context else line
            )
        else:
            turn = line

        messages = [{"role": "system", "content": system}] + history[-2 * args.history_turns :]
        messages.append({"role": "user", "content": turn})
        reply = client.chat(
            messages, stream=not args.no_stream, temperature=args.temperature
        )
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": reply})
        if use_rag and last_hits:
            print("sources: " + ", ".join(sorted({Path(p).name for _, p, _ in last_hits})))

    store.close()


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", help="API base url, e.g. https://api.openai.com/v1")
    common.add_argument("--api-key", dest="api_key")
    common.add_argument("--model", help="chat model for this invocation")
    common.add_argument("--embed-model", dest="embed_model")
    common.add_argument(
        "-w", "--workspace", metavar="DIR",
        help="directory to resolve the workspace from (default: cwd)",
    )
    common.add_argument(
        "--no-inherit", action="store_true",
        help="ignore config in parent directories",
    )

    gen = argparse.ArgumentParser(add_help=False)
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--no-stream", action="store_true")

    ret = argparse.ArgumentParser(add_help=False)
    ret.add_argument("-k", "--top-k", type=int, default=4)
    ret.add_argument("--min-score", type=float, default=0.0)

    parser = argparse.ArgumentParser(prog="llmcli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", parents=[common],
                       help="create a workspace here; unset values inherit from parents")
    p.add_argument("--force", action="store_true", help="overwrite an existing workspace")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("config", parents=[common],
                       help="show the effective settings and where each came from")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("model", parents=[common],
                       help="show or change the model for this directory")
    p.add_argument("name", nargs="?", help="model to switch to; omit to print the current one")
    p.add_argument("--embed", action="store_true", help="set the embedding model instead")
    p.add_argument("--here", action="store_true",
                   help="write to this directory instead of the nearest workspace")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("chat", parents=[common, gen, ret], help="interactive chat")
    p.add_argument("--rag", action="store_true", help="ground answers on ingested docs")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--history-turns", type=int, default=8)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ask", parents=[common, gen, ret], help="one-shot RAG question")
    p.add_argument("question")
    p.add_argument("--show-context", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true", help="hide the sources line")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("ingest", parents=[common], help="add files or folders to the store")
    p.add_argument("paths", nargs="+")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    p.add_argument("--no-recursive", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("docs", parents=[common], help="list ingested documents")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("forget", parents=[common], help="delete documents matching a substring")
    p.add_argument("pattern")
    p.set_defaults(func=cmd_forget)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
