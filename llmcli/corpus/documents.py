"""Reading documents off disk and splitting them into chunks."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List

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
            raise RuntimeError("PDF support needs: pip install 'llmcli[corpus]'")
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            raise RuntimeError("DOCX support needs: pip install 'llmcli[corpus]'")
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
