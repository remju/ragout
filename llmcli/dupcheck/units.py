"""Splitting source files into comparable code units.

Function and class boundaries come from tree-sitter, a real grammar, rather
than pattern matching; anything that cannot be parsed falls back to fixed-size
overlapping line windows so a run always produces units.
"""

from __future__ import annotations

import sys
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tree_sitter_language_pack import detect_language_from_path, get_parser
except ImportError:  # extraction degrades to line windows without it
    detect_language_from_path = None
    get_parser = None


# A definition node is recognised generically rather than with a per-language
# table: across tree-sitter grammars these node types are named consistently as
# <what>_<form>, e.g. function_definition, method_declaration, struct_item,
# class_specifier. Requiring both halves keeps out the lookalikes that share a
# suffix (field_declaration, parameter_declaration, access_specifier).
DEF_PREFIXES = (
    "function", "method", "class", "struct", "interface", "constructor",
    "impl", "trait", "module", "enum", "union", "namespace", "record",
    "protocol", "object", "type", "subroutine", "procedure", "singleton",
)
DEF_SUFFIXES = ("_definition", "_declaration", "_item", "_specifier")
# grammars where the whole construct is a single bare word (Ruby, and similar)
DEF_EXACT = {"method", "class", "module", "singleton_method", "function", "def"}
# lambda-ish forms worth capturing even though they break the naming pattern
DEF_EXTRA = {"arrow_function", "function_expression"}

NAME_NODE_TYPES = {
    "identifier", "type_identifier", "field_identifier", "constant",
    "word", "name", "property_identifier", "scoped_identifier",
    "qualified_identifier", "simple_identifier",
}

_PARSER_CACHE: Dict[str, object] = {}
_PARSER_WARNED: set = set()
_PACKAGE_WARNED = False

DEFAULT_EXCLUDES = [
    ".git", "__pycache__", "node_modules", "vendor", "dist", "build",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
]


@dataclass
class CodeUnit:
    path: str
    kind: str  # "function" | "class" | "window"
    name: str
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    text: str
    # corroborating signals, filled in when tree-sitter parsed the file.
    # `shape` is a multiset of AST node-type bigrams - identical for clones
    # whose identifiers were renamed, unlike anything an embedding sees.
    shape: Optional[Counter] = None
    idents: frozenset = frozenset()


def _shape_and_idents(node) -> Tuple[Counter, frozenset]:
    """Preorder walk collecting node-type bigrams and identifier text."""
    types: List[str] = []
    idents: set = set()

    def visit(n) -> None:
        types.append(n.type)
        if n.type in NAME_NODE_TYPES and not n.children:
            idents.add(n.text.decode("utf-8", "replace"))
        for c in n.children:
            visit(c)

    visit(node)
    bigrams = Counter(zip(types, types[1:]))
    return bigrams, frozenset(idents)


def _is_definition(node_type: str) -> bool:
    if node_type in DEF_EXACT or node_type in DEF_EXTRA:
        return True
    if not node_type.endswith(DEF_SUFFIXES):
        return False
    return node_type.startswith(DEF_PREFIXES)


def _kind_of(node_type: str) -> str:
    if node_type.startswith(("function", "method", "constructor", "subroutine", "procedure")):
        return "function"
    if node_type in DEF_EXTRA or node_type in {"method", "function", "def", "singleton_method"}:
        return "function"
    return "class"


def _node_name(node) -> str:
    """Best-effort name for a definition node.

    Grammars disagree: Python/Go/Java expose a `name` field, while C and C++
    bury the identifier in a `declarator` chain, so fall back to walking it.
    """
    field = node.child_by_field_name("name")
    if field is not None:
        return field.text.decode("utf-8", "replace")

    cur = node
    for _ in range(6):  # unwrap pointer/array/function declarator layers
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            break
        cur = nxt
        if cur.type in NAME_NODE_TYPES:
            return cur.text.decode("utf-8", "replace")

    for child in cur.children:  # e.g. function_declarator -> identifier
        if child.type in NAME_NODE_TYPES:
            return child.text.decode("utf-8", "replace")

    # Go wraps the name a level down: type_declaration -> type_spec(name:)
    for child in node.children:
        if child.type.endswith(("_spec", "_declarator")):
            nested = child.child_by_field_name("name")
            if nested is not None:
                return nested.text.decode("utf-8", "replace")

    # anonymous forms (arrow functions, function expressions) take the name of
    # whatever binds them: const foo = (x) => ..., foo: function () { ... }
    parent = node.parent
    for _ in range(2):
        if parent is None:
            break
        bound = parent.child_by_field_name("name") or parent.child_by_field_name("key")
        if bound is not None and bound.id != node.id:
            return bound.text.decode("utf-8", "replace")
        parent = parent.parent

    for child in node.children:
        if child.type in NAME_NODE_TYPES:
            return child.text.decode("utf-8", "replace")
    return ""


def _get_parser(language: str):
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    try:
        parser = get_parser(language)
    except Exception as exc:  # unknown grammar, or download failed offline
        if language not in _PARSER_WARNED:
            _PARSER_WARNED.add(language)
            print(f"no tree-sitter grammar for {language} ({exc}); using line windows", file=sys.stderr)
        parser = None
    _PARSER_CACHE[language] = parser
    return parser


def _extract_tree_sitter(path: Path, source: bytes, min_lines: int) -> List[Tuple]:
    if get_parser is None or detect_language_from_path is None:
        global _PACKAGE_WARNED
        if not _PACKAGE_WARNED:
            _PACKAGE_WARNED = True
            print(
                "tree-sitter-language-pack is not installed - every file falls back to "
                "fixed-size line windows, so real functions won't be matched against each "
                "other. Install it with: pip install 'llmcli[dupcheck]'",
                file=sys.stderr,
            )
        return []
    try:
        language = detect_language_from_path(str(path))
    except Exception:
        return []
    if not language:
        return []
    parser = _get_parser(language)
    if parser is None:
        return []

    try:
        root = parser.parse(source).root_node
    except Exception as exc:
        print(f"parse failed for {path}: {exc}", file=sys.stderr)
        return []

    units: List[Tuple[str, str, int, int, Optional[Counter], frozenset]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            stack.append(child)
        if node is root:  # Python's root node is literally named "module"
            continue
        if not _is_definition(node.type):
            continue
        start = node.start_point[0] + 1
        end = node.end_point[0] + 1
        if end - start + 1 >= min_lines:
            shape, idents = _shape_and_idents(node)
            units.append((_kind_of(node.type), _node_name(node), start, end, shape, idents))
    units.sort(key=lambda u: (u[2], u[3]))
    return units


def _extract_windows(
    lines: List[str], window_lines: int, overlap: int, min_lines: int
) -> List[Tuple]:
    n = len(lines)
    if n == 0:
        return []
    units = []
    step = max(window_lines - overlap, 1)
    i = 0
    while i < n:
        end = min(i + window_lines, n)
        if end - i >= min_lines and any(l.strip() for l in lines[i:end]):
            units.append(("window", "", i + 1, end, None, frozenset()))
        if end == n:
            break
        i += step
    return units


def extract_units(path: Path, min_lines: int, window_lines: int, window_overlap: int) -> List[CodeUnit]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        print(f"skipping {path}: {exc}", file=sys.stderr)
        return []
    text = raw_bytes.decode("utf-8", errors="replace")
    # split on \n only: tree-sitter counts rows the same way, whereas
    # str.splitlines() also breaks on \f, \v and U+2028 and would desync
    lines = [ln.rstrip("\r") for ln in text.split("\n")]

    raw = _extract_tree_sitter(path, raw_bytes, min_lines)
    if not raw:
        raw = _extract_windows(lines, window_lines, window_overlap, min_lines)

    units = []
    for kind, name, start, end, shape, idents in raw:
        block = textwrap.dedent("\n".join(lines[start - 1 : end])).strip("\n")
        if block.strip():
            units.append(CodeUnit(str(path), kind, name, start, end, block, shape, idents))
    return units


def iter_source_files(includes: List[str], extensions: List[str], excludes: List[str]) -> List[Path]:
    exts = {e if e.startswith(".") else f".{e}" for e in extensions}
    found: List[Path] = []
    for raw in includes:
        p = Path(raw).expanduser()
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if not child.is_file() or child.suffix.lower() not in exts:
                    continue
                posix = child.as_posix()
                if any(pat in posix for pat in excludes):
                    continue
                found.append(child)
        elif p.is_file():
            if p.suffix.lower() in exts:
                found.append(p)
        else:
            print(f"skipping (not found): {p}", file=sys.stderr)
    return found
