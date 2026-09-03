"""The pairwise scan and its clustering."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from llmcli.dupcheck.structure import build_idf, distinctiveness, structural_similarity
from llmcli.dupcheck.units import CodeUnit
from llmcli.tools.progress import Progress
from llmcli.tools.vectors import dot, np


def _overlaps(a: CodeUnit, b: CodeUnit) -> bool:
    return a.path == b.path and a.start <= b.end and b.start <= a.end


def _calls_each_other(a: CodeUnit, b: CodeUnit) -> bool:
    """True when one unit references the other's name.

    A caller shares its callee's name and vocabulary, which is enough for an
    embedding to rank the pair as similar even though one merely uses the
    other. That is a call relationship, not duplication.
    """
    return bool(a.name and a.name in b.idents) or bool(b.name and b.name in a.idents)


def find_pairs(
    units: List[CodeUnit],
    vectors: List[Sequence[float]],
    threshold: float,
    progress: bool = True,
    min_structure: float = 0.5,
    filter_callers: bool = True,
    idf_weighting: bool = True,
    min_distinct: float = 0.10,
) -> List[Tuple[float, float, float, int, int]]:
    """Pairs above `threshold` that also survive the structural corroboration.

    Embedding similarity alone conflates 'these do the same thing' with 'these
    talk about the same thing', so each candidate must also match structurally
    (AST bigrams) and must not be a caller/callee pair.
    """
    n = len(units)
    pairs: List[Tuple[float, float, float, int, int]] = []
    idf = build_idf(units) if idf_weighting else None

    def keep(s: float, i: int, j: int) -> None:
        a, b = units[i], units[j]
        if _overlaps(a, b):
            return
        if filter_callers and _calls_each_other(a, b):
            return
        shaped = structural_similarity(a.shape, b.shape, idf)
        # -1.0 means at least one side has no parse tree (window fallback), so
        # there is nothing to corroborate with; keep the pair rather than drop it
        if shaped >= 0.0 and shaped < min_structure:
            return
        distinct = distinctiveness(a.shape, b.shape, idf)
        if distinct >= 0.0 and distinct < min_distinct:
            return
        pairs.append((s, shaped, distinct, i, j))

    with Progress("comparing", n, progress) as prog:
        if np is not None:
            matrix = np.asarray(vectors, dtype="f4")
            sims = matrix @ matrix.T
            for i in range(n):
                row = sims[i]
                for j in range(i + 1, n):
                    s = float(row[j])
                    if s >= threshold:
                        keep(s, i, j)
                prog.advance()
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    s = dot(vectors[i], vectors[j])
                    if s >= threshold:
                        keep(s, i, j)
                prog.advance()
    pairs.sort(key=lambda t: -t[0])
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_clusters(units: List[CodeUnit], pairs: List[Tuple[float, float, float, int, int]]) -> List[List[int]]:
    uf = UnionFind(len(units))
    for *_, i, j in pairs:
        uf.union(i, j)
    groups: Dict[int, List[int]] = {}
    for idx in range(len(units)):
        groups.setdefault(uf.find(idx), []).append(idx)
    clusters = [g for g in groups.values() if len(g) > 1]
    clusters.sort(key=len, reverse=True)
    return clusters
