"""Structural corroboration from the parse tree.

Embedding similarity conflates "these do the same thing" with "these talk
about the same thing", so every candidate pair also has to agree structurally.
These are the gates that make that judgement.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from llmcli.dupcheck.units import CodeUnit

# below this many parsed units, inverse-document-frequency weighting is noise
IDF_MIN_UNITS = 25


@dataclass
class Idf:
    """Rarity weights for AST bigrams, plus the scale needed to normalise them."""

    weights: Dict[Tuple[str, str], float]
    norm: float  # weight of a bigram seen exactly once: the maximum

    def w(self, bigram: Tuple[str, str]) -> float:
        return self.weights.get(bigram, self.norm)


def build_idf(units: List[CodeUnit]) -> Optional[Idf]:
    """Rarity weight per AST bigram, over the units in this run.

    A codebase with a house style repeats the same shapes everywhere - in C
    with error codes, every accessor is `if (bad) return ERR; ...; return OK;`.
    Those shapes are what everything has in common, so they say nothing about
    duplication. Weighting by inverse document frequency makes ubiquitous
    structure count for almost nothing and rare structure count for a lot.
    Smoothed so a tiny corpus degrades gracefully instead of collapsing to 0.
    """
    shapes = [u.shape for u in units if u.shape]
    n = len(shapes)
    # document frequency is meaningless over a handful of units, and punishing
    # "common" structure in a 10-unit run drops real duplicates, so stay plain
    if n < IDF_MIN_UNITS:
        return None
    df: Counter = Counter()
    for shape in shapes:
        df.update(shape.keys())  # distinct bigrams per unit
    weights = {bg: math.log((n + 1) / (d + 0.5)) for bg, d in df.items()}
    return Idf(weights, math.log((n + 1) / 0.5))


def distinctiveness(a: Optional[Counter], b: Optional[Counter], idf: Optional[Idf]) -> float:
    """How *informative* the structure two units share is, on a 0..1 scale.

    Structural similarity is a ratio, so it is scale-invariant: two thin typed
    wrappers with byte-identical shape score a perfect 1.0 even though all they
    share is the codebase's boilerplate. This is the absolute counterpart - the
    mean rarity of the shared structure - which stays near 0 for boilerplate
    however identical it is. Returns -1.0 ("no signal") without an IDF corpus.
    """
    if not a or not b or idf is None:
        return -1.0
    keys = a.keys() & b.keys()
    shared = sum(min(a[k], b[k]) for k in keys)
    if not shared:
        return 0.0
    evidence = sum(min(a[k], b[k]) * idf.w(k) for k in keys)
    return (evidence / shared) / idf.norm


def structural_similarity(
    a: Optional[Counter], b: Optional[Counter], idf: Optional[Idf] = None
) -> float:
    """Weighted Jaccard over AST bigrams. Returns -1.0 when either side has no
    shape (window fallback), meaning 'no signal' rather than 'no similarity'."""
    if not a or not b:
        return -1.0
    if idf is None:
        inter = sum((a & b).values())
        union = sum((a | b).values())
        return inter / union if union else 0.0

    inter = union = 0.0
    for bg in a.keys() | b.keys():
        w = idf.w(bg)
        ca, cb = a.get(bg, 0), b.get(bg, 0)
        inter += min(ca, cb) * w
        union += max(ca, cb) * w
    return inter / union if union > 1e-9 else 0.0
