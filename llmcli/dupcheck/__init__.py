"""Duplicate / near-duplicate code detection.

Source code, not documents: files are parsed into functions and classes with
tree-sitter, embedded through llmcli.tools, and every candidate pair has to
corroborate structurally before it is reported.
"""

from llmcli.dupcheck.job import Job
from llmcli.dupcheck.scan import find_clusters, find_pairs
from llmcli.dupcheck.structure import build_idf, distinctiveness, structural_similarity
from llmcli.dupcheck.units import CodeUnit, extract_units, iter_source_files

__all__ = [
    "Job", "CodeUnit", "extract_units", "iter_source_files",
    "find_pairs", "find_clusters",
    "build_idf", "structural_similarity", "distinctiveness",
]
