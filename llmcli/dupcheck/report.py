"""Rendering a run as text, markdown or json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from llmcli.dupcheck.units import CodeUnit


def _ref_str(u: CodeUnit) -> str:
    label = f" {u.name}" if u.name else ""
    return f"{u.path}:{u.start}-{u.end}{label}"


def _ref_dict(u: CodeUnit) -> dict:
    return {"path": u.path, "start": u.start, "end": u.end, "name": u.name, "kind": u.kind}


@dataclass
class Timings:
    """Wall-clock split of a run. `embed` covers the API calls (and is ~0 on a
    fully cached run); `compare` covers the similarity scan."""

    embed: float = 0.0
    compare: float = 0.0
    total: float = 0.0

    def line(self) -> str:
        return (
            f"{self.total:.2f}s total "
            f"(embedding {self.embed:.2f}s, comparison {self.compare:.2f}s)"
        )

    def as_dict(self) -> dict:
        return {
            "total_seconds": round(self.total, 3),
            "embed_seconds": round(self.embed, 3),
            "compare_seconds": round(self.compare, 3),
        }


def _struct_str(v: float) -> str:
    return "n/a" if v < 0 else f"{v:.3f}"


def render_pairwise(
    units: List[CodeUnit], pairs: List[Tuple[float, float, float, int, int]], output: str, top_n: Optional[int], timings: Timings
) -> str:
    if top_n:
        pairs = pairs[:top_n]
    if output == "json":
        data = {
            "elapsed": timings.as_dict(),
            "pairs": [
                {
                    "similarity": round(s, 4),
                    "structural": None if st < 0 else round(st, 4),
                    "distinctiveness": None if di < 0 else round(di, 4),
                    "a": _ref_dict(units[i]),
                    "b": _ref_dict(units[j]),
                }
                for s, st, di, i, j in pairs
            ],
        }
        return json.dumps(data, indent=2)
    if output == "markdown":
        lines = [
            f"_Analysis took {timings.line()}._", "",
            "| similarity | structural | distinct | a | b |", "|---|---|---|---|---|",
        ]
        lines += [
            f"| {s:.3f} | {_struct_str(st)} | {_struct_str(di)} | `{_ref_str(units[i])}` | `{_ref_str(units[j])}` |"
            for s, st, di, i, j in pairs
        ]
        return "\n".join(lines)
    lines = [f"analysis took {timings.line()}", ""]
    lines += [
        f"{s:.3f}  struct {_struct_str(st)}  distinct {_struct_str(di)}  {_ref_str(units[i])}  <->  {_ref_str(units[j])}"
        for s, st, di, i, j in pairs
    ]
    return "\n".join(lines)


def render_clusters(
    units: List[CodeUnit], clusters: List[List[int]], output: str, top_n: Optional[int], timings: Timings
) -> str:
    if top_n:
        clusters = clusters[:top_n]
    if output == "json":
        data = {
            "elapsed": timings.as_dict(),
            "clusters": [{"members": [_ref_dict(units[k]) for k in idxs]} for idxs in clusters],
        }
        return json.dumps(data, indent=2)
    lines = []
    if output == "markdown":
        lines.append(f"_Analysis took {timings.line()}._")
        lines.append("")
    else:
        lines.append(f"analysis took {timings.line()}")
        lines.append("")
    for n, idxs in enumerate(clusters, 1):
        if output == "markdown":
            lines.append(f"### Cluster {n} ({len(idxs)} matches)")
            lines += [f"- `{_ref_str(units[k])}`" for k in idxs]
            lines.append("")
        else:
            lines.append(f"cluster {n} ({len(idxs)} matches):")
            lines += [f"  {_ref_str(units[k])}" for k in idxs]
    return "\n".join(lines)
