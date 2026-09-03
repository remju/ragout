"""The job config: what to scan and how strict to be.

Purely about the analysis. Endpoint, credentials, TLS and batching live in
.llmcli/config.json, shared with every other llmcli command, so a job file can
be committed to a repository without carrying anything environment-specific.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from llmcli.dupcheck.units import DEFAULT_EXCLUDES

# fields that used to live here and now come from .llmcli/config.json
RETIRED_KEYS = {
    "url": "llmcli init --url ...",
    "api_key": "llmcli init --api-key ...",
    "embed_model": "llmcli model --embed ...",
    "ca_bundle": "llmcli init --ca-bundle ...",
    "verify_ssl": "llmcli init --insecure",
    "timeout": "llmcli init --timeout ...",
    "embed_batch": "$LLM_EMBED_BATCH or --batch",
    "workspace": "llmcli dupcheck run --workspace DIR ...",
}

# canonical extension per output format, and the extensions we recognise as
# ours to correct. A name ending in anything else - report.out, results.dat -
# is the user's own choice and is left exactly as written.
OUTPUT_SUFFIX = {"text": ".txt", "json": ".json", "markdown": ".md"}
ACCEPTED_SUFFIXES = {
    "text": {".txt", ".text"},
    "json": {".json"},
    "markdown": {".md", ".markdown"},
}
REPORT_SUFFIXES = {s for group in ACCEPTED_SUFFIXES.values() for s in group}


@dataclass
class Job:
    include: List[str] = field(default_factory=list)
    extensions: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    min_lines: int = 4
    window_lines: int = 40
    window_overlap: int = 8
    threshold: float = 0.9
    min_structure: float = 0.5  # AST-shape agreement required alongside the embedding score
    filter_callers: bool = True  # drop pairs where one unit calls the other
    idf_weighting: bool = True  # discount structure the whole codebase shares
    min_distinct: float = 0.10  # how informative the shared structure must be
    report: str = "pairwise"
    top_n: Optional[int] = None
    cache: bool = True
    cache_file: Optional[str] = None
    output: str = "text"
    output_file: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "Job":
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"could not read job config {path}: {exc}")
        known = {f.name for f in dataclasses.fields(cls)}

        retired = sorted(set(data) & set(RETIRED_KEYS))
        if retired:
            print(
                f"note: {path} still carries API settings, which now live in "
                f".llmcli/config.json and are ignored here:", file=sys.stderr,
            )
            for key in retired:
                print(f"  {key:<12} -> {RETIRED_KEYS[key]}", file=sys.stderr)
            print("  run `llmcli dupcheck update` to drop them", file=sys.stderr)

        unknown = set(data) - known - set(RETIRED_KEYS)
        if unknown:
            print(f"warning: ignoring unknown config keys: {', '.join(sorted(unknown))}", file=sys.stderr)
        return cls(**{k: v for k, v in data.items() if k in known})

    def report_path(self) -> Optional[Path]:
        """Where the report actually gets written.

        `output` can be overridden per run (--output json), so a stored
        `output_file` of duplicate_report.md would otherwise hand json or plain
        text a .md name. The extension follows the format.
        """
        if not self.output_file:
            return None
        path = Path(self.output_file)
        suffix = path.suffix.lower()
        if suffix in ACCEPTED_SUFFIXES.get(self.output, set()):
            return path
        if suffix in REPORT_SUFFIXES:
            return path.with_suffix(OUTPUT_SUFFIX[self.output])
        return path

    def validate(self) -> None:
        if self.report not in ("pairwise", "clusters"):
            sys.exit(f"invalid report mode: {self.report!r} (want 'pairwise' or 'clusters')")
        if self.output not in ("text", "json", "markdown"):
            sys.exit(f"invalid output: {self.output!r} (want 'text', 'json' or 'markdown')")
        if not self.include:
            sys.exit("job config needs at least one path in 'include'")
        if not self.extensions:
            sys.exit("job config needs at least one extension in 'extensions'")


def apply_overrides(job: Job, args: argparse.Namespace) -> None:
    if args.threshold is not None:
        job.threshold = args.threshold
    if args.min_structure is not None:
        job.min_structure = args.min_structure
    if args.no_filter_callers:
        job.filter_callers = False
    if args.no_idf:
        job.idf_weighting = False
    if args.min_distinct is not None:
        job.min_distinct = args.min_distinct
    if args.report is not None:
        job.report = args.report
    if args.output is not None:
        job.output = args.output
    if args.top_n is not None:
        job.top_n = args.top_n
    if args.output_file is not None:
        job.output_file = args.output_file
    if args.no_cache:
        job.cache = False
