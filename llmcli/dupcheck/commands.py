"""Commands for duplicate-code detection: init, update, run."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import List

from llmcli.dupcheck.job import RETIRED_KEYS, Job, apply_overrides
from llmcli.dupcheck.report import Timings, render_clusters, render_pairwise
from llmcli.dupcheck.scan import find_clusters, find_pairs
from llmcli.dupcheck.units import CodeUnit, extract_units, iter_source_files
from llmcli.tools.args import Parents
from llmcli.tools.cache import EmbeddingCache
from llmcli.tools.client import LLMClient
from llmcli.tools.embed import embed_texts
from llmcli.tools.workspace import Config


def _cache_path(cfg_path: Path, job: Job) -> Path:
    if job.cache_file:
        return Path(job.cache_file)
    path = cfg_path.with_name(cfg_path.stem + ".dupcheck_cache.db")
    # carry over a cache written under the tool's former name, so the rename
    # doesn't silently cost a full re-embed
    legacy = cfg_path.with_name(cfg_path.stem + ".dupcode_cache.db")
    if legacy.exists() and not path.exists():
        legacy.rename(path)
        print(f"renamed cache {legacy.name} -> {path.name}", file=sys.stderr)
    return path


def cmd_init(args, cfg: Config) -> None:
    path = Path(args.config)
    if path.exists() and not args.force:
        sys.exit(f"{path} already exists (use --force to overwrite)")
    job = Job(
        include=args.include or ["."],
        extensions=[e.strip() for e in (args.ext or ".py").split(",") if e.strip()],
    )
    path.write_text(json.dumps(dataclasses.asdict(job), indent=2) + "\n")
    print(f"wrote {path}")


def cmd_update(args, cfg: Config) -> None:
    """Bring an existing job config up to date: add new fields, drop the API
    settings that moved to .llmcli/config.json."""
    path = Path(args.config)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"could not read {path}: {exc}")

    defaults = dataclasses.asdict(Job())
    added = [k for k in defaults if k not in data]
    removed = sorted(set(data) & set(RETIRED_KEYS))
    if not added and not removed:
        print(f"{path} is up to date - nothing to change")
        return

    for k in added:
        data[k] = defaults[k]
    for k in removed:
        data.pop(k)
    path.write_text(json.dumps(data, indent=2) + "\n")

    if added:
        print(f"added to {path}: {', '.join(added)}")
    if removed:
        print(f"dropped from {path} (now in .llmcli/config.json): {', '.join(removed)}")


def cmd_run(args, cfg: Config) -> None:
    start = time.monotonic()
    cfg.require()

    cfg_path = Path(args.config)
    job = Job.load(cfg_path)
    apply_overrides(job, args)
    job.validate()

    files = iter_source_files(job.include, job.extensions, job.exclude)
    if not files:
        sys.exit("no matching files found")

    units: List[CodeUnit] = []
    for f in files:
        units.extend(extract_units(f, job.min_lines, job.window_lines, job.window_overlap))
    if len(units) < 2:
        sys.exit(f"only {len(units)} code unit(s) extracted - nothing to compare")

    print(f"{len(files)} files, {len(units)} code units", file=sys.stderr)

    client = LLMClient(cfg)
    cache = EmbeddingCache(_cache_path(cfg_path, job)) if job.cache else None

    show_progress = not args.no_progress
    timings = Timings()

    mark = time.monotonic()
    vectors = embed_texts(
        [u.text for u in units], client, cache, progress=show_progress
    )
    timings.embed = time.monotonic() - mark
    if cache:
        cache.close()

    mark = time.monotonic()
    pairs = find_pairs(
        units, vectors, job.threshold, progress=show_progress,
        min_structure=job.min_structure, filter_callers=job.filter_callers,
        idf_weighting=job.idf_weighting, min_distinct=job.min_distinct,
    )
    timings.compare = time.monotonic() - mark

    timings.total = time.monotonic() - start

    if job.report == "clusters":
        clusters = find_clusters(units, pairs)
        report = render_clusters(units, clusters, job.output, job.top_n, timings)
        summary = f"{len(clusters)} cluster(s)"
    else:
        report = render_pairwise(units, pairs, job.output, job.top_n, timings)
        summary = f"{len(pairs)} pair(s)"

    print(f"analysis took {timings.line()}", file=sys.stderr)

    target = job.report_path()
    if target:
        target.write_text(report + "\n")
        print(f"{summary} written to {target}", file=sys.stderr)
    else:
        print(report)


DESCRIPTION = """\
Find duplicate / near-duplicate code across source files.

Source files are split into code units (functions/classes where a tree-sitter
grammar recognizes them, fixed-size line windows otherwise), each unit is
embedded, and units whose cosine similarity clears a threshold - and which also
corroborate structurally - are reported as likely duplicates.

The job config describes the analysis only: what to scan, and how strict to be.
The endpoint, credentials and TLS settings come from .llmcli/config.json like
every other llmcli command.

Job config fields (JSON)
    include        list of files/dirs to scan                    (required)
    extensions     list of extensions to include, e.g. [".py"]   (required)
    exclude        substrings to skip (matched against the path)
    min_lines      skip code units shorter than this many lines   (default 4)
    window_lines   fallback window size when no functions/classes
                   are recognized in a file                       (default 40)
    window_overlap overlap between fallback windows                (default 8)
    threshold      cosine similarity to call something a duplicate (default 0.9)
    min_structure  AST-shape agreement a pair must also reach, 0..1 (default
                   0.5; 0 disables). Embedding similarity alone conflates "does
                   the same thing" with "talks about the same thing", so a
                   caller and its callee score highly; this gate needs the two
                   to have the same shape, not just the same vocabulary
    filter_callers drop pairs where one unit references the other's name, i.e.
                   a call relationship rather than duplication (default true)
    min_distinct   how informative the structure a pair shares must be, 0..1
                   (default 0.10; 0 disables). Structural similarity is a ratio,
                   so thin typed wrappers around one generic function score a
                   perfect 1.0 while sharing nothing but boilerplate; this is
                   the absolute check that keeps those out. Needs idf_weighting
    idf_weighting  discount AST structure that the whole codebase shares, so a
                   house idiom (in C: "if (bad) return ERR; ...; return OK;")
                   stops counting as evidence of duplication (default true;
                   needs at least 25 parsed units to kick in)
    report         "pairwise" or "clusters"                        (default "pairwise")
    top_n          cap on results shown (null = no cap)
    cache          cache embeddings by content hash                (default true)
    cache_file     override cache path (default: <config>.dupcheck_cache.db)
    output         "text", "json" or "markdown"                    (default "text")
    output_file    write the report here instead of stdout; the extension
                   follows `output`, so --output json on a .md name writes .json

Parsing needs tree-sitter grammars: pip install 'llmcli[dupcheck]'. The
language is detected from the file extension and grammars are fetched on first
use, so the very first run against a new language needs network access.
"""


def register(sub, parents: Parents) -> None:
    # the settings flags go on the leaf commands only: repeating them here
    # would let `dupcheck --url X run` be silently reset by the subparser
    dup = sub.add_parser(
        "dupcheck",
        help="find duplicate / near-duplicate code",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    jobs = dup.add_subparsers(dest="dupcheck_command", required=True)

    p = jobs.add_parser("init", parents=[parents.common], help="scaffold a job config file")
    p.add_argument("config")
    p.add_argument("--include", nargs="+", help="paths to scan (default: .)")
    p.add_argument("--ext", help="comma-separated extensions, e.g. .py,.js (default: .py)")
    p.add_argument("--force", action="store_true", help="overwrite an existing config file")
    p.set_defaults(func=cmd_init)

    p = jobs.add_parser("update", parents=[parents.common],
                        help="add missing fields to a job config and drop retired ones")
    p.add_argument("config")
    p.set_defaults(func=cmd_update)

    p = jobs.add_parser("run", parents=[parents.common],
                        help="run a job config and report duplicates")
    p.add_argument("config")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min-structure", dest="min_structure", type=float, default=None,
                   help="AST-shape agreement a pair must also reach, 0..1 (default 0.5; 0 disables)")
    p.add_argument("--no-filter-callers", dest="no_filter_callers", action="store_true",
                   help="keep pairs where one unit calls the other (off by default)")
    p.add_argument("--min-distinct", dest="min_distinct", type=float, default=None,
                   help="how informative the shared structure must be, 0..1 (default 0.10; 0 disables)")
    p.add_argument("--no-idf", dest="no_idf", action="store_true",
                   help="don't discount structure the whole codebase shares")
    p.add_argument("--report", choices=["pairwise", "clusters"], default=None)
    p.add_argument("--output", choices=["text", "json", "markdown"], default=None)
    p.add_argument("--output-file", dest="output_file", default=None)
    p.add_argument("--top-n", dest="top_n", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-progress", action="store_true",
                   help="suppress the progress bar (auto-off when not a TTY)")
    p.add_argument("--batch", dest="embed_batch", type=int, default=None,
                   help="code units per embedding request (default 64)")
    p.set_defaults(func=cmd_run)
