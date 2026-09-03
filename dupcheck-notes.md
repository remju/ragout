# dupcheck — design notes

How `llmcli dupcheck` got built, what broke, and why the defaults are what they are.
Every number below was measured, not estimated.

## What it does

Extracts code units from source files, embeds them, and reports pairs that are
likely duplicates. A JSON job config describes the analysis;
`llmcli dupcheck run job.json` executes it.

```bash
llmcli dupcheck init job.json --include src --ext .c   # scaffold
llmcli dupcheck update job.json                        # add fields added since
llmcli dupcheck run job.json                           # run
```

Endpoint, credentials and TLS come from `.llmcli/config.json`, nearest-wins up
the tree, then env vars, then CLI flags - the same resolution every llmcli
command uses. The job config holds no API settings at all, so it can be
committed to a repository; a project already set up with `llmcli init` needs
nothing extra.

## Initial design decisions

| decision | choice |
|---|---|
| extraction | function/class level, line-window fallback |
| credentials | reuse `.llmcli/` workspace |
| report mode | config parameter (`pairwise` / `clusters`) |
| caching | by content hash, SQLite |

## The extraction rewrite

The first implementation used per-language regexes plus brace/indent counting.
It produced two real bugs during testing, and then failed outright on a large
codebase.

**Bug 1 — Allman braces.** The C pattern required `{` on the same line as the
signature. Most C puts it on its own line, so `main.c` matched *nothing* and the
whole file silently fell through to line windows.

**Bug 2 — phantom functions.** `else if (sorted[mid] < target)` parsed as a
function named `if` (`else` read as a return type). Patched with a keyword
denylist — a patch on a patch.

Both were symptoms, not causes: pattern-matching code is not parsing code. Now
**tree-sitter** via `tree-sitter-language-pack` (100+ grammars, language detected
from extension). Rather than a hand-written table per language, definitions are
matched by a generic rule grounded in how tree-sitter names nodes —
`<what>_<form>`: `function_definition`, `method_declaration`, `struct_item`,
`class_specifier` — which excludes lookalikes like `field_declaration` and
`access_specifier`.

Name resolution handles grammar disagreements: C/C++ bury the identifier in a
`declarator` chain, Go puts it in a `type_spec`, JS arrow functions take the name
of whatever binds them.

Verified on a purpose-built nasty C file, all correct: brace inside a string
literal, a commented-out function (correctly *not* extracted), `else if`, K&R
parameter declarations, multi-line signatures. On 120 Python stdlib files:
**4178 units in 0.98s, 97.6% named**, with window fallback firing only on files
that genuinely contain no definitions.

Anything unparseable still falls back to line windows, so a run always produces
units.

## The three false-positive classes

Embedding similarity measures *topical* similarity, not structural equivalence.
Three distinct failure modes surfaced, each needing a different fix.

### 1. Caller and callee

A caller contains its callee's name and vocabulary, so it lands nearby.

This is **not** a threshold problem:

| pair | embedding |
|---|---|
| `validate_total ↔ process_order` (caller/callee) | **0.735** |
| `bubble_sort ↔ sort_ascending` (real duplicate) | 0.707 |

The false positive *outranks* the true duplicate — no threshold separates them.

**Fix:** `filter_callers` (drop pairs where one references the other's name) and
`min_structure` (AST node-type bigram agreement). A renamed clone keeps its shape
(0.937); a caller does not (0.439).

On stdlib `shutil` + `textwrap`: **153 pairs → 10**. Removed `copyfile ↔ copy`
(0.912), `_copytree ↔ copytree` (0.928). Kept genuine twins `_get_gid ↔ _get_uid`.

### 2. House idioms

A codebase with consistent style repeats one shape everywhere. In C with error
codes, every accessor is `if (bad) return ERR; ...; return OK;`. That shape is
what *everything* has in common, so it is not evidence of duplication.

**Fix:** `idf_weighting` — weight each AST bigram by how rare it is in this run,
so ubiquitous structure stops counting.

| pair | structural | after IDF |
|---|---|---|
| bounds-check accessor pair | 0.688 | **0.093** |
| real copy-pasted sort | 1.000 | **1.000** |

Guarded at ≥25 parsed units: document frequency is meaningless across 10
documents, and enabling it on a tiny corpus dropped a *true* duplicate
(0.656 → 0.262). Below that it stays plain.

### 3. Thin typed wrappers — the hard one

```c
return Jacon_get_value_by_name(content, name, JACON_VALUE_INT, value);
```

Twenty of these around one generic function. They have **byte-identical AST
shapes**, so structural similarity — a *ratio* — returns a perfect 1.000 no
matter how worthless the shared structure is. IDF weights numerator and
denominator alike, so it cancels out. **IDF cannot fix a ratio.**

**Fix:** `min_distinct` — the absolute counterpart, the mean *rarity* of what a
pair shares.

| pair | structural | distinctiveness |
|---|---|---|
| `get_int_by_name ↔ get_float_by_name` | 1.000 | **0.032** |
| `exist_string ↔ exist_int` | 1.000 | **0.012** |
| `get_value_by_name ↔ get_value` (real) | 0.753 | **0.473** |

Measured across four corpora, noise landed at 0.011–0.032 and real duplicates at
0.180–0.473 — a 5.6× gap with nothing between. Default 0.10.

On the real C JSON library: **81 pairs → 2**, both worth attention — two
copy-pasted `switch` blocks, and a duplicated hashmap/hashset implementation
(distinctiveness 0.406).

### Combined effect

| corpus | raw pairs | after gates |
|---|---|---|
| C JSON library | 81 | **2** |
| stdlib `shutil` + `textwrap` | 153 | **4** |
| synthetic idiom corpus | 560 | **1** |

All gates report `n/a` and are skipped rather than silently dropping results when
the signal is unavailable (window fallback, or corpus below the IDF threshold).

## Other things fixed

**Interrupted runs lost all work.** `cache.put_many()` ran only after the *entire*
embedding loop finished. Measured: interrupt after 4 of 8 successful batches →
**0 rows cached**. Now each batch commits as it lands.

| step | result |
|---|---|
| interrupt mid-run | 3 of 8 batches survived |
| resume | `3 cached, 5 to embed` — 7.60s ≈ 5 × 1.5s |
| third run | fully cached, 0.00s |

Granularity is per *batch*, so lower `embed_batch` if you expect to interrupt.

**numpy was never installed.** Every early run used the pure-Python vector path.
The comparison is O(n²):

| units | without numpy | with numpy |
|---|---|---|
| 1,500 | 49.2s | 0.22s |
| 4,000 | ~6 min (extrapolated) | 1.31s |

Effectively required, so it is a hard dependency of the package rather than an
optional one.

**TLS failure in a venv.** Not a bad certificate: Debian patches distro
`python3-requests` to use the system CA store, while a venv gets PyPI `requests`
using certifi's public-CA-only list, which has never heard of a private CA.
Added `--ca-bundle` / `--insecure`, and the raw SSL traceback is replaced by a
diagnosis that detects the system bundle and prints the exact flag to copy. Both
now live in `.llmcli/config.json` (`ca_bundle`, `verify_ssl`), set once per tree
with `llmcli init --ca-bundle ...` and shared by every command.

**Timeouts.** Default raised 120s → 300s; `timeout` and `embed_batch` are
workspace config fields with CLI flags. `--timeout 0` waits indefinitely.

**Line-number desync (preemptive).** `str.splitlines()` breaks on `\f`, `\v` and
U+2028; tree-sitter counts rows on `\n` only. Switched to `\n` splitting so
reported line numbers can't drift on files containing form feeds.

## Progress and timing

A live stderr line covers both phases, auto-off when stderr isn't a TTY
(`--no-progress` forces it):

```
⠼ embedding ████████████░░░░░░░░░░░░  50%  4/8   4.4s
```

Percentages are real where genuinely known — per completed batch, per matrix row.
Elapsed seconds are on the line because with the default `embed_batch: 64` a
small job is a *single* request: the bar legitimately can't move until it
returns. Batch size is deliberately **not** shrunk behind the user's back to
manufacture smoother progress.

Embedding and comparison are timed separately, which makes the cache effect
obvious at a glance:

```
analysis took 23.50s total (embedding 23.48s, comparison 0.00s)
```

## Tuning

| symptom | knob |
|---|---|
| real duplicate missing | `--min-distinct 0.05`, then `--min-structure 0.35` |
| still too noisy | raise `min_lines`; short functions have little to duplicate |
| family of 20 lookalikes | `"report": "clusters"` — one group, not 190 pairs |
| want raw ranking | `--min-structure 0 --min-distinct 0 --no-filter-callers` |
| requests timing out | lower `embed_batch` before raising `timeout` |

The tightest margin is above, not below: the lowest measured true positive is
stdlib's 0.130 against a 0.10 default. If a real duplicate disappears, lower
`min_distinct` first.

## Known limits

- Grammars download on first use per language (cached after), so the first run
  against a new language needs network.
- Changing the parser changes unit boundaries, so existing caches re-embed once.
- Brute-force O(n²) scan — fine to roughly tens of thousands of units with numpy.
  Past that, swap for FAISS or sqlite-vec.
- Short functions are inherently hard: there is little to duplicate in 5 lines,
  and embeddings of small snippets cluster tightly regardless of content.

## The v2 split

One 1,200-line script became a module in a toolchain. `llmcli-tools` owns every
request that leaves the machine - the client, batching, the content-hash
embedding cache, vector math, workspace settings - and `llmcli-dupcheck` is one
of two peers built on it, alongside `llmcli-corpus`. One works with documents,
the other with source code it parses into an AST first; neither speaks to an
API directly.

The measurable consequence is what left the job config. `url`, `api_key`,
`embed_model`, `ca_bundle`, `verify_ssl`, `timeout`, `embed_batch` and
`workspace` all moved to `.llmcli/config.json`, which every llmcli command
already read. A job config is now purely *what to scan and how strict to be*,
which makes it safe to commit: no endpoint, no key, no machine-specific CA path.
`llmcli dupcheck update` strips the retired fields from an existing config, and a
run that still finds them says where each one went instead of silently ignoring
it. Caches written as `<config>.dupcode_cache.db` are renamed on first run, so
the rename costs no re-embedding.
