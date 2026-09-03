# Embedding & storage scale review — 2026-09-03

A review of how `corpus` (documents) and `dupcheck` (code) embed and store
vectors, across four scale points, and the two fixes that came out of it. This
is a decision record, not a spec — it explains what was found, what was
chosen, and what was deliberately left alone.

## The question

How does the current embedding/storage strategy behave for:

- a small document (10-20 pages)
- a small codebase (1-5k LOC)
- a big document (400-500+ pages)
- a big codebase (50k+ LOC)

and where is it not optimal for a given size.

## What the two pipelines actually do

`corpus` and `dupcheck` are structurally different, by design, not by
accident:

- **Documents** (`corpus/documents.py`) — paragraph-aware packing into fixed
  1200-char chunks, 200-char overlap. No language awareness, no token
  counting — a char-count proxy for tokens.
- **Code** (`dupcheck/units.py`) — real AST extraction via tree-sitter
  (functions/classes), falling back to 40-line windows only when a file can't
  be parsed. See [`dupcheck-notes.md`](dupcheck-notes.md) for how that
  extraction was arrived at.

Both hand their text to the same `tools/embed.py::embed_texts`, batched
(`embed_batch`, default 64) and cached by content hash — except `corpus
ingest`, which calls `client.embed()` directly and so skips that cache (still
true after this review; see **Left alone** below).

## Findings by scale point

| scale | verdict | why |
|---|---|---|
| small document (10-20p) | fine | a few dozen chunks, one or two embed batches, trivial store scan |
| small codebase (1-5k LOC) | fine | tree-sitter is fast (~4200 units/s per `dupcheck-notes.md`); similarity matrix is tiny |
| big document (400-500p) | inefficient, not broken | whole file loaded eagerly; `corpus ingest` bypasses the embedding cache so an interrupted ingest re-embeds from scratch; `store.search` rescans the whole store per query with no index |
| big codebase (50k+ LOC) | genuine ceiling | `find_pairs` builds a dense `n × n` similarity matrix in one allocation — ~10GB at 50k units, 1536-dim vectors. `dupcheck-notes.md` already named this ("fine to tens of thousands... past that, swap for FAISS or sqlite-vec") but it had never been acted on |

The two real problems were: **`find_pairs`'s unbounded matrix allocation**
(a hard ceiling, not just a slowdown), and **`Store.search`'s brute-force
Python scan with no vector index** (a slowdown, not a ceiling, but a
compounding one as a corpus grows).

## What we chose

### 1. A memory bound for `dupcheck`, not a hard cutoff

Considered: refuse to run past some unit count. Rejected — it turns a large
but survivable codebase into an outright failure instead of degrading
gracefully, and picking a safe cutoff number is guesswork per-machine.

Chosen: compute the similarity matrix in **row blocks** sized to stay under a
memory budget, instead of one `n × n` allocation. Same exact brute-force
result — this is not an approximation — just bounded peak memory. Below the
budget it's a single block, i.e. today's exact code path.

Exposed as a workspace setting, `dupcheck_max_memory_mb` (default `2048`,
`0` = unlimited), resolved the same way every other setting is — config file,
`LLM_DUPCHECK_MAX_MEMORY_MB`, nearest-wins up the tree.

Verified: blocked and unblocked runs return the identical set of pairs (at
n=120 and n=1000 units); scores differ by ~1e-7, which is float32 BLAS
rounding noise from a different summation order, not a behavior change. A
small codebase's matrix already fits in one block, so nothing changes for it.

### 2. sqlite-vec for the corpus store, with a graceful fallback

Considered and set aside:

| option | why not (for now) |
|---|---|
| FAISS | index-only — still needs SQLite alongside for metadata, two systems to keep in sync for no real gain over sqlite-vec here |
| LanceDB | vector-native and well-suited, but a bigger rewrite than the problem currently justifies |
| Qdrant / Weaviate / Milvus / pgvector | server-based — wrong shape for a local, zero-ops, single-file-per-workspace CLI tool |
| True ANN (HNSW) indexing | has real overhead and approximation error at small scale (build cost, recall < 100%); would need to be threshold-gated. Moot for now anyway — `sqlite-vec`'s `vec0` doesn't ship an ANN index yet, only exact brute-force in C |

Chosen: `sqlite-vec`, added to the `.[corpus]` extra. `Store` now keeps
vectors in a `vec0` virtual table and lets KNN run in C over the whole table,
instead of unpacking every row into a python float32 array and scoring it in
a python loop. This does **not** change the algorithmic complexity — `vec0`'s
default index is still an exact, linear scan — but it removes the Python-level
per-row overhead that was most of the actual cost, and it's an honest stepping
stone rather than a claim of sub-linear search.

Backend selection happens once, at store creation, based on whether the
extension loads at that moment; an existing `store.db` always keeps the
schema it was built with. Nothing changes for a project that already has a
store, whether or not `sqlite-vec` gets installed or removed later — no
silent migration, no forced re-ingest. A brand-new store on a machine without
`sqlite-vec` installed still works exactly as before (legacy BLOB + Python
scan), matching the existing numpy/no-numpy fallback philosophy in
`tools/vectors.py`.

Verified: `vec0`-mode search matches a brute-force reference exactly;
`replace_document`/`forget` don't orphan vector rows; a vector-dimension
mismatch (e.g. after switching embedding models) raises a clear error instead
of crashing; an existing legacy store stays on the legacy path even after
`sqlite-vec` becomes available in the environment; a `vec0`-mode store reopens
correctly in a fresh process without re-running `CREATE VIRTUAL TABLE`.

## Left alone (identified, not fixed this pass)

- **`corpus ingest` bypasses the embedding cache.** It calls `client.embed()`
  directly rather than `embed_texts()`/`EmbeddingCache`, so an interrupted
  ingest on a large document re-embeds everything on retry. `dupcheck` already
  gets this for free via `embed_texts`; `corpus ingest` doesn't. Worth fixing,
  not part of this pass.
- **Chunking is char-count based, not token-aware.** 1200 chars is a rough
  proxy for tokens — fine for prose, imprecise for token-dense content.
- **True ANN indexing**, for either the corpus store past "tens of thousands
  of chunks" or `dupcheck`'s candidate generation past "tens of thousands of
  units" — still an open item; would mean FAISS, a future `sqlite-vec`
  release, or a different engine, and (for `dupcheck` specifically) a real
  decision about trading exact results for approximate ones.
- **No CLI flag for `dupcheck_max_memory_mb`.** Config/env only, matching how
  most settings resolve; only `embed_batch` gets a per-command override flag
  (`--batch` on `dupcheck run`) today. Easy to add the same way if wanted.

## Files touched

- [`llmcli/dupcheck/scan.py`](../llmcli/dupcheck/scan.py) — blocked similarity matrix
- [`llmcli/tools/workspace.py`](../llmcli/tools/workspace.py) — `dupcheck_max_memory_mb` setting
- [`llmcli/dupcheck/commands.py`](../llmcli/dupcheck/commands.py) — wires the setting into `find_pairs`
- [`llmcli/corpus/store.py`](../llmcli/corpus/store.py) — `sqlite-vec` backend + legacy fallback
- [`pyproject.toml`](../pyproject.toml) — `sqlite-vec` added to the `corpus`/`all` extras
- [`README.md`](../README.md) — settings table, install section, Documents/dupcheck scale notes
