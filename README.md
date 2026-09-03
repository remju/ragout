# llmcli

A small toolchain around any OpenAI-compatible LLM endpoint (OpenAI, Together,
Groq, OpenRouter, vLLM, Ollama's `/v1`, LM Studio, ...): chat, document RAG, and
duplicate-code detection, sharing one workspace configuration.

```bash
llmcli chat --rag                  # talk to a model, grounded on your documents
llmcli ask "what's the notice period?"
llmcli ingest ./notes report.pdf   # chunk + embed documents
llmcli dupcheck run job.json       # find near-duplicate code in a source tree
```

## Modules

| module | owns |
|---|---|
| `llmcli-tools` | workspace settings and **all** communication with the models: the client, streaming, embedding batching, the content-hash cache, TLS, vector math |
| `llmcli-corpus` | documents - reading pdf/docx/text, chunking, the per-project embedding store, retrieval |
| `llmcli-chat` | the interactive REPL, optionally grounded on the corpus |
| `llmcli-dupcheck` | source code - tree-sitter extraction, structural corroboration, duplicate reporting |

`tools` is the only module that makes a request. `corpus` and `dupcheck` are
peers on top of it: one processes documents, the other processes source code
(which it parses into an AST first). Neither knows the other exists.

## Install

```bash
pip install .                 # or: pipx install .
pip install '.[all]'          # + pdf/docx reading and tree-sitter parsers
```

Extras are per module: `.[corpus]` pulls `pypdf` and `python-docx`, `.[dupcheck]`
pulls `tree-sitter-language-pack`. `requests` and `numpy` are always installed -
numpy is not optional in practice, because the duplicate scan is O(n²) and takes
~49s for 1500 code units in pure python where numpy takes 0.22s.

On a Debian/Ubuntu system python you'll hit PEP 668, so use `pipx`, a venv, or
`pip install --break-system-packages`.

## Workspaces

State lives in `.llmcli/` folders (`config.json` + `store.db`). Starting at the
cwd, llmcli walks up the directory tree:

- **Settings** are inherited from every ancestor `.llmcli/config.json`, nearest wins.
- **The embedding store** is always the *nearest* `.llmcli/`.

So credentials go in a parent, documents stay per project:

```bash
cd ~/work
llmcli init --url https://api.openai.com/v1 --api-key sk-... --model gpt-4o-mini

cd ~/work/handbook && llmcli init          # no flags: inherits everything, own store.db
cd ~/work/legal    && llmcli init --model claude-sonnet-4-6   # inherits url+key, own model
```

```
~/work/.llmcli/           url + api key + models      (shared)
~/work/handbook/.llmcli/  store.db                    (handbook's docs)
~/work/legal/.llmcli/     store.db + model override   (legal's docs)
```

`init` writes **only the flags you pass**, so inherited values stay linked — rotate the
key in `~/work` and every project below picks it up on the next command. Nothing is copied down.

```bash
llmcli config              # effective settings + where each value came from
llmcli config --no-inherit # ignore parents
llmcli ask -w ~/work/legal "what's the notice period?"   # act on a tree from anywhere
```

Every command except `init` errors out if no workspace is found at or above the cwd,
so you can never ingest into the wrong store by accident.

### Settings

| key | env | default | what it does |
|---|---|---|---|
| `url` | `LLM_API_URL` | — | API base url (required) |
| `api_key` | `LLM_API_KEY` | — | bearer token, also sent as `x-api-key` |
| `model` | `LLM_MODEL` | `gpt-4o-mini` | chat model |
| `embed_model` | `LLM_EMBED_MODEL` | `text-embedding-3-small` | embedding model |
| `ca_bundle` | `LLM_CA_BUNDLE` | — | CA file to trust for TLS |
| `verify_ssl` | `LLM_VERIFY_SSL` | `true` | `false` skips verification entirely |
| `timeout` | `LLM_TIMEOUT` | `300` | seconds per request; `0` waits forever |
| `embed_batch` | `LLM_EMBED_BATCH` | `64` | texts per embedding request |

Resolution order: **CLI flags > env vars > nearest config > ... > farthest config**
(plus `LLMCLI_WORKSPACE` to pick the starting directory). Config files are written
`chmod 600`; add `.llmcli/` to your `.gitignore`.

### Custom CA certificates

Running from a venv gets you PyPI `requests`, which trusts only certifi's public
CA list, so an endpoint signed by a private CA fails with
`CERTIFICATE_VERIFY_FAILED` even though the system python works fine (distro
`requests` uses the system trust store). Point the whole tree at that store once:

```bash
llmcli init --ca-bundle /etc/ssl/certs/ca-certificates.crt
```

or set `"ca_bundle"` in `.llmcli/config.json` directly. It applies to every
command — chat, ingest, ask, dupcheck — because they all share one client. A
certificate failure prints the detected system bundle and the exact flag to copy
rather than a raw traceback. `--insecure` / `"verify_ssl": false` skips
verification entirely.

## Models

```bash
llmcli model                      # current chat + embedding model, and their origin
llmcli model mistral-large        # write to the nearest workspace
llmcli model --here mistral-large # write to this directory instead
llmcli model --embed bge-m3       # change embedding model (then: ingest --force)
llmcli chat --model o3-mini       # one invocation only, config untouched
llmcli ask  --model o3-mini "..."
```

## Chat

```bash
llmcli chat                       # plain streaming chat
llmcli chat --rag                 # every turn is grounded on this project's documents
```

In-REPL: `/rag on|off`, `/sources`, `/system TEXT`, `/reset`, `/save FILE`, `/help`, `/exit`.

## Documents

```bash
llmcli ingest ./notes report.pdf          # folders recurse; unchanged files are skipped
llmcli docs                               # what's indexed in this project
llmcli ask "what did we decide about pricing?" -k 6 --show-context
llmcli forget notes/old                   # substring match on path
```

- **Chunking** — paragraph-aware packing to ~1200 chars with 200-char overlap
  (`--chunk-size`, `--overlap`). Oversized paragraphs are hard-split.
- **Storage** — SQLite in the nearest workspace. Chunks in one table, embeddings as
  packed float32 blobs alongside them. Files are hashed, so re-running `ingest` only
  re-embeds what changed (`--force` overrides).
- **Retrieval** — vectors are L2-normalised at write time, so search is a dot product;
  top-k chunks are packed into a 6000-char context budget with source labels, and the
  model is told to cite filenames and to admit when the context doesn't cover the question.
- **Scale** — brute-force scan, fine to roughly tens of thousands of chunks. Past that,
  swap `Store.search` for FAISS or sqlite-vec; nothing else needs to change.

Embeddings are only comparable within one model. If you change the embedding
model — including inheriting a different one from a parent — re-run
`llmcli ingest --force` over that project's sources.

## dupcheck

Finds duplicate / near-duplicate code across source files, driven by a JSON job
config.

```bash
llmcli dupcheck init job.json --include src --ext .py,.js   # scaffold a job config
llmcli dupcheck run job.json                                 # run it, report to stdout
llmcli dupcheck run job.json --threshold 0.85 --report clusters --output json
llmcli dupcheck update job.json                              # add fields added since
```

Job configs are plain JSON and describe **the analysis only** — `include` (paths
to scan) and `extensions` are the only required fields. Everything else —
`threshold`, `report` (`pairwise` or `clusters`), `output` (`text`/`json`/`markdown`),
`min_lines`, `exclude`, window-fallback sizing, caching — has a sane default and can
be overridden either in the file or with matching CLI flags. `output_file`'s
extension follows the format, so a stored `duplicate_report.md` becomes
`duplicate_report.json` under `--output json`; a name ending in anything that
isn't a report extension (`report.out`) is written exactly as given. The endpoint,
credentials and TLS settings come from `.llmcli/config.json` like every other
command, so a job config carries nothing machine-specific and is safe to commit.
See `llmcli dupcheck --help` for the full field list.

- **Extraction** — function/class bodies come from **tree-sitter**, a real
  parser, via `tree-sitter-language-pack` (100+ languages; the language is
  detected from the file extension). Grammars are downloaded on first use per
  language and cached, so the first run against a new language needs network.
  Anything that can't be parsed — unsupported language, missing grammar, a file
  with no definitions — falls back to fixed-size overlapping line windows, so a
  run always produces units.
- **Why embedding score alone isn't enough** — embeddings measure *topical*
  similarity, so a function and the function that calls it score very highly:
  the caller contains the callee's name and vocabulary. On stdlib `shutil`,
  `copyfile ↔ copy` scores 0.91 and `_copytree ↔ copytree` 0.93, neither of
  which is a duplicate. Thresholds can't fix this — in the bundled test corpus
  the caller/callee pair (0.735) outranks a real duplicate (0.707). So each
  candidate must clear two more gates, both from the parse tree:
  `min_structure` (default 0.5) compares AST node-type bigrams, which stay
  near-identical when a clone is renamed but diverge for a caller; and
  `filter_callers` (default on) drops pairs where one unit references the
  other's name. On `shutil` + `textwrap` this cut 153 raw pairs to 10, keeping
  genuine twins like `_get_gid ↔ _get_uid`. Loosen with `--min-structure 0.35`
  if you'd rather see borderline rewrites, or `--min-structure 0 --no-filter-callers`
  for the raw embedding ranking.
- **House idioms** — a codebase with a consistent style repeats the same shape
  everywhere; in C with error codes, every accessor is `if (bad) return ERR;
  ...; return OK;`. That shape is what *everything* has in common, so it is not
  evidence of duplication. `idf_weighting` (default on, needs ≥25 parsed units)
  weights each AST bigram by how rare it is in this run, so ubiquitous
  structure stops counting. On a [Jacon](https://github.com/remju/Jacon)-style 
  corpus this took a bounds-check accessor pair from 0.69 structural to 0.09
  while leaving a real copy-pasted sort at 1.00. Disable with `--no-idf`.
- **Thin typed wrappers** — the hardest case, because structural similarity is
  a *ratio*: twenty one-line delegations like
  `return get_value(content, TYPE_INT, value);` have byte-identical shape, so
  they score a perfect 1.000 no matter how worthless the shared structure is.
  IDF cannot fix a ratio, so `min_distinct` (default 0.10) is the absolute
  counterpart — the mean rarity of what a pair shares. On a real (toy) C JSON
  library this took the report from 81 pairs to 2, dropping every typed wrapper
  (distinctiveness 0.012–0.032) while keeping the genuinely copy-pasted
  `get_value_by_name ↔ get_value` (0.47) and a duplicated hashmap/hashset
  implementation (0.41). Needs `idf_weighting`, so it is inert below 25 units
  and reports `n/a`. `--min-distinct 0` disables it.
- **Reading a noisy report** — `"report": "clusters"` collapses a family of
  twenty lookalikes into one group instead of 190 pairs, which is usually what
  you want on a first pass over an unfamiliar codebase. Raising `min_lines` is
  the blunter knob if short boilerplate still dominates.
- **Caching** — embeddings are cached by content hash in a small sqlite file
  next to the job config (`<config>.dupcheck_cache.db`, or `cache_file` in the
  config), so re-running after small edits only re-embeds what changed. Worth
  gitignoring, along with `.llmcli/`.
- **Scale** — same brute-force pairwise scan as document retrieval. With numpy
  it's a single matrix multiply (4000 units ≈ 1.3s); without it, the same run is
  minutes. Fine to roughly tens of thousands of units either way; past that,
  swap the scan for FAISS or sqlite-vec.

`dupcheck-notes.md` records how the defaults above were arrived at, with the
measurements behind each one.

## Upgrading from the single-file scripts

- `llmcli.py` and `dupcode.py` are gone; install the package and use `llmcli`.
- `dupcode run job.json` is now `llmcli dupcheck run job.json`.
- API settings left the job config: `url`, `api_key`, `embed_model`, `ca_bundle`,
  `verify_ssl`, `timeout`, `embed_batch` and `workspace` now live in
  `.llmcli/config.json`. Run `llmcli dupcheck update job.json` to strip them; a
  run that still finds them tells you where each one went instead of silently
  ignoring it.
- Existing `<config>.dupcode_cache.db` caches are renamed on first run, so the
  rename costs no re-embedding.
- Document stores are untouched — no re-ingest needed.

## Roadmap

- Add document conversion to markdown using the microsoft/markitdown tool
