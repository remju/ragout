# llmcli

A single-file chat + RAG client for any OpenAI-compatible LLM endpoint
(OpenAI, Together, Groq, OpenRouter, vLLM, Ollama's `/v1`, LM Studio, ...).

Install once, system-wide. Configure credentials once per tree, keep embeddings per project.

## Install

```bash
pip install requests numpy        # pypdf / python-docx only if you need them
install -m 755 llmcli.py /usr/local/bin/llmcli
```

## Workspaces

State lives in `.llmcli/` folders (`config.json` + `store.db`). Starting at the cwd,
llmcli walks up the directory tree:

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

A single directory with its own `.llmcli/` and no ancestors behaves exactly as before:
config and embeddings side by side.

```bash
llmcli config              # effective settings + where each value came from
llmcli config --no-inherit # ignore parents
llmcli ask -w ~/work/legal "what's the notice period?"   # act on a tree from anywhere
```

Resolution order: **CLI flags > env vars > nearest config > ... > farthest config**.
Env: `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_EMBED_MODEL`, `LLMCLI_WORKSPACE`.
Config files are written `chmod 600`; add `.llmcli/` to your `.gitignore`.

Every command except `init` errors out if no workspace is found at or above the cwd,
so you can never ingest into the wrong store by accident.

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

## RAG

```bash
llmcli ingest ./notes report.pdf          # folders recurse; unchanged files are skipped
llmcli docs                               # what's indexed in this project
llmcli ask "what did we decide about pricing?" -k 6 --show-context
llmcli forget notes/old                   # substring match on path
```

## How it works

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

## Caveat

Embeddings are only comparable within one model. If you change the embedding model —
including inheriting a different one from a parent — re-run `llmcli ingest --force`
over that project's sources.
