"""llmcli - chat, documents and duplicate-code detection over one endpoint.

Settings live in .llmcli/ folders. Starting at the cwd (or --workspace/-w DIR),
llmcli walks up the tree: settings are inherited from every ancestor with the
nearest winning, while the embedding store is always the nearest .llmcli/. So a
parent directory can hold the url, api key, models and CA bundle once, and each
project subdirectory keeps its own documents.

    ~/work/.llmcli/           url + api key + models      (shared)
    ~/work/handbook/.llmcli/  store.db                    (handbook's docs)
    ~/work/legal/.llmcli/     store.db + model override   (legal's docs)

Commands
    init      create a workspace here; unset values inherit from parents
    config    show the effective settings and where each came from
    model     show or change the model for this directory
    chat      interactive chat REPL (optionally grounded on your documents)
    ask       one-shot question answered from your ingested documents
    ingest    chunk + embed files or directories into the nearest store
    docs      list what has been ingested
    forget    remove documents from the store
    dupcheck  find duplicate / near-duplicate code in a source tree

Commands other than init exit with an error if no workspace is found.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from llmcli.chat import repl as chat_commands
from llmcli.corpus import commands as corpus_commands
from llmcli.dupcheck import commands as dupcheck_commands
from llmcli.tools.args import Parents
from llmcli.tools.workspace import SETTINGS, Config, origin, shown_value


def cmd_init(args, cfg: Config) -> None:
    if cfg.ws.local.joinpath("config.json").exists() and not args.force:
        sys.exit(f"workspace already exists at {cfg.ws.local} (use --force to overwrite)")

    # write only what was passed, so inherited settings stay live and keep
    # tracking the parent if it changes later
    updates = {s.name: getattr(args, s.name, None) for s in SETTINGS}
    if not any(v is not None for v in updates.values()) and not cfg.ws.exists():
        sys.exit("init needs at least --url (e.g. --url https://api.openai.com/v1)")
    if not cfg.url:
        sys.exit("no --url given and none inherited from a parent directory")

    path = cfg.save_local(updates)
    inherited = [s.name for s in SETTINGS if updates.get(s.name) is None]
    cfg = Config.load(args)  # reload so provenance reflects the new file

    print(f"initialised workspace {cfg.ws.start}")
    print(f"  config      {path}")
    print(f"  database    {cfg.ws.db_path}")
    for s in SETTINGS:
        print(f"  {s.name:<11} {shown_value(cfg, s.name)} {origin(cfg, s.name)}")
    if inherited and len(cfg.ws.chain) > 1:
        print("\nsettings not written here stay linked to the parent workspace")


def cmd_config(args, cfg: Config) -> None:
    cfg.require()
    print(f"workspace   {cfg.ws.start}")
    print(f"database    {cfg.ws.db_path}")
    for s in SETTINGS:
        print(f"{s.name:<11} {shown_value(cfg, s.name)} {origin(cfg, s.name)}")
    print("\nconfig files, nearest first:")
    for folder in cfg.ws.chain:
        print(f"  {folder / 'config.json'}")


def cmd_model(args, cfg: Config) -> None:
    cfg.require()
    if not args.name:
        print(f"chat model   {cfg.model} {origin(cfg, 'model')}")
        print(f"embed model  {cfg.embed_model} {origin(cfg, 'embed_model')}")
        return

    field = "embed_model" if args.embed else "model"
    label = "embed model" if args.embed else "chat model"
    # default target is the config actually in use; --here pins it to this dir
    path = cfg.save_local({field: args.name}) if args.here else cfg.save_active({field: args.name})
    print(f"{label} -> {args.name}")
    print(f"written to {path}")
    if args.embed:
        print("note: existing embeddings were built with the old model - "
              "re-run `llmcli ingest --force` to rebuild them")


def build_parser() -> argparse.ArgumentParser:
    parents = Parents.build()
    parser = argparse.ArgumentParser(
        prog="llmcli", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", parents=[parents.common],
                       help="create a workspace here; unset values inherit from parents")
    p.add_argument("--force", action="store_true", help="overwrite an existing workspace")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("config", parents=[parents.common],
                       help="show the effective settings and where each came from")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("model", parents=[parents.common],
                       help="show or change the model for this directory")
    p.add_argument("name", nargs="?", help="model to switch to; omit to print the current one")
    p.add_argument("--embed", action="store_true", help="set the embedding model instead")
    p.add_argument("--here", action="store_true",
                   help="write to this directory instead of the nearest workspace")
    p.set_defaults(func=cmd_model)

    # each module owns its own commands
    chat_commands.register(sub, parents)
    corpus_commands.register(sub, parents)
    dupcheck_commands.register(sub, parents)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
