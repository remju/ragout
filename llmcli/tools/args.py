"""Argument groups shared by every command.

Anything that overrides a setting from `.llmcli/config.json` for one
invocation belongs here, so the flags stay identical across the toolchain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


def common_parser() -> argparse.ArgumentParser:
    """Settings overrides + workspace selection. Defaults are None so that
    `Config.load` can tell 'not passed' from 'passed as false/zero'."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--url", help="API base url, e.g. https://api.openai.com/v1")
    p.add_argument("--api-key", dest="api_key")
    p.add_argument("--model", help="chat model for this invocation")
    p.add_argument("--embed-model", dest="embed_model")
    p.add_argument("--ca-bundle", dest="ca_bundle",
                   help="CA file to trust for TLS (normally set once in config.json)")
    p.add_argument("--insecure", dest="verify_ssl", action="store_const", const=False,
                   default=None, help="skip TLS verification entirely")
    p.add_argument("--timeout", type=int, default=None,
                   help="per-request seconds before giving up; 0 = wait forever")
    p.add_argument("--system-prompt", dest="system_prompt",
                   help="default system prompt for chat/ask (normally set once in config.json)")
    p.add_argument("-w", "--workspace", metavar="DIR",
                   help="directory to resolve the workspace from (default: cwd)")
    p.add_argument("--no-inherit", action="store_true",
                   help="ignore config in parent directories")
    return p


def generation_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--no-stream", action="store_true")
    return p


def retrieval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-k", "--top-k", type=int, default=4)
    p.add_argument("--min-score", type=float, default=0.0)
    return p


@dataclass
class Parents:
    """The parent parsers handed to each module's `register`."""

    common: argparse.ArgumentParser
    gen: argparse.ArgumentParser
    ret: argparse.ArgumentParser

    @classmethod
    def build(cls) -> "Parents":
        return cls(common_parser(), generation_parser(), retrieval_parser())
