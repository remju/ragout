"""Workspace discovery and layered settings.

Every module in the toolchain reads its endpoint, credentials and TLS
settings from here, so a tree that has been `llmcli init`-ed needs no
per-command and no per-job configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

WORKSPACE_DIRNAME = ".llmcli"


@dataclass(frozen=True)
class Setting:
    """One configurable value: its config.json key, env var and type."""

    name: str
    env: str
    kind: str  # "str" | "bool" | "int"
    default: Any
    secret: bool = False

    def present(self, value: Any) -> bool:
        """True when a config file actually specifies this setting.

        Strings use truthiness (an empty url means 'unset'), but booleans and
        numbers must be type-checked or `"verify_ssl": false` would read as
        absent and silently keep verification on.
        """
        if self.kind == "bool":
            return isinstance(value, bool)
        if self.kind == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        return bool(value)

    def parse_env(self, raw: str) -> Any:
        if self.kind == "bool":
            return raw.strip().lower() not in ("0", "false", "no", "off", "")
        if self.kind == "int":
            try:
                return int(raw)
            except ValueError:
                sys.exit(f"${self.env} must be a whole number, got {raw!r}")
        return raw


SETTINGS = (
    Setting("url", "LLM_API_URL", "str", ""),
    Setting("api_key", "LLM_API_KEY", "str", "", secret=True),
    Setting("model", "LLM_MODEL", "str", "gpt-4o-mini"),
    Setting("embed_model", "LLM_EMBED_MODEL", "str", "text-embedding-3-small"),
    # TLS: a venv installs requests from PyPI, which trusts only certifi's
    # public CA list, so an endpoint signed by a private CA needs the system
    # trust store named here once per tree
    Setting("ca_bundle", "LLM_CA_BUNDLE", "str", ""),
    Setting("verify_ssl", "LLM_VERIFY_SSL", "bool", True),
    Setting("timeout", "LLM_TIMEOUT", "int", 300),
    Setting("embed_batch", "LLM_EMBED_BATCH", "int", 64),
    Setting("system_prompt", "LLM_SYSTEM_PROMPT", "str", ""),
)

SYSTEM_CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # debian/ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",  # fedora/rhel
    "/etc/ssl/cert.pem",  # alpine, macos ports
)


class Workspace:
    """Locates the .llmcli/ folders that apply to a directory.

    Resolution starts at --workspace/-w DIR, else $LLMCLI_WORKSPACE, else the
    cwd, and then walks up to the filesystem root collecting every .llmcli/
    it finds. Settings are inherited down the chain with the nearest file
    winning, while the embedding store is always the *nearest* .llmcli/ - so a
    parent directory can hold the url, key and models once, and each project
    subdirectory keeps its own documents:

        ~/work/.llmcli/          url + api key + models   (shared)
        ~/work/handbook/.llmcli/ store.db                 (handbook's docs)
        ~/work/legal/.llmcli/    store.db + model override (legal's docs)

    A directory with its own .llmcli/ and no ancestors behaves exactly as
    before: config and embeddings side by side in one place.
    """

    def __init__(self, start: Path, chain: List[Path]):
        self.start = start
        self.chain = chain  # existing .llmcli dirs, nearest first

    @property
    def local(self) -> Path:
        """Where `init` would create a workspace."""
        return self.start / WORKSPACE_DIRNAME

    @property
    def active(self) -> Optional[Path]:
        """Nearest existing .llmcli - owns the embedding store."""
        return self.chain[0] if self.chain else None

    @property
    def db_path(self) -> Path:
        return (self.active or self.local) / "store.db"

    def exists(self) -> bool:
        return bool(self.chain)

    @classmethod
    def resolve(cls, args: argparse.Namespace) -> "Workspace":
        raw = (
            getattr(args, "workspace", None)
            or os.environ.get("LLMCLI_WORKSPACE")
            or Path.cwd()
        )
        start = Path(raw).expanduser().resolve()
        chain = [
            d / WORKSPACE_DIRNAME
            for d in (start, *start.parents)
            if (d / WORKSPACE_DIRNAME / "config.json").exists()
        ]
        if getattr(args, "no_inherit", False):
            chain = chain[:1]
        return cls(start, chain)


@dataclass
class Config:
    url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    ca_bundle: str = ""
    verify_ssl: bool = True
    timeout: int = 300
    embed_batch: int = 64
    system_prompt: str = ""
    ws: Optional[Workspace] = None

    FIELDS = tuple(s.name for s in SETTINGS)

    def __post_init__(self):
        self.sources: dict = {}  # field -> where the effective value came from

    @property
    def db(self) -> str:
        return str(self.ws.db_path)

    @property
    def request_timeout(self) -> Optional[int]:
        """None means 'wait forever', which is what requests wants."""
        return self.timeout if self.timeout and self.timeout > 0 else None

    @classmethod
    def load(cls, args: argparse.Namespace) -> "Config":
        ws = Workspace.resolve(args)
        cfg = cls(ws=ws)
        for s in SETTINGS:
            cfg.sources[s.name] = "default" if s.default != "" else None

        # farthest ancestor first, so nearer files override
        for folder in reversed(ws.chain):
            path = folder / "config.json"
            try:
                data = json.loads(path.read_text() or "{}")
            except (OSError, json.JSONDecodeError):
                print(f"warning: could not parse {path}", file=sys.stderr)
                continue
            for s in SETTINGS:
                if s.name in data and s.present(data[s.name]):
                    setattr(cfg, s.name, data[s.name])
                    cfg.sources[s.name] = str(path)

        for s in SETTINGS:
            raw = os.environ.get(s.env)
            if raw:
                setattr(cfg, s.name, s.parse_env(raw))
                cfg.sources[s.name] = f"${s.env}"

        for s in SETTINGS:
            val = getattr(args, s.name, None)
            if val is not None:
                setattr(cfg, s.name, val)
                cfg.sources[s.name] = "--flag"
        return cfg

    # -- writing ----------------------------------------------------------

    def _write(self, folder: Path, updates: dict) -> Path:
        """Merge `updates` into folder/config.json, leaving other keys alone."""
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.json"
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                pass
        # `is not None` rather than truthiness, so `verify_ssl: false` sticks
        data.update({k: v for k, v in updates.items() if v is not None and v != ""})
        path.write_text(json.dumps(data, indent=2))
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path

    def save_local(self, updates: dict) -> Path:
        """Write to the workspace being created/used here."""
        return self._write(self.ws.local, updates)

    def save_active(self, updates: dict) -> Path:
        """Write to the nearest existing workspace."""
        return self._write(self.ws.active or self.ws.local, updates)

    def require(self) -> None:
        """Every command except init needs a workspace here or above."""
        if not self.ws.exists():
            sys.exit(
                f"no llmcli workspace at or above {self.ws.start}\n"
                f"  expected {self.ws.local / 'config.json'} (or one in a parent directory)\n"
                f"  run `llmcli init --url ... --model ...` here, "
                f"or point at one with --workspace /path/to/dir"
            )
        if not self.url:
            sys.exit(
                f"no API url for {self.ws.start} - run `llmcli init --url ...` "
                f"here or in a parent directory"
            )


def origin(cfg: Config, field: str) -> str:
    """Human label for where a setting came from."""
    src = cfg.sources.get(field)
    if not src or src in ("default", "--flag"):
        return {"default": "(default)", "--flag": "(this command)"}.get(src, "")
    if src.startswith("$"):
        return f"({src})"
    folder = Path(src).parent
    if cfg.ws.active and folder == cfg.ws.active:
        return "(here)"
    return f"(inherited from {folder.parent})"


def shown_value(cfg: Config, field: str) -> str:
    """Display form of a setting - secrets are never printed."""
    setting = next(s for s in SETTINGS if s.name == field)
    value = getattr(cfg, field)
    if setting.secret:
        return "set" if value else "(unset)"
    if setting.kind == "str" and not value:
        return "(unset)"
    if setting.kind == "bool":
        return "true" if value else "false"
    return str(value)
