"""Shared plumbing: workspace settings and all communication with the models.

corpus and dupcheck are peers built on top of this - one processes documents,
the other source code - and neither of them speaks to an API directly.
"""

from llmcli.tools.cache import EmbeddingCache
from llmcli.tools.client import LLMClient, ssl_help
from llmcli.tools.embed import embed_texts
from llmcli.tools.progress import Progress
from llmcli.tools.vectors import dot, normalize, np, pack, unpack
from llmcli.tools.workspace import (
    SETTINGS,
    SYSTEM_CA_BUNDLES,
    WORKSPACE_DIRNAME,
    Config,
    Setting,
    Workspace,
    origin,
    shown_value,
)

__all__ = [
    "Config", "Workspace", "Setting", "SETTINGS", "SYSTEM_CA_BUNDLES",
    "WORKSPACE_DIRNAME", "origin", "shown_value",
    "LLMClient", "ssl_help", "EmbeddingCache", "embed_texts", "Progress",
    "normalize", "pack", "unpack", "dot", "np",
]
