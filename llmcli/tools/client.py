"""The one place that talks to a model.

Everything that leaves the machine goes through LLMClient: chat completions,
streaming, embeddings, TLS trust and timeouts. corpus and dupcheck hand it
text and get vectors or replies back; neither of them knows about requests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from llmcli.tools.workspace import SYSTEM_CA_BUNDLES, Config

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install requests")


def ssl_help(exc: Exception) -> str:
    """Certificate failures here are nearly always a venv/certifi mismatch:
    a private CA that the system trust store knows about but certifi does not."""
    lines = [f"TLS verification failed: {exc}", ""]
    found = [p for p in SYSTEM_CA_BUNDLES if Path(p).exists()]
    if found:
        lines += [
            "This usually means the endpoint uses a private/internal CA that your",
            "system trusts but this Python environment does not (a venv installs",
            "requests from PyPI, which trusts only certifi's public CA list).",
            "",
            "Point llmcli at the system trust store, once for the whole tree:",
            f"    llmcli init --ca-bundle {found[0]}",
            f'    or "ca_bundle": "{found[0]}" in .llmcli/config.json',
        ]
    else:
        lines.append(
            "Set the signing CA with `llmcli init --ca-bundle /path/to/ca.crt`."
        )
    lines += ["", "As a last resort, --insecure skips verification entirely."]
    return "\n".join(lines)


class LLMClient:
    """Minimal client for OpenAI-compatible /chat/completions and /embeddings."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = cfg.model
        self.embed_model = cfg.embed_model
        self.timeout = cfg.request_timeout  # None = wait forever
        self.batch = cfg.embed_batch
        self.session = requests.Session()

        if not cfg.verify_ssl:
            self.session.verify = False
            print("warning: TLS verification disabled", file=sys.stderr)
        elif cfg.ca_bundle:
            path = str(Path(cfg.ca_bundle).expanduser())
            if not Path(path).exists():
                sys.exit(f"ca_bundle not found: {path}")
            self.session.verify = path

        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
            headers["x-api-key"] = cfg.api_key  # some gateways want this instead
        self.session.headers.update(headers)

    def _endpoint(self, path: str) -> str:
        base = self.cfg.url.rstrip("/")
        # allow the user to paste a full endpoint url
        for suffix in ("/chat/completions", "/embeddings", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}/{path.lstrip('/')}"

    def _post(self, path: str, payload: dict, stream: bool = False):
        try:
            resp = self.session.post(
                self._endpoint(path), json=payload, timeout=self.timeout, stream=stream
            )
        except requests.exceptions.SSLError as exc:
            raise SystemExit(ssl_help(exc))
        except requests.RequestException as exc:
            raise SystemExit(f"request failed: {exc}")
        if resp.status_code >= 400:
            raise SystemExit(f"API error {resp.status_code}: {resp.text[:500]}")
        return resp

    def chat(self, messages: List[dict], stream: bool = True, **kw) -> str:
        payload = {"model": self.model, "messages": messages, "stream": stream}
        payload.update({k: v for k, v in kw.items() if v is not None})

        if not stream:
            data = self._post("chat/completions", payload).json()
            text = data["choices"][0]["message"]["content"]
            print(text)
            return text

        resp = self._post("chat/completions", payload, stream=True)
        parts: List[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        sys.stdout.write("\n")
        return "".join(parts)

    def embed(
        self,
        texts: Sequence[str],
        batch: Optional[int] = None,
        on_batch: Optional[Callable[[int, List[List[float]]], None]] = None,
    ) -> List[List[float]]:
        batch = batch or self.batch
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            window = list(texts[i : i + batch])
            payload = {"model": self.embed_model, "input": window}
            data = self._post("embeddings", payload).json()
            rows = sorted(data["data"], key=lambda r: r.get("index", 0))
            vectors = [r["embedding"] for r in rows]
            out.extend(vectors)
            # hand the batch over as soon as it lands, so the caller can persist
            # it - an interrupted run then keeps everything embedded so far
            if on_batch:
                on_batch(i, vectors)
        return out
